#include "../gpu_include/gpu_mlp.h"
#include "../include/mlp.h"
#include "../gpu_include/bench_utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>
#include <cstring>
#include <cstdlib>

struct mlp_time{
    float kernel;
    float transpose;       

    void print(){
        cout << "   Kernel (ms)     : " << kernel << endl;
        cout << "   Transpose (ms)  : " << transpose << endl;
    }

    mlp_time(float _kernel = 0.0f, float _transpose = 0.0f):
        kernel(_kernel),
        transpose(_transpose)
    {}

    void to_JSON(int batch, int params[]){
        int mlp_type      = params[0];
        int stride_val    = params[1];

        cout << "{\n"
            << "\"batch\":" << batch << ",\n"
            << "\"params\": {\n" 
                << "\"stride_val\":" << stride_val << ",\n"
                << "\"mlp_type\":" << mlp_type << "\n"
            << "},\n"
            << "\"time\": {\n" 
                << "\"kernel\":" << kernel << ",\n"
                << "\"transpose\":" << transpose << "\n"
            << "}\n"
            << "}\n";
    }
};

// 0)
mlp_time unfused_mlp(
    cublasLtHandle_t & handle, cudaStream_t & stream,
    u_int batch, u_int tokens, u_int channels,u_int k_channels,
    void * d_workspace,
    void * d_x, void * d_fc1, void * d_h,void * d_b1, void * d_fc2, void * d_b2, 
    void * d_y, int stride_val
){
    //Create the descriptors
    mlp_dimensions dim(batch, tokens, channels, k_channels, channels);
    cublasLt_matmul_desc matmul[2];
    cublasLtMatmulAlgo_t algo[2];
    create_mlp_descriptors(
        handle,
        matmul,
        d_workspace,
        algo,
        dim,
        false
    );
    float avg_ms = time_kernel(WARM_UP, N, stream,[&]() {
        gpu_mlp(
            handle,stream,
            batch,tokens,k_channels,channels,
            matmul, algo, d_workspace,
            d_x, d_fc1, d_h, d_b1, d_fc2, d_b2, d_y, 
            stride_val
        );
    });

    return mlp_time(avg_ms); //No transpose needed for this approach
}

// 1)
mlp_time fused_mlp(
    cublasLtHandle_t & handle, cudaStream_t & stream,
    int batch, int tokens, int channels, int k_channels,
    void * d_workspace,
    void * d_x, void * d_fc1, void * d_h, half * gpu_b1 , void * d_fc2, half * gpu_b2, 
    void * d_y
){
    size_t input_elements_n = batch * tokens * channels;
    size_t hidden_elements_n = batch * tokens * k_channels;

    // -bias matrix (done only once)
    vector<half> b1_gpu_mtx(hidden_elements_n); 
    vector<half> b2_gpu_mtx(input_elements_n);   
    bias_matrix(gpu_b1, b1_gpu_mtx.data(), k_channels, batch*tokens);
    bias_matrix(gpu_b2, b2_gpu_mtx.data(), channels, batch*tokens);

    void * d_b1_mtx, * d_b2_mtx;
    CUDA_CHECK(cudaMalloc(&d_b1_mtx, sizeof(half) * hidden_elements_n));
    CUDA_CHECK(cudaMalloc(&d_b2_mtx, sizeof(half) * input_elements_n));
    CUDA_CHECK(cudaMemcpy(d_b1_mtx, b1_gpu_mtx.data(), sizeof(half) * hidden_elements_n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2_mtx, b2_gpu_mtx.data(), sizeof(half) * input_elements_n, cudaMemcpyHostToDevice));

    // -descriptors creation(done only once)
    cublasLt_matmul_desc matmul[2];
    cublasLtMatmulAlgo_t algo[2];
    mlp_dimensions dim(batch, tokens, channels, k_channels, channels);

    create_mlp_descriptors(
        handle,
        matmul,
        d_workspace,
        algo,
        dim,
        true
    );

    cublasLtMatrixLayout_t mlp_out_desc, res_in_desc;
    cublasLtMatrixTransformDesc_t transposeDesc;
    cublasOperation_t op = CUBLAS_OP_T;
    CUBLAS_CHECK(cublasLtMatrixTransformDescCreate(&transposeDesc, CUDA_R_32F));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&mlp_out_desc, CUDA_R_16F, /*rows*/batch*tokens, /*cols*/channels, /*ld*/batch*tokens));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&res_in_desc, CUDA_R_16F, /*rows*/channels, /*cols*/batch*tokens, /*ld*/channels));
    CUBLAS_CHECK(cublasLtMatrixTransformDescSetAttribute(
        transposeDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &op, sizeof(op)
    ));

    // -kernel
    float avg_ms = time_kernel(WARM_UP, N, stream, [&]() {
        fused_gpu_mlp(
            handle,
            stream,
            matmul, algo, 
            d_workspace,
            d_x, d_fc1, d_h, d_b1_mtx, d_fc2, d_b2_mtx, d_y
        );
    });

    /* - This memcopy isn't present in Vit so no time taken as postprocessing - */
    CUDA_CHECK(cudaMemcpy(d_h, d_y, sizeof(half) * input_elements_n, cudaMemcpyDeviceToDevice));
    float mlp_alpha = 1.0f, mlp_beta = 0.0f;
    float postprocess = time_kernel(WARM_UP, N, stream, [&]() {
    // -transpose
        cublasLtMatrixTransform(
            handle, transposeDesc,
            &mlp_alpha, d_h, mlp_out_desc,
            &mlp_beta, nullptr, nullptr,
            d_y, res_in_desc, stream
        );
    });

    return mlp_time(avg_ms, postprocess); // Need to transpose the result
    
}

void single_run(
    cublasLtHandle_t & handle, cudaStream_t & stream,
    u_int batch, u_int tokens, u_int channels,u_int k_channels,
    void * d_workspace,
    void * d_x, void * d_fc1, void * d_h,void * d_b1, void * d_fc2, void * d_b2,
    half * gpu_b1 , half * gpu_b2, 
    void * d_y, int stride_val, bool kernel_type
){
    if(kernel_type){
        size_t input_elements_n = batch * tokens * channels;
        size_t hidden_elements_n = batch * tokens * k_channels;

        // -bias matrix (done only once)
        vector<half> b1_gpu_mtx(hidden_elements_n); 
        vector<half> b2_gpu_mtx(input_elements_n);   
        bias_matrix(gpu_b1, b1_gpu_mtx.data(), k_channels, batch*tokens);
        bias_matrix(gpu_b2, b2_gpu_mtx.data(), channels, batch*tokens);

        void * d_b1_mtx, * d_b2_mtx;
        CUDA_CHECK(cudaMalloc(&d_b1_mtx, sizeof(half) * hidden_elements_n));
        CUDA_CHECK(cudaMalloc(&d_b2_mtx, sizeof(half) * input_elements_n));
        CUDA_CHECK(cudaMemcpy(d_b1_mtx, b1_gpu_mtx.data(), sizeof(half) * hidden_elements_n, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b2_mtx, b2_gpu_mtx.data(), sizeof(half) * input_elements_n, cudaMemcpyHostToDevice));

        // -descriptors creation(done only once)
        cublasLt_matmul_desc matmul[2];
        cublasLtMatmulAlgo_t algo[2];
        mlp_dimensions dim(batch, tokens, channels, k_channels, channels);

        create_mlp_descriptors(
            handle,
            matmul,
            d_workspace,
            algo,
            dim,
            true
        );

        cublasLtMatrixLayout_t mlp_out_desc, res_in_desc;
        cublasLtMatrixTransformDesc_t transposeDesc;
        cublasOperation_t op = CUBLAS_OP_T;
        CUBLAS_CHECK(cublasLtMatrixTransformDescCreate(&transposeDesc, CUDA_R_32F));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&mlp_out_desc, CUDA_R_16F, /*rows*/batch*tokens, /*cols*/channels, /*ld*/batch*tokens));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&res_in_desc, CUDA_R_16F, /*rows*/channels, /*cols*/batch*tokens, /*ld*/channels));
        CUBLAS_CHECK(cublasLtMatrixTransformDescSetAttribute(
            transposeDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &op, sizeof(op)
        ));

        // -kernel
        fused_gpu_mlp(
            handle,
            stream,
            matmul, algo, 
            d_workspace,
            d_x, d_fc1, d_h, d_b1_mtx, d_fc2, d_b2_mtx, d_y
        );

        /* - This memcopy isn't present in Vit so no time taken as postprocessing - */
        CUDA_CHECK(cudaMemcpy(d_h, d_y, sizeof(half) * input_elements_n, cudaMemcpyDeviceToDevice));
        float mlp_alpha = 1.0f, mlp_beta = 0.0f;
        
            // -transpose
        cublasLtMatrixTransform(
            handle, transposeDesc,
            &mlp_alpha, d_h, mlp_out_desc,
            &mlp_beta, nullptr, nullptr,
            d_y, res_in_desc, stream
        );
    }
    else {
        //Create the descriptors
        mlp_dimensions dim(batch, tokens, channels, k_channels, channels);
        cublasLt_matmul_desc matmul[2];
        cublasLtMatmulAlgo_t algo[2];
        create_mlp_descriptors(
            handle,
            matmul,
            d_workspace,
            algo,
            dim,
            false
        );
        
        gpu_mlp(
            handle,stream,
            batch,tokens,k_channels,channels,
            matmul, algo, d_workspace,
            d_x, d_fc1, d_h, d_b1, d_fc2, d_b2, d_y, 
            stride_val
        );
    } 
    
}

int main(int argc, char** argv)
{
    int kernel_id           = get_arg(argc, argv, "--kernel", 0);
    int batch               = get_arg(argc, argv, "--batch", 32);
    int tokens              = get_arg(argc, argv, "--tokens", 197);
    int embeddings          = get_arg(argc, argv, "--embeddings", 768);
    int hidden_channels     = get_arg(argc, argv, "--hidden_channels", 3072);
    int stride_val          = get_arg(argc, argv, "--stride", 2);

    cout << "LayerNorm Benchmark\n"
              << " batch_size:          " << batch << "\n"
              << " tokens:              " << tokens          << "\n"
              << " embeddings:          " << embeddings      << "\n"
              << " hidden_channels:     " << hidden_channels << "\n"
              << " stride:              " << stride_val << "\n"
              << " warmup_iters:        " << WARM_UP << "\n"
              << " timed_iters:         " << N << "\n";

    
    // -  Memory allocation
    size_t elements_n = batch * tokens * embeddings;
    size_t hidden_elements_n = batch * tokens * hidden_channels;
    size_t fc_matrix_n = embeddings * hidden_channels;

    size_t total_bytes = elements_n * sizeof(half);
    size_t total_hidden_bytes = hidden_elements_n * sizeof(half);
    size_t fc_matrix_bytes = fc_matrix_n * sizeof(half);
    size_t hidden_bytes = hidden_channels * sizeof(half);
    size_t embeddings_bytes = embeddings * sizeof(half);

    vector<float> h_input(elements_n);
    vector<float> h_fc1(fc_matrix_n);    
    vector<float> h_fc2(fc_matrix_n);    
    vector<float> h_b1 (hidden_channels);
    vector<float> h_b2 (embeddings);
    vector<half> gpu_input(elements_n);
    vector<half> gpu_output(elements_n);
    vector<half> gpu_fc1(fc_matrix_n);    
    vector<half> gpu_fc2(fc_matrix_n);    
    vector<half> gpu_b1(hidden_channels);
    vector<half> gpu_b2(embeddings);

    random_device rd;          
    mt19937 gen(rd());         
    uniform_real_distribution<float> dist(-1.0f, 1.0f);

    size_t loop_range = max(fc_matrix_n, elements_n);
    for(size_t i = 0; i < loop_range; i++){
        if(i < hidden_channels){
            h_b1[i] = dist(gen);
        }
        if(i < embeddings){
            h_b2[i] = dist(gen);
        }
        if(i < fc_matrix_n){
            h_fc1[i] = dist(gen);
            h_fc2[i] = dist(gen);
        }
        if(i < elements_n){
            h_input[i] = dist(gen);
        }

    }

    f32_to_f16(h_input.data(), gpu_input.data(), elements_n);
    f32_to_f16(h_fc1.data(), gpu_fc1.data(), fc_matrix_n);
    f32_to_f16(h_fc2.data(), gpu_fc2.data(), fc_matrix_n);
    f32_to_f16(h_b1 .data(), gpu_b1.data(), hidden_channels);
    f32_to_f16(h_b2 .data(), gpu_b2.data(), embeddings);
    

    void *d_x = nullptr, *d_h = nullptr, *d_y = nullptr;
    void *d_fc1 = nullptr, *d_fc2 = nullptr, 
    *d_b1 = nullptr, *d_b2 = nullptr;
    void *d_workspace = nullptr;

    CUDA_CHECK(cudaMalloc(&d_x  , total_bytes));
    CUDA_CHECK(cudaMalloc(&d_h  , total_hidden_bytes));
    CUDA_CHECK(cudaMalloc(&d_y  , total_bytes));
    CUDA_CHECK(cudaMalloc(&d_fc1, fc_matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_fc2, fc_matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_b1 , hidden_bytes));
    CUDA_CHECK(cudaMalloc(&d_b2 , embeddings_bytes));
    CUDA_CHECK(cudaMalloc(&d_workspace, WORKSPACE_SIZE));

    CUDA_CHECK(cudaMemcpy(d_x, gpu_input.data(), total_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1 , gpu_b1.data(), hidden_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2 , gpu_b2.data(), embeddings_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fc1, gpu_fc1.data(), fc_matrix_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fc2, gpu_fc2.data(), fc_matrix_bytes, cudaMemcpyHostToDevice));

    cudaStream_t stream; CUDA_CHECK(cudaStreamCreate(&stream));
    cublasLtHandle_t handle; CUBLAS_CHECK(cublasLtCreate(&handle));

    // - Reference creation
    RowVector cpu_b1(h_b1.data(), hidden_channels);
    RowVector cpu_b2(h_b2.data(), embeddings);
    Matrix fc1(h_fc1.data(), fc_matrix_n, hidden_channels, embeddings);
    Matrix fc2(h_fc2.data(), fc_matrix_n, embeddings, hidden_channels);
    Linear cpu_fc1(embeddings,hidden_channels);
    Linear cpu_fc2(hidden_channels, embeddings);
    cpu_fc1.move_A(fc1); cpu_fc1.move_b(cpu_b1);
    cpu_fc2.move_A(fc2); cpu_fc2.move_b(cpu_b2);

    Mlp cpu_mlp(embeddings,hidden_channels,embeddings,GELU,true);
    cpu_mlp.move_fc1(cpu_fc1);
    cpu_mlp.move_fc2(cpu_fc2);

    Tensor cpu_x(h_input.data(), elements_n, batch, tokens, embeddings);
    Tensor cpu_y(batch, tokens, embeddings);
    cpu_mlp.forward(cpu_x, cpu_y);

    if (kernel_id == 0 || kernel_id == 1){
        cout << "|| Unfused kernel ||" << endl;
        mlp_time res_time = unfused_mlp(
            handle,
            stream,
            batch, tokens, embeddings, hidden_channels,
            d_workspace,
            d_x, d_fc1, d_h, d_b1, d_fc2, d_b2, d_y,
            stride_val
        );
        CUDA_CHECK(cudaMemcpy(gpu_output.data(), d_y, total_bytes, cudaMemcpyDeviceToHost));
        cout << "Last iteration comparison with CPU: " << compare_results(cpu_y, gpu_output.data()) * 100.0f<< "%" <<endl;
        res_time.print();
        res_time.to_JSON(batch, new int[2]{0, stride_val});
    }
    if (kernel_id == 0 || kernel_id == 2){
        cout << "|| Fused kernel ||" << endl;
        mlp_time res_time = fused_mlp(
            handle,
            stream,
            batch, tokens, embeddings, hidden_channels,
            d_workspace,
            d_x, d_fc1, d_h, gpu_b1.data(), d_fc2, gpu_b2.data(), d_y
        );
        CUDA_CHECK(cudaMemcpy(gpu_output.data(), d_y, total_bytes, cudaMemcpyDeviceToHost));
        cout << "Last iteration comparison with CPU: " << compare_results(cpu_y, gpu_output.data()) * 100.0f<< "%" <<endl;
        res_time.print();
        res_time.to_JSON(batch, new int[2]{1, stride_val});
    }
    if(kernel_id == 3){
        cout << " || Single Run unfused ||" << endl;
        single_run(
            handle,
            stream,
            batch, tokens, embeddings, hidden_channels,
            d_workspace,
            d_x, d_fc1, d_h, d_b1, d_fc2, d_b2,
            gpu_b1.data(), gpu_b2.data(),
            d_y,
            stride_val, false
        );
        CUDA_CHECK(cudaMemcpy(gpu_output.data(), d_y, total_bytes, cudaMemcpyDeviceToHost));
        cout << "Single run comparison with CPU: " << compare_results(cpu_y, gpu_output.data()) * 100.0f<< "%" <<endl;
        
    }
    if(kernel_id == 4){
        cout << " || Single Run fused ||" << endl;
        single_run(
            handle,
            stream,
            batch, tokens, embeddings, hidden_channels,
            d_workspace,
            d_x, d_fc1, d_h, d_b1, d_fc2, d_b2,
            gpu_b1.data(), gpu_b2.data(),
            d_y,
            stride_val, true
        );
        CUDA_CHECK(cudaMemcpy(gpu_output.data(), d_y, total_bytes, cudaMemcpyDeviceToHost));
        cout << "Single run comparison with CPU: " << compare_results(cpu_y, gpu_output.data()) * 100.0f<< "%" <<endl;
    }

    // - Cleanup
    
    CUDA_CHECK(cudaFree(d_x  ));
    CUDA_CHECK(cudaFree(d_h  ));
    CUDA_CHECK(cudaFree(d_y  ));
    CUDA_CHECK(cudaFree(d_fc1));
    CUDA_CHECK(cudaFree(d_fc2));
    CUDA_CHECK(cudaFree(d_b1 ));
    CUDA_CHECK(cudaFree(d_b2 ));
    CUDA_CHECK(cudaFree(d_workspace));

    return 0;
}
