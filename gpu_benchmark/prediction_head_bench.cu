#include "../gpu_include/gpu_vit.h"
#include "../include/vision_transformer.h"
#include "../gpu_include/bench_utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>
#include <cstring>
#include <cstdlib>

#define EPS 1e-4
#define CLASS_N 100

struct ph_time{
    float kernel_time;
    float linear_time;
    float softmax_time;
    float prediction_time; //CPU argmax

    ph_time(float total, float linear = 0.0f, float softmax = 0.0f, float prediction_time_ = 0.0f)
        : kernel_time(total), linear_time(linear), softmax_time(softmax), prediction_time(prediction_time_) {}

    void print(){
        cout << "   Total time (ms)     : " << kernel_time << endl;
        cout << "   Linear time (ms)    : " << linear_time << endl;
        cout << "   Softmax time (ms)   : " << softmax_time << endl;
        cout << "   Prediction time (ms): " << prediction_time << endl;
    }

    void to_JSON(int batch, int params[]){
        int tokens_per_block = params[0];
        int stride_val       = params[1];

        cout << "{\n"
            << "\"batch\":" << batch << ",\n"
            << "\"params\": {\n" 
                << "\"tokens_per_block\":" << tokens_per_block << ",\n"
                << "\"stride_val\":" << stride_val << "\n"
            << "},\n"
            << "\"time\": {\n" 
                << "\"kernel_time\":" << kernel_time << ",\n"
                << "\"linear_time\":" << linear_time << ",\n"
                << "\"softmax_time\":" << softmax_time << ",\n"
                << "\"prediction_time\":" << prediction_time << "\n"
            << "}\n"
            << "}\n";
    }
};

ph_time full_prediction_head(
    cudaStream_t &stream, cudnnHandle_t &cudnn_handle, cublasLtHandle_t &cublas_handle,
    int batch, int tokens, int embeddings, int class_num,
    void *d_x, void *d_t, void *d_y, void *d_pred, void *d_workspace,
    void *d_ln_scale, void *d_ln_bias, void *d_lin_w, void *d_lin_bias,    
    half *gpu_output, int * predictions, //For CPU comparison
    int tokens_per_block, int stride_val
){
    size_t pred_n = batch * class_num;

    GpuPredictionHead gpu_ph(
        batch, tokens, embeddings, class_num,
        cudnn_handle, cublas_handle,stream
    );

    gpu_ph.init_descriptors();
    gpu_ph.set_shared_buffers(d_x, d_t, d_y, d_pred, d_workspace);
    gpu_ph.set_shared_weights(d_ln_scale, d_ln_bias, d_lin_w, d_lin_bias);
    
    gpu_ph.tokens_per_block = tokens_per_block;
    gpu_ph.stride_val = stride_val;

    gpu_ph.forward(false);
    gpu_ph.compute_predictions();
    CUDA_CHECK(cudaMemcpy(gpu_output, gpu_ph.d_pred, sizeof(half) * pred_n, cudaMemcpyDeviceToHost));
    for(int i = 0; i < batch; i++){
        predictions[i] = gpu_ph.class_prediction[i];
    }

    float avg_ms = time_kernel(WARM_UP, N, stream,[&]() {
        gpu_ph.forward(false);
    });


    gpu_ph.destroy_descriptors();

    return ph_time(avg_ms);
}

void single_run(cudaStream_t &stream, cudnnHandle_t &cudnn_handle, cublasLtHandle_t &cublas_handle,
    int batch, int tokens, int embeddings, int class_num,
    void *d_x, void *d_t, void *d_y, void *d_pred, void *d_workspace,
    void *d_ln_scale, void *d_ln_bias, void *d_lin_w, void *d_lin_bias,    
    half *gpu_output, int * predictions, //For CPU comparison
    int tokens_per_block, int stride_val
){
    size_t pred_n = batch * class_num;

    GpuPredictionHead gpu_ph(
        batch, tokens, embeddings, class_num,
        cudnn_handle, cublas_handle,stream
    );

    gpu_ph.init_descriptors();
    gpu_ph.set_shared_buffers(d_x, d_t, d_y, d_pred, d_workspace);
    gpu_ph.set_shared_weights(d_ln_scale, d_ln_bias, d_lin_w, d_lin_bias);
    
    gpu_ph.tokens_per_block = tokens_per_block;
    gpu_ph.stride_val = stride_val;

    gpu_ph.forward(false);
    gpu_ph.compute_predictions();
    CUDA_CHECK(cudaMemcpy(gpu_output, gpu_ph.d_pred, sizeof(half) * pred_n, cudaMemcpyDeviceToHost));
    for(int i = 0; i < batch; i++){
        predictions[i] = gpu_ph.class_prediction[i];
    }

    gpu_ph.destroy_descriptors();

}
ph_time all_times(
    cudaStream_t &stream, cudnnHandle_t &cudnn_handle, cublasLtHandle_t &cublas_handle,
    int batch, int tokens, int embeddings, int class_num,
    void *d_x, void *d_t, void *d_y, void *d_pred, void *d_workspace,
    void *d_ln_scale, void *d_ln_bias, void *d_lin_w, void *d_lin_bias,    
    half *gpu_output, int * predictions, //For CPU comparison
    int tokens_per_block, int stride_val
){
    GpuPredictionHead gpu_ph(
        batch, tokens, embeddings, class_num,
        cudnn_handle, cublas_handle,stream
    );
    gpu_ph.init_descriptors();
    softmax_desc softmax = gpu_ph.softmax; //TO CHECK maybe copy not working
    float alpha = 1.0f, beta = 0.0f;

    float avg_ms = full_prediction_head(
        stream, cudnn_handle, cublas_handle,
        batch, tokens, embeddings, class_num,
        d_x, d_t, d_y, d_pred, d_workspace,
        d_ln_scale, d_ln_bias, d_lin_w, d_lin_bias,
        gpu_output, predictions,
        tokens_per_block, stride_val
    ).kernel_time;

    /*pool*/
    half * pool_iterator = (half *)d_t;
    half * tokens_iterator = (half *)d_x;
    for(int i = 0; i < batch; i++){
        cudaMemcpyAsync(pool_iterator, tokens_iterator, sizeof(half) * embeddings, cudaMemcpyDeviceToDevice); //Using default stream
        pool_iterator += embeddings;
        tokens_iterator += embeddings * tokens;
    }

    // /*matmul*/ CHECK THE WEIGHTS LOADING
    float avg_linear_time = time_kernel(WARM_UP, N, stream,[&]() {
        strided_linear_layer(
            cublas_handle,stream,
            batch,  1, class_num, stride_val,
            gpu_ph.matmul, gpu_ph.algo, d_workspace,
            d_t, d_lin_w, d_lin_bias, d_y,
            false
        );
    });


    /*cuDNN softmax*/
    float avg_softmax_time = time_kernel(WARM_UP, N, stream,[&]() {
        CUDNN_CHECK(
            cudnnSoftmaxForward(
                cudnn_handle,
                softmax.algo,
                softmax.mode,
                &alpha, softmax.x_desc, d_y, 
                &beta, softmax.x_desc, d_pred
            )
        )
    });

    float avg_prediction_time = time_cpu(WARM_UP, N, [&]() {
        gpu_ph.compute_predictions();
    });

    
    gpu_ph.destroy_descriptors();

    return ph_time(avg_ms, avg_linear_time, avg_softmax_time, avg_prediction_time);
}

int main(int argc, char** argv)
{
    int kernel_id           = get_arg(argc, argv, "--kernel", 1);
    int batch               = get_arg(argc, argv, "--batch", 32);
    int tokens_per_block    = get_arg(argc, argv, "--tokens_per_block", 32);
    int stride_val          = get_arg(argc, argv, "--stride", 2);
    int tokens              = TOKENS_NUM_VIT;
    int embeddings          = EMBEDDINGS_SIZE;
    int num_classes         = CLASS_N;

    cout << "Block Benchmark\n"
              << " batch_size:          " << batch << "\n"
              << " tokens:              " << tokens          << "\n"
              << " embeddings:          " << embeddings      << "\n"
              << " num_classes:         " << num_classes << "   \n"
              << " tokens_per_block:    " << tokens_per_block << "\n"
              << " residual stride:     " << stride_val << "\n"
              << " warmup_iters:        " << WARM_UP << "\n"
              << " timed_iters:         " << N << "\n";

    
    // -  Memory allocation
    size_t elements_n = batch * tokens * embeddings;
    size_t pred_elem_n = batch * num_classes;

    vector<float> h_input   (elements_n);
    vector<float> h_ln_scale(embeddings);     //Layer norm
    vector<float> h_ln_bias (embeddings);
    vector<float> h_lin_w   (embeddings * num_classes);
    vector<float> h_lin_bias(num_classes);
    
    vector<half> gpu_input   (elements_n);
    vector<half> gpu_output  (pred_elem_n);
    vector<half> gpu_ln_scale(embeddings);     //Layer norm
    vector<half> gpu_ln_bias (embeddings);
    vector<half> gpu_lin_w   (embeddings * num_classes);
    vector<half> gpu_lin_bias(num_classes);

    random_device rd;          
    mt19937 gen(rd());         
    uniform_real_distribution<float> dist(-0.1f, 0.1f);

    size_t loop_range = elements_n;
    for(size_t i = 0; i < loop_range; i++){
        if(i < num_classes){
            h_lin_bias[i] = dist(gen);
        }
        if(i < embeddings){
            h_ln_scale[i] = dist(gen);
            h_ln_bias [i] = dist(gen);
        }
        if(i < embeddings * num_classes){
            h_lin_w[i] = dist(gen);
        }
        if(i < elements_n){
            h_input[i] = dist(gen);
        }
        
    }

    f32_to_f16(h_input.data(), gpu_input.data(), elements_n);

    f32_to_f16(h_ln_scale.data(), gpu_ln_scale.data(), embeddings);           
    f32_to_f16(h_ln_bias.data(), gpu_ln_bias.data(), embeddings);           
    f32_to_f16(h_lin_w.data(), gpu_lin_w.data(), embeddings * num_classes);           
    f32_to_f16(h_lin_bias.data(), gpu_lin_bias.data(), num_classes);           


    void * d_x, * d_t, * d_y, * d_pred,* d_ln_scale, * d_ln_bias, * d_lin_w, * d_lin_bias;
    void *d_workspace = nullptr;

    CUDA_CHECK(cudaMalloc(&d_x       , sizeof(half) * elements_n));
    CUDA_CHECK(cudaMalloc(&d_t       , sizeof(half) * batch * embeddings));
    CUDA_CHECK(cudaMalloc(&d_y       , sizeof(half) * batch * num_classes));
    CUDA_CHECK(cudaMalloc(&d_pred    , sizeof(half) * batch * num_classes));
    CUDA_CHECK(cudaMalloc(&d_ln_scale, sizeof(half) * embeddings ));
    CUDA_CHECK(cudaMalloc(&d_ln_bias , sizeof(half) * embeddings ));
    CUDA_CHECK(cudaMalloc(&d_lin_w   , sizeof(half) * embeddings * num_classes));
    CUDA_CHECK(cudaMalloc(&d_lin_bias, sizeof(half) * num_classes ));
    CUDA_CHECK(cudaMemcpy(d_x        , gpu_input.data(), sizeof(half) * elements_n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ln_scale , gpu_ln_scale.data(), sizeof(half) * embeddings, cudaMemcpyHostToDevice ));
    CUDA_CHECK(cudaMemcpy(d_ln_bias  , gpu_ln_bias.data(), sizeof(half) * embeddings, cudaMemcpyHostToDevice ));
    CUDA_CHECK(cudaMemcpy(d_lin_w    , gpu_lin_w.data(), sizeof(half) * embeddings * num_classes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lin_bias , gpu_lin_bias.data(), sizeof(half) * num_classes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_workspace, WORKSPACE_SIZE));

    cudaStream_t stream; CUDA_CHECK(cudaStreamCreate(&stream));
    cudnnHandle_t cudnn_handle; CUDNN_CHECK(cudnnCreate(&cudnn_handle));
    CUDNN_CHECK(cudnnSetStream(cudnn_handle, stream));    
    cublasLtHandle_t cublas_handle; CUBLAS_CHECK(cublasLtCreate(&cublas_handle));


    // - Reference creation
    
    Tensor cpu_x(h_input.data(), elements_n, batch, tokens, embeddings);
    Tensor head_in(batch, 1 , embeddings);
    Tensor head_out(batch,1, num_classes);
    Matrix head_w(h_lin_w.data(), embeddings * num_classes, num_classes, embeddings);
    RowVector ln_g(h_ln_scale.data(), embeddings);
    RowVector ln_bias(h_ln_bias.data(), embeddings);
    RowVector head_bias(h_lin_bias.data(), num_classes);

    LayerNorm ln(embeddings, EPS, true);
    ln.move_g(ln_g); ln.move_b(ln_bias);
    Linear head(embeddings, num_classes, true);
    head.move_A(head_w); head.move_b(head_bias);
    
    // - CPU forward (equivalent to my GPU class forward)
    ln(cpu_x);
    global_pool_nlc(cpu_x, head_in, pool_token, 1, true); //num_prefix_tokens = 1 (cls token)
    head(head_in, head_out);
    PredictionBatch pb(head_out);
    
    Tensor cpu_y(batch, 1, num_classes);
    vector<int> gpu_predictions(batch);

    pb.get_prediction_probability_tensor(cpu_y);

    if (kernel_id == 0 || kernel_id == 1){
        cout << "|| GPU Prediction Head ||" << endl;
        ph_time res_time = full_prediction_head(
            stream, cudnn_handle, cublas_handle,
            batch, tokens, embeddings, num_classes,
            d_x, d_t, d_y, d_pred, d_workspace,
            d_ln_scale, d_ln_bias, d_lin_w, d_lin_bias,
            gpu_output.data(), gpu_predictions.data(),
            tokens_per_block, stride_val
        );
        cout << "First iteration comparison with CPU: " << compare_results(cpu_y, gpu_output.data()) * 100.0f<< "%" <<endl;
        res_time.print();
        cout << "Prediction comparison with CPU: " << compare_predictions(pb, gpu_predictions.data()) * 100.0f<< "%" <<endl;        
        res_time.to_JSON(batch, new int[2]{tokens_per_block, stride_val});
    }
    if (kernel_id == 0 || kernel_id == 2){
        cout << "|| Single Run ||" << endl;
        if(kernel_id == 0){
            CUDA_CHECK(cudaMemcpy(d_x, gpu_input.data(), sizeof(half) * elements_n, cudaMemcpyHostToDevice));
        }
        single_run(
            stream, cudnn_handle, cublas_handle,
            batch, tokens, embeddings, num_classes,
            d_x, d_t, d_y, d_pred, d_workspace,
            d_ln_scale, d_ln_bias, d_lin_w, d_lin_bias,
            gpu_output.data(), gpu_predictions.data(),
            tokens_per_block, stride_val
        );
        cout << "Single run comparison with CPU: " << compare_results(cpu_y, gpu_output.data()) * 100.0f<< "%" <<endl;
        cout << "Prediction comparison with CPU: " << compare_predictions(pb, gpu_predictions.data()) * 100.0f<< "%" <<endl;
    }
    if (kernel_id == 0 || kernel_id == 3){
        cout << "|| All times ||" << endl;
        ph_time res_time = all_times(
            stream, cudnn_handle, cublas_handle,
            batch, tokens, embeddings, num_classes,
            d_x, d_t, d_y, d_pred, d_workspace,
            d_ln_scale, d_ln_bias, d_lin_w, d_lin_bias,
            gpu_output.data(), gpu_predictions.data(),
            tokens_per_block, stride_val
        );
        res_time.print();
        res_time.to_JSON(batch, new int[2]{tokens_per_block, stride_val});
    }

    // - Cleanup
    
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_t));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_pred));
    CUDA_CHECK(cudaFree(d_workspace));
    CUDNN_CHECK(cudnnDestroy(cudnn_handle));
    CUBLAS_CHECK(cublasLtDestroy(cublas_handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}