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
#define MLP_HIDDEN 3072
#define SCALE 1.0f
struct block_time{
    float total_time;
    float attn_time;

    block_time(float total_time_, float attn_time_ = 0.0f): 
        total_time(total_time_),
        attn_time(attn_time_)
    {}

    void print(){
        cout << "   Total time (ms): " << total_time << endl;
        cout << "   Attention time(ms): " << attn_time << endl;
    }

    void to_JSON(int batch, int params[]){
        int tokens_per_block = params[0];
        int stride_val      = params[1];
        bool mlp_type       = params[2];  

        cout << "{\n"
            << "\"batch\":" << batch << ",\n"
            << "\"params\": {\n" 
                << "\"tokens_per_block\":" << tokens_per_block << ",\n"
                << "\"stride_val\":" << stride_val << ",\n"
                << "\"mlp_type\":" << mlp_type << "\n"
            << "},\n"
            << "\"time\": {\n" 
                << "\"total_time\":" << total_time << ",\n"
                << "\"attn_time\":" << attn_time << ",\n"
            << "}\n"
            << "}\n";
    }
};

block_time full_block(
    cudaStream_t &stream, cudnnHandle_t &cudnn_handle, cublasLtHandle_t &cublas_handle,
    int batch, int tokens, int embeddings, int hidden_channels,
    void *d_x, void *d_t, void *d_h, void *d_y, void *d_workspace,
    half * gpu_nb1, half *gpu_ns1, half *gpu_nb2, half *gpu_ns2,
    half * gpu_fc1, half *gpu_b1, half *gpu_fc2, half *gpu_b2,
    attn_data_gpu<half> &gpu_attn_weights,
    half *gpu_output, //For CPU comparison
    int tokens_per_block, int stride_val,
    int scale, int num_heads,
    bool mlp_type = false
){
    size_t total_elem_n = batch * tokens * embeddings;

    GpuBlock block(
        stream, cudnn_handle, cublas_handle,
        batch, tokens, embeddings, hidden_channels,
        mlp_type,
        EPS, scale, num_heads,
        false, false
    );

    block.set_buffers(d_x, d_t, d_y, d_h, d_workspace);
    block.init_descriptors();
    block.allocate_weights();
    block.load_weights(
        gpu_nb1, gpu_ns1,
        gpu_nb2, gpu_ns2,
        gpu_fc1, gpu_b1,
        gpu_fc2, gpu_b2,
        gpu_attn_weights
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    block.forward(false, tokens_per_block);
    CUDA_CHECK(cudaMemcpy(gpu_output, d_x, sizeof(half) * total_elem_n, cudaMemcpyDeviceToHost));
    float avg_ms = time_kernel(WARM_UP, N, stream,[&]() {
        block.forward(false, tokens_per_block);
    });

    block.destroy_descriptors();
    return block_time(avg_ms);
}

void single_run(
    cudaStream_t &stream, cudnnHandle_t &cudnn_handle, cublasLtHandle_t &cublas_handle,
    int batch, int tokens, int embeddings, int hidden_channels,
    void *d_x, void *d_t, void *d_h, void *d_y, void *d_workspace,
    half * gpu_nb1, half *gpu_ns1, half *gpu_nb2, half *gpu_ns2,
    half * gpu_fc1, half *gpu_b1, half *gpu_fc2, half *gpu_b2,
    attn_data_gpu<half> &gpu_attn_weights,
    half *gpu_output, //For CPU comparison
    int tokens_per_block, int stride_val,
    int scale, int num_heads,
    bool mlp_type = false
){
    size_t total_elem_n = batch * tokens * embeddings;

    GpuBlock block(
        stream, cudnn_handle, cublas_handle,
        batch, tokens, embeddings, hidden_channels,
        mlp_type,
        EPS, scale, num_heads,
        false, false
    );

    block.set_buffers(d_x, d_t, d_y, d_h, d_workspace);
    block.init_descriptors();
    block.allocate_weights();
    block.load_weights(
        gpu_nb1, gpu_ns1,
        gpu_nb2, gpu_ns2,
        gpu_fc1, gpu_b1,
        gpu_fc2, gpu_b2,
        gpu_attn_weights
    );
    // CUDA_CHECK(cudaDeviceSynchronize());
    block.forward(false, tokens_per_block);
    CUDA_CHECK(cudaMemcpy(gpu_output, d_x, sizeof(half) * total_elem_n, cudaMemcpyDeviceToHost));
}

block_time all_times(
    cudaStream_t &stream, cudnnHandle_t &cudnn_handle, cublasLtHandle_t &cublas_handle,
    int batch, int tokens, int embeddings, int hidden_channels,
    void *d_x, void *d_t, void *d_h, void *d_y, void *d_workspace,
    half * gpu_nb1, half *gpu_ns1, half *gpu_nb2, half *gpu_ns2,
    half * gpu_fc1, half *gpu_b1, half *gpu_fc2, half *gpu_b2,
    attn_data_gpu<half> &gpu_attn_weights,
    half *gpu_output, //For CPU comparison
    int tokens_per_block, int stride_val,
    int scale, int num_heads,
    bool mlp_type = false
){
    
    float avg_ms = full_block(
        stream, cudnn_handle, cublas_handle,
        batch, tokens, embeddings, hidden_channels,
        d_x, d_t, d_h, d_y, d_workspace,
        gpu_nb1, gpu_ns1, gpu_nb2, gpu_ns2,
        gpu_fc1, gpu_b1, gpu_fc2, gpu_b2,
        gpu_attn_weights,
        gpu_output,
        tokens_per_block, stride_val,
        scale, num_heads,
        mlp_type
    ).total_time;


    GpuBlock block(
        stream, cudnn_handle, cublas_handle,
        batch, tokens, embeddings, hidden_channels,
        mlp_type,
        EPS, scale, num_heads,
        false, false
    );
    
    block.set_buffers(d_x, d_t, d_y, d_h, d_workspace);
    block.init_descriptors();
    block.allocate_weights();
    block.load_weights(
        gpu_nb1, gpu_ns1,
        gpu_nb2, gpu_ns2,
        gpu_fc1, gpu_b1,
        gpu_fc2, gpu_b2,
        gpu_attn_weights
    );

    
    float avg_attn_time = time_kernel(WARM_UP, N, stream,[&]() {
        attention_device(
            cudnn_handle,
            d_x, d_t,
            block.fused_desc
        );
    });

    block.destroy_descriptors();
    return block_time(avg_ms, avg_attn_time);
}

int main(int argc, char** argv)
{
    int kernel_id           = get_arg(argc, argv, "--kernel", 0);
    int batch               = get_arg(argc, argv, "--batch", 32);
    int tokens_per_block    = get_arg(argc, argv, "--tokens_per_block", 32);
    int stride_val          = get_arg(argc, argv, "--stride", 2);
    bool mlp_type           = get_arg(argc, argv, "--mlp_type", 0) == 0 ? false : true;
    int tokens              = TOKENS_NUM_VIT;
    int embeddings          = EMBEDDINGS_SIZE;
    int hidden_channels     = MLP_HIDDEN;
    int num_heads           = NUM_HEADS;
    int scale               = SCALE;

    cout << "Block Benchmark\n"
              << " batch_size:          " << batch << "\n"
              << " tokens:              " << tokens          << "\n"
              << " embeddings:          " << embeddings      << "\n"
              << " hidden_channels:     " << hidden_channels << "\n"
              << " num_heads:           " << num_heads << "\n"
              << " scale:               " << scale << "\n"
              << " mlp_type:            " << yesno(mlp_type) << "\n"
              << " tokens_per_block:    " << tokens_per_block << "\n"
              << " residual stride:     " << stride_val << "\n"
              << " warmup_iters:        " << WARM_UP << "\n"
              << " timed_iters:         " << N << "\n";

    
    // -  Memory allocation
    size_t elements_n = batch * tokens * embeddings;
    size_t hidden_elements_n = batch * tokens * hidden_channels;
    size_t fc_matrix_n = embeddings * hidden_channels;

    size_t total_bytes = elements_n * sizeof(half);

    vector<float> h_input(elements_n);

    vector<float> h_nb1(embeddings);     //Layer norm
    vector<float> h_ns1(embeddings);
    vector<float> h_nb2(embeddings);
    vector<float> h_ns2(embeddings);
    vector<float> h_fc1(fc_matrix_n);          //Mlp
    vector<float> h_fc2(fc_matrix_n);    
    vector<float> h_b1(hidden_channels);
    vector<float> h_b2(embeddings);
    vector<float> h_q(embeddings * embeddings);//Attention
    vector<float> h_k(embeddings * embeddings);        
    vector<float> h_v(embeddings * embeddings);        
    vector<float> h_p(embeddings * embeddings);        
    vector<float> h_qb(embeddings);        
    vector<float> h_kb(embeddings);        
    vector<float> h_vb(embeddings);        
    vector<float> h_pb(embeddings);        

    vector<half> gpu_input(elements_n);
    vector<half> gpu_output(elements_n);

    vector<half> gpu_nb1(embeddings);           
    vector<half> gpu_ns1(embeddings);           
    vector<half> gpu_nb2(embeddings);           
    vector<half> gpu_ns2(embeddings);           
    vector<half> gpu_fc1(fc_matrix_n);          
    vector<half> gpu_fc2(fc_matrix_n);          
    vector<half> gpu_b1(hidden_channels);       
    vector<half> gpu_b2(embeddings);            
    vector<half> gpu_q(embeddings * embeddings);
    vector<half> gpu_k(embeddings * embeddings);        
    vector<half> gpu_v(embeddings * embeddings);        
    vector<half> gpu_p(embeddings * embeddings);        
    vector<half> gpu_qb(embeddings);            
    vector<half> gpu_kb(embeddings);            
    vector<half> gpu_vb(embeddings);            
    vector<half> gpu_pb(embeddings);            


    random_device rd;          
    mt19937 gen(rd());         
    uniform_real_distribution<float> dist(-0.1f, 0.1f);

    size_t loop_range = max(fc_matrix_n, elements_n);
    for(size_t i = 0; i < loop_range; i++){
        if(i < embeddings){
            h_nb1[i] = dist(gen);
            h_ns1[i] = dist(gen);
            h_nb2[i] = dist(gen);
            h_ns2[i] = dist(gen);
            h_qb[i] = dist(gen);
            h_kb[i] = dist(gen);
            h_vb[i] = dist(gen);
            h_pb[i] = dist(gen);
            h_b2[i] = dist(gen);
        }
        if(i < hidden_channels){
            h_b1[i] = dist(gen);
        }
        if(i < embeddings * embeddings){
            h_q[i] = dist(gen);
            h_k[i] = dist(gen);
            h_v[i] = dist(gen);
            h_p[i] = dist(gen);
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

    f32_to_f16(h_nb1.data() ,gpu_nb1.data(), embeddings);           
    f32_to_f16(h_ns1.data() ,gpu_ns1.data(), embeddings);           
    f32_to_f16(h_nb2.data() ,gpu_nb2.data(), embeddings);           
    f32_to_f16(h_ns2.data() ,gpu_ns2.data(), embeddings);           
    f32_to_f16(h_fc1.data() ,gpu_fc1.data(), fc_matrix_n);          
    f32_to_f16(h_fc2.data() ,gpu_fc2.data(), fc_matrix_n);          
    f32_to_f16(h_b1.data() , gpu_b1.data(), hidden_channels);       
    f32_to_f16(h_b2.data() , gpu_b2.data(), embeddings);            
    f32_to_f16(h_qb.data() , gpu_qb.data(), embeddings);            
    f32_to_f16(h_kb.data() , gpu_kb.data(), embeddings);            
    f32_to_f16(h_vb.data() , gpu_vb.data(), embeddings);            
    f32_to_f16(h_pb.data() , gpu_pb.data(), embeddings);            
    
    transpose_out_of_place(h_q.data(), gpu_q.data(), embeddings, embeddings);
    transpose_out_of_place(h_k.data(), gpu_k.data(), embeddings, embeddings);
    transpose_out_of_place(h_v.data(), gpu_v.data(), embeddings, embeddings);
    transpose_out_of_place(h_p.data(), gpu_p.data(), embeddings, embeddings);

    attn_data_gpu gpu_attn_weights(
        gpu_q.data(), gpu_k.data(), gpu_v.data(), gpu_p.data(),
        gpu_qb.data(), gpu_kb.data(), gpu_vb.data(), gpu_pb.data()
    );

    void *d_x = nullptr, *d_t = nullptr,*d_h = nullptr, *d_y = nullptr;
    void *d_workspace = nullptr;

    CUDA_CHECK(cudaMalloc(&d_x      ,sizeof(half) * elements_n));
    CUDA_CHECK(cudaMalloc(&d_t      ,sizeof(half) * elements_n));
    CUDA_CHECK(cudaMalloc(&d_y      ,sizeof(half) * elements_n));
    CUDA_CHECK(cudaMalloc(&d_h      ,sizeof(half) * hidden_elements_n));
    CUDA_CHECK(cudaMalloc(&d_workspace, WORKSPACE_SIZE));
  

    CUDA_CHECK(cudaMemcpy(d_x, gpu_input.data(), total_bytes, cudaMemcpyHostToDevice));
   
    cudaStream_t stream; CUDA_CHECK(cudaStreamCreate(&stream));
    cudnnHandle_t cudnn_handle; CUDNN_CHECK(cudnnCreate(&cudnn_handle));
    CUDNN_CHECK(cudnnSetStream(cudnn_handle, stream));    
    cublasLtHandle_t cublas_handle; CUBLAS_CHECK(cublasLtCreate(&cublas_handle));


    // - Reference creation
    
    Tensor cpu_x(h_input.data(), elements_n, batch, tokens, embeddings);
    Block cpu_block(embeddings, num_heads, hidden_channels / embeddings, true, false, scale, GELU);
    Tensor cpu_y(batch, tokens, embeddings);
    
    //-Attention
    Matrix q(h_q.data(), embeddings*embeddings, embeddings, embeddings);
    Matrix k(h_k.data(), embeddings*embeddings, embeddings, embeddings);
    Matrix v(h_v.data(), embeddings*embeddings, embeddings, embeddings);
    Matrix p(h_p.data(), embeddings*embeddings, embeddings, embeddings);
    RowVector qb(h_qb.data(), embeddings);
    RowVector kb(h_kb.data(), embeddings);
    RowVector vb(h_vb.data(), embeddings);
    RowVector pb(h_pb.data(), embeddings);
    Attention attn(embeddings, num_heads, true, false);
    Linear q_gen(embeddings, embeddings, true);
    q_gen.move_A(q);
    q_gen.move_b(qb);
    Linear k_gen(embeddings, embeddings, true);
    k_gen.move_A(k);
    k_gen.move_b(kb);
    Linear v_gen(embeddings, embeddings, true);
    v_gen.move_A(v);
    v_gen.move_b(vb);
    Linear proj(embeddings, embeddings, true);
    proj.move_A(p);
    proj.move_b(pb);

    attn.move_qkv_gen(q_gen, k_gen, v_gen);
    attn.move_proj(proj);

    // Mlp Initialization
    Matrix A1(h_fc1.data(), hidden_channels * embeddings, hidden_channels, embeddings);
    RowVector b1(h_b1.data(), hidden_channels);
    Matrix A2(h_fc2.data(), embeddings * hidden_channels, embeddings, hidden_channels);
    RowVector b2(h_b2.data(), embeddings);
    
    Linear fc1(embeddings, hidden_channels, true);
    fc1.move_A(A1);
    fc1.move_b(b1);
    Linear fc2(hidden_channels, embeddings, true);
    fc2.move_A(A2);
    fc2.move_b(b2);

    Mlp mlp(embeddings, hidden_channels, embeddings, GELU, true, false);

    mlp.move_fc1(fc1);
    mlp.move_fc2(fc2);

    RowVector n1g(h_ns1.data(), embeddings);
    RowVector n1b(h_nb1.data(), embeddings);
    RowVector n2g(h_ns2.data(), embeddings);
    RowVector n2b(h_nb2.data(), embeddings);
    
    LayerNorm block_n1(embeddings, EPS, true);
    block_n1.move_g(n1g);
    block_n1.move_b(n1b);
    LayerNorm block_n2(embeddings, EPS, true);
    block_n2.move_g(n2g);
    block_n2.move_b(n2b);

    // Block Initialization
    cpu_block.move_attn(attn);
    cpu_block.move_mlp(mlp);
    cpu_block.move_norm1(block_n1);
    cpu_block.move_norm2(block_n2);

    cpu_block.forward(cpu_x, cpu_y);

    if (kernel_id == 0 || kernel_id == 1){
        cout << "|| GPU Block ||" << endl;
        block_time res_time = full_block(
            stream, cudnn_handle, cublas_handle,
            batch, tokens, embeddings, hidden_channels,
            d_x, d_t, d_h, d_y, d_workspace,
            gpu_nb1.data(), gpu_ns1.data(),
            gpu_nb2.data(), gpu_ns2.data(),
            gpu_fc1.data(), gpu_b1.data(),
            gpu_fc2.data(), gpu_b2.data(),
            gpu_attn_weights, gpu_output.data(),
            stride_val, tokens_per_block,
            scale, num_heads,
            mlp_type
        );
        cout << "Last iteration comparison with CPU: " << compare_results(cpu_y, gpu_output.data()) * 100.0f<< "%" <<endl;
        res_time.print();
        res_time.to_JSON(batch, new int[3]{tokens_per_block, stride_val, mlp_type});
    }
    if (kernel_id == 0 || kernel_id == 2){
        cout << "|| Single Run ||" << endl;
        if(kernel_id == 0){
            CUDA_CHECK(cudaMemcpy(d_x, gpu_input.data(), sizeof(half) * elements_n, cudaMemcpyHostToDevice));
        }
        single_run(
            stream, cudnn_handle, cublas_handle,
            batch, tokens, embeddings, hidden_channels,
            d_x, d_t, d_h, d_y, d_workspace,
            gpu_nb1.data(), gpu_ns1.data(),
            gpu_nb2.data(), gpu_ns2.data(),
            gpu_fc1.data(), gpu_b1.data(),
            gpu_fc2.data(), gpu_b2.data(),
            gpu_attn_weights, gpu_output.data(),
            stride_val, tokens_per_block,
            scale, num_heads,
            mlp_type
        );

        cout << "Comparison with CPU: " << compare_results(cpu_y, gpu_output.data()) * 100.0f<< "%" <<endl;
    }

    if (kernel_id == 0 || kernel_id == 3){
        cout << "|| Total + Attention ||" << endl;
        if(kernel_id == 0){
            CUDA_CHECK(cudaMemcpy(d_x, gpu_input.data(), sizeof(half) * elements_n, cudaMemcpyHostToDevice));
        }
        block_time res_time = all_times(
            stream, cudnn_handle, cublas_handle,
            batch, tokens, embeddings, hidden_channels,
            d_x, d_t, d_h, d_y, d_workspace,
            gpu_nb1.data(), gpu_ns1.data(),
            gpu_nb2.data(), gpu_ns2.data(),
            gpu_fc1.data(), gpu_b1.data(),
            gpu_fc2.data(), gpu_b2.data(),
            gpu_attn_weights, gpu_output.data(),
            stride_val, tokens_per_block,
            scale, num_heads,
            mlp_type
        );

        cout << "Comparison with CPU: " << compare_results(cpu_y, gpu_output.data()) * 100.0f<< "%" <<endl;
        res_time.print();
        res_time.to_JSON(batch, new int[3]{tokens_per_block, stride_val, mlp_type});
    }

    // - Cleanup
    
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_t));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_h));
    CUDA_CHECK(cudaFree(d_workspace));
    CUDNN_CHECK(cudnnDestroy(cudnn_handle));
    CUBLAS_CHECK(cublasLtDestroy(cublas_handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}