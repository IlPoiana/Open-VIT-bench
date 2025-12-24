#include "../gpu_include/gpu_layer.h"
#include "../include/modules.h"
#include "../gpu_include/bench_utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>
#include <cstring>
#include <cstdlib>

#define LAYER_WARM_UP 20
#define LAYER_N 100
#define EPS 1e-4

// 0)
void bench_gpu_ln(
    half * d_x, half * d_y,
    half * d_scale, half * d_bias,
    int batch_size, int tokens, int embeddings
){
    int blocks_n =  batch_size * tokens;
    int threads_n = embeddings / 2;
    assert(embeddings % 2 == 0);
    float avg_ms = time_kernel(LAYER_WARM_UP, LAYER_N, [&]() {
        gpu_layer_norm<<<blocks_n, threads_n>>>(
            embeddings,
            (half*)d_x, (half*)d_y,
            (half*)d_scale, (half*)d_bias,
            EPS
        );
    });

    cout << "Average time for kernel 0(gpu_layer_norm): " << avg_ms << "ms" << endl;
}

// 1)
void bench_cub_ln(half * d_x, half * d_y,
    half * d_scale, half * d_bias,
    int batch_size, int tokens
){
    int blocks_n =  batch_size * tokens;
    int threads_n = CUB_LAYER_BLOCK_DIM;
    float avg_ms = time_kernel(LAYER_WARM_UP, LAYER_N, [&]() {
        cub_layer_norm<<<blocks_n, threads_n>>>(
            (half*)d_x,
            (half*)d_scale, (half*)d_bias,
            EPS,
            1
        );
    });

    cout << "Average time for kernel 1(cub_layer_norm): " << avg_ms << "ms" << endl;
}

// 2)
void bench_multi_tok_cub_ln(
    half * d_x, half * d_y,
    half * d_scale, half * d_bias,
    int batch_size, int tokens, int tokens_per_block
){
    int blocks_n =  (batch_size * tokens) / tokens_per_block;
    int threads_n = CUB_LAYER_BLOCK_DIM;
    assert((batch_size * tokens) % tokens_per_block == 0);
    float avg_ms = time_kernel(LAYER_WARM_UP, LAYER_N, [&]() {
        cub_layer_norm<<<blocks_n, threads_n>>>(
            (half*)d_x,
            (half*)d_scale, (half*)d_bias,
            EPS,
            tokens_per_block
        );
    });

    cout << "Average time for kernel 2(multi_tok_cub_layer_norm): " << avg_ms << "ms" << endl;
}

// 3)
void bench_multi_tok_elem_cub_ln(
    half * d_x, half * d_y,
    half * d_scale, half * d_bias,
    int batch_size, int tokens, int tokens_per_block
){
    int blocks_n =  (batch_size * tokens) / tokens_per_block;
    int threads_n = CUB_LAYER_MULTI_BLOCK_DIM;
    assert((batch_size * tokens) % tokens_per_block == 0);
    float avg_ms = time_kernel(LAYER_WARM_UP, LAYER_N, [&]() {
        multi_elem_cub_ln<<<blocks_n, threads_n>>>(
            (half*)d_x,
            (half*)d_scale, (half*)d_bias,
            EPS,
            tokens_per_block
        );
    });

    cout << "Average time for kernel 3(multi_tok_elem_cub_layer_norm): " << avg_ms << "ms" << endl;
}

// 4) "mtec" stands for multi tokens & elements cub layer norm
void bench_unrolled_mtec_ln(
    half * d_x, half * d_y,
    half * d_scale, half * d_bias,
    int batch_size, int tokens
){
    int blocks_n =  (batch_size * tokens) / TOKENS_PER_BLOCK;
    int threads_n = CUB_LAYER_MULTI_BLOCK_DIM;
    assert((batch_size * tokens) % TOKENS_PER_BLOCK == 0);
    float avg_ms = time_kernel(LAYER_WARM_UP, LAYER_N, [&]() {
        unrolled_multi_elem_cub_ln<<<blocks_n, threads_n>>>(
            (half*)d_x,
            (half*)d_scale, (half*)d_bias,
            EPS
        );
    });

    cout << "Average time for kernel 4(unrolled_mtec_ln): " << avg_ms << "ms" << endl;
}


int main(int argc, char** argv)
{
    int kernel_id           = get_arg(argc, argv, "--kernel", 0);
    int batch               = get_arg(argc, argv, "--batch", 32);
    int tokens_per_block    = get_arg(argc, argv, "--tokens_per_block", 32);
    int embeddings          = EMBEDDINGS_SIZE; // MIGHT CHANGE
    int tokens              = TOKENS_NUM_VIT;

    cout << "LayerNorm Benchmark\n"
              << " batch_size:          " << batch << "\n"
              << " tokens_per_block:    " << (kernel_id == 5 ? TOKENS_PER_BLOCK : tokens_per_block) << "\n"
              << " elements_per_thread: " << ELEMENTS_PER_TH << "\n"
              << " warmup_iters:        " << LAYER_WARM_UP << "\n"
              << " timed_iters:         " << LAYER_N << "\n";

    
    // -  Memory allocation
    size_t elements_n = batch * tokens * embeddings;
    size_t total_bytes = elements_n * sizeof(half);
    size_t embeddings_bytes = embeddings * sizeof(half);

    vector<float> h_input(elements_n);
    vector<float> h_scale(embeddings);
    vector<float> h_bias(embeddings);
    vector<half> gpu_input(elements_n);
    vector<half> gpu_scale(embeddings);
    vector<half> gpu_bias(embeddings);

    
    random_device rd;          
    mt19937 gen(rd());         
    uniform_real_distribution<float> dist(0.1f, 1.0f);

    for(size_t i = 0; i < elements_n; i++){
        if(i < embeddings){
            h_scale[i] = dist(gen);
            h_bias[i] = dist(gen);
        }

        h_input[i] = dist(gen);

    }

    f32_to_f16(h_input.data(), gpu_input.data(), elements_n);
    f32_to_f16(h_scale.data(), gpu_scale.data(), embeddings);
    f32_to_f16(h_bias.data(), gpu_bias.data(), embeddings);

    void *d_input = nullptr, *d_output = nullptr;
    void *d_scale = nullptr, *d_bias = nullptr; 

    CUDA_CHECK(cudaMalloc(&d_input, total_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, total_bytes));
    CUDA_CHECK(cudaMalloc(&d_scale, embeddings_bytes));
    CUDA_CHECK(cudaMalloc(&d_bias, embeddings_bytes));
    CUDA_CHECK(cudaMemcpy(d_input, gpu_input.data(), total_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scale, gpu_scale.data(), total_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, gpu_bias.data(), total_bytes, cudaMemcpyHostToDevice));

    // - Reference creation
    RowVector cpu_scale(h_scale.data(), embeddings);
    RowVector cpu_bias(h_bias.data(), embeddings);

    LayerNorm cpu_ln(embeddings, EPS, true);
    cpu_ln.move_g(cpu_scale);
    cpu_ln.move_b(cpu_bias);

    Tensor cpu_x(h_input.data(), elements_n, batch, tokens, embeddings);
    cpu_ln(cpu_x);

    if (kernel_id == 0 || kernel_id == 1)
        bench_gpu_ln(
            (half*)d_input, (half*)d_output,
            (half*)d_scale, (half*)d_bias,
            batch, tokens, embeddings
        );

    if (kernel_id == 0 || kernel_id == 2)
        bench_cub_ln(
            (half*)d_input, (half*)d_output,
            (half*)d_scale, (half*)d_bias,
            batch, tokens
        );

    if (kernel_id == 0 || kernel_id == 3)
        bench_multi_tok_cub_ln(
            (half*)d_input, (half*)d_output,
            (half*)d_scale, (half*)d_bias,
            batch, tokens, tokens_per_block
        );

    if (kernel_id == 0 || kernel_id == 4)
        bench_multi_tok_elem_cub_ln(
            (half*)d_input, (half*)d_output,
            (half*)d_scale, (half*)d_bias,
            batch, tokens, tokens_per_block
        );

    if (kernel_id == 0 || kernel_id == 5)
        bench_unrolled_mtec_ln(
            (half*)d_input, (half*)d_output,
            (half*)d_scale, (half*)d_bias,
            batch, tokens
        );
    
    // - Cleanup
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
