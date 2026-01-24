#include "../gpu_include/gpu_layer.h"
#include "../include/modules.h"
#include "../gpu_include/bench_utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>
#include <cstring>
#include <cstdlib>

#define EPS 1e-4

enum KERNEL_ID {
    GPU_LN = 6,
    CUB_LN,
    MULTI_TOK_CUB_LN,
    MULTI_TOK_ELEM_CUB_LN,
    UNROLLED_MTEC_LN
};

// 0)
benchmark_time bench_gpu_ln(
    half * d_x, half * d_y,
    half * d_scale, half * d_bias,
    int batch_size, int tokens, int embeddings
){
    int blocks_n =  batch_size * tokens;
    int threads_n = embeddings / 2;
    assert(embeddings % 2 == 0);
    benchmark_time k_time = time_kernel_variance(WARM_UP, N, 0,[&]() {
        gpu_layer_norm<<<blocks_n, threads_n>>>(
            embeddings,
            (half*)d_x, (half*)d_y,
            (half*)d_scale, (half*)d_bias,
            EPS
        );
    });
    return k_time;
}

// 1)
benchmark_time bench_cub_ln(
    half * d_x, half * d_y,
    half * d_scale, half * d_bias,
    int batch_size, int tokens
){
    int blocks_n =  batch_size * tokens;
    int threads_n = CUB_LAYER_BLOCK_DIM;
    benchmark_time k_time = time_kernel_variance(WARM_UP, N, 0,[&]() {
        cub_layer_norm<<<blocks_n, threads_n>>>(
            (half*)d_x, (half*)d_y,
            (half*)d_scale, (half*)d_bias,
            EPS,
            1
        );
    });

    return k_time;
}

// 2)
benchmark_time bench_multi_tok_cub_ln(
    half * d_x, half * d_y,
    half * d_scale, half * d_bias,
    int batch_size, int tokens, int tokens_per_block
){
    int blocks_n =  (batch_size * tokens) / tokens_per_block;
    int threads_n = CUB_LAYER_BLOCK_DIM;
    assert((batch_size * tokens) % tokens_per_block == 0);
    benchmark_time k_time = time_kernel_variance(WARM_UP, N, 0,[&]() {
        cub_layer_norm<<<blocks_n, threads_n>>>(
            (half*)d_x, (half*)d_y,
            (half*)d_scale, (half*)d_bias,
            EPS,
            tokens_per_block
        );
    });

    return k_time;
}

// 3)
benchmark_time bench_multi_tok_elem_cub_ln(
    half * d_x, half * d_y,
    half * d_scale, half * d_bias,
    int batch_size, int tokens, int tokens_per_block
){
    int blocks_n =  (batch_size * tokens) / tokens_per_block;
    int threads_n = CUB_LAYER_MULTI_BLOCK_DIM;
    assert((batch_size * tokens) % tokens_per_block == 0);
    benchmark_time k_time = time_kernel_variance(WARM_UP, N, 0,[&]() {
        multi_elem_cub_ln<<<blocks_n, threads_n>>>(
            (half*)d_x, (half*)d_y,
            (half*)d_scale, (half*)d_bias,
            EPS,
            tokens_per_block
        );
    });

    return k_time;
}

// 4) "mtec" stands for multi tokens & elements cub layer norm
benchmark_time bench_unrolled_mtec_ln(
    half * d_x, half * d_y,
    half * d_scale, half * d_bias,
    int batch_size, int tokens
){
    int blocks_n =  (batch_size * tokens) / TOKENS_PER_BLOCK;
    int threads_n = CUB_LAYER_MULTI_BLOCK_DIM;
    assert((batch_size * tokens) % TOKENS_PER_BLOCK == 0);
    benchmark_time k_time = time_kernel_variance(WARM_UP, N, 0,[&]() {
        unrolled_multi_elem_cub_ln<<<blocks_n, threads_n>>>(
            (half*)d_x, (half*)d_y,
            (half*)d_scale, (half*)d_bias,
            EPS
        );
    });

    return k_time;
}

void single_run(
    half * d_x, half * d_y,
    half * d_scale, half * d_bias,
    int batch_size, int tokens, int embeddings,
    int tokens_per_block,
    KERNEL_ID id
){
    switch (id){
        case GPU_LN:
        {
            int blocks_n =  batch_size * tokens;
            int threads_n = embeddings / 2;
            assert(embeddings % 2 == 0);
            gpu_layer_norm<<<blocks_n, threads_n>>>(
                embeddings,
                (half*)d_x, (half*)d_y,
                (half*)d_scale, (half*)d_bias,
                EPS
            );
        }
            break;
        case CUB_LN:
        {
            int blocks_n =  batch_size * tokens;
            int threads_n = CUB_LAYER_BLOCK_DIM;
            cub_layer_norm<<<blocks_n, threads_n>>>(
                (half*)d_x, (half*)d_y,
                (half*)d_scale, (half*)d_bias,
                EPS,
                1
            );
        }
            break;
        case MULTI_TOK_CUB_LN:
        {
            int blocks_n =  (batch_size * tokens) / tokens_per_block;
            int threads_n = CUB_LAYER_BLOCK_DIM;
            assert((batch_size * tokens) % tokens_per_block == 0);
            cub_layer_norm<<<blocks_n, threads_n>>>(
                (half*)d_x, (half*)d_y,
                (half*)d_scale, (half*)d_bias,
                EPS,
                tokens_per_block
            );
        }
            break;
        case MULTI_TOK_ELEM_CUB_LN:
        {
            int blocks_n =  (batch_size * tokens) / tokens_per_block;
            int threads_n = CUB_LAYER_MULTI_BLOCK_DIM;
            assert((batch_size * tokens) % tokens_per_block == 0);
            multi_elem_cub_ln<<<blocks_n, threads_n>>>(
                (half*)d_x, (half*)d_y,
                (half*)d_scale, (half*)d_bias,
                EPS,
                tokens_per_block
            );
        }
            break;
        case UNROLLED_MTEC_LN:
        {
            int blocks_n =  (batch_size * tokens) / TOKENS_PER_BLOCK;
            int threads_n = CUB_LAYER_MULTI_BLOCK_DIM;
            assert((batch_size * tokens) % TOKENS_PER_BLOCK == 0);
            unrolled_multi_elem_cub_ln<<<blocks_n, threads_n>>>(
                (half*)d_x, (half*)d_y,
                (half*)d_scale, (half*)d_bias,
                EPS
            );
        }
            break;
        default:
            break;
    }
}


int main(int argc, char** argv)
{
    bool help               = get_arg(argc, argv, "--help", 0);
    int kernel_id           = get_arg(argc, argv, "--kernel", 0);
    int batch               = get_arg(argc, argv, "--batch", 32);
    int tokens_per_block    = get_arg(argc, argv, "--tokens_per_block", 32);
    int embeddings          = EMBEDDINGS_SIZE; // MIGHT CHANGE
    int tokens              = TOKENS_NUM_VIT;

    if(help){
        cout << "LayerNorm Benchmark Options:\n"
             << " --help                      Print this help message\n"
             << " --kernel <int>              Select the kernel to benchmark (default 0: all)\n"
             << "                             0: All\n"
             << "                             1: GPU LayerNorm\n"
             << "                             2: CUB LayerNorm\n"
             << "                             3: Multi-token CUB LayerNorm\n"
             << "                             4: Multi-token & element CUB LayerNorm\n"
             << "                             5: Unrolled Multi-token & element CUB LayerNorm\n"
             << "                             6-10: Single run of the (n + 5)kernel \n"
             << " --batch <int>               Batch size (default 32)\n"
             << " --tokens_per_block <int>    Tokens per block for multi-token kernels (default 32)\n";
        return 0;
    }

    cout << "LayerNorm Benchmark\n"
              << " batch_size:          " << batch << "\n"
              << " tokens:              " << tokens << "\n"
              << " embeddings:          " << EMBEDDINGS_SIZE << "\n"
              << " warmup_iters:        " << WARM_UP << "\n"
              << " timed_iters:         " << N << "\n";

    
    // -  Memory allocation
    size_t elements_n = batch * tokens * embeddings;
    size_t total_bytes = elements_n * sizeof(half);
    size_t embeddings_bytes = embeddings * sizeof(half);

    vector<float> h_input(elements_n);
    vector<float> h_scale(embeddings);
    vector<float> h_bias(embeddings);
    vector<half> gpu_input(elements_n);
    vector<half> gpu_output(elements_n);
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
    CUDA_CHECK(cudaMemcpy(d_scale, gpu_scale.data(), embeddings_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, gpu_bias.data(), embeddings_bytes, cudaMemcpyHostToDevice));

    // - Reference creation
    RowVector cpu_scale(h_scale.data(), embeddings);
    RowVector cpu_bias(h_bias.data(), embeddings);

    LayerNorm cpu_ln(embeddings, EPS, true);
    cpu_ln.move_g(cpu_scale);
    cpu_ln.move_b(cpu_bias);

    Tensor cpu_x(h_input.data(), elements_n, batch, tokens, embeddings);
    cpu_ln(cpu_x);

    if (kernel_id == 0 || kernel_id == 1){        
        cout << "|| Gpu layer norm ||" << endl;
        benchmark_time avg_time = bench_gpu_ln(
            (half*)d_input, (half*)d_output,
            (half*)d_scale, (half*)d_bias,
            batch, tokens, embeddings
        );

        CUDA_CHECK(cudaMemcpy(gpu_output.data(), d_output, total_bytes, cudaMemcpyDeviceToHost));
        cout << "last iteration Absolute Mean Relative Error: " << compare_results(cpu_x, gpu_output.data()) * 100.0f<< "%"  << endl;
        avg_time.print();
        avg_time.to_JSON(batch, new int[2]{2, 1});
    }
    if (kernel_id == 0 || kernel_id == 2){
        cout << "|| CUB layer norm ||" << endl;
        benchmark_time avg_time = bench_cub_ln(
            (half*)d_input, (half*)d_output,
            (half*)d_scale, (half*)d_bias,
            batch, tokens
        );

        CUDA_CHECK(cudaMemcpy(gpu_output.data(), d_output, total_bytes, cudaMemcpyDeviceToHost));
        cout << "last iteration Absolute Mean Relative Error: " << compare_results(cpu_x, gpu_output.data()) * 100.0f<< "%"  << endl;
        avg_time.print();
        avg_time.to_JSON(batch, new int[2]{2, 1});
    }
    if (kernel_id == 0 || kernel_id == 3){
        cout << "|| Multi-token CUB layer norm ||" << endl;
        benchmark_time avg_time = bench_multi_tok_cub_ln(
            (half*)d_input, (half*)d_output,
            (half*)d_scale, (half*)d_bias,
            batch, tokens, tokens_per_block
        );

        CUDA_CHECK(cudaMemcpy(gpu_output.data(), d_output, total_bytes, cudaMemcpyDeviceToHost));
        cout << "last iteration Absolute Mean Relative Error: " << compare_results(cpu_x, gpu_output.data()) * 100.0f<< "%"  << endl;
    avg_time.print();
        avg_time.to_JSON(batch, new int[2]{2, tokens_per_block});
    }
    if (kernel_id == 0 || kernel_id == 4){
        cout << "|| Multi-token & element CUB layer norm ||" << endl;
        benchmark_time avg_time = bench_multi_tok_elem_cub_ln(
            (half*)d_input, (half*)d_output,
            (half*)d_scale, (half*)d_bias,
            batch, tokens, tokens_per_block
        );
        CUDA_CHECK(cudaMemcpy(gpu_output.data(), d_output, total_bytes, cudaMemcpyDeviceToHost));
        cout << "last iteration Absolute Mean Relative Error: " << compare_results(cpu_x, gpu_output.data()) * 100.0f<< "%"  << endl;
    avg_time.print();
        avg_time.to_JSON(batch, new int[2]{ELEMENTS_PER_TH, tokens_per_block});
    }
    if (kernel_id == 0 || kernel_id == 5){
        cout << "|| Unrolled Multi-token & element CUB layer norm ||" << endl;
        benchmark_time avg_time = bench_unrolled_mtec_ln(
            (half*)d_input, (half*)d_output,
            (half*)d_scale, (half*)d_bias,
            batch, tokens
        );
        CUDA_CHECK(cudaMemcpy(gpu_output.data(), d_output, total_bytes, cudaMemcpyDeviceToHost));
        cout << "last iteration Absolute Mean Relative Error: " << compare_results(cpu_x, gpu_output.data()) * 100.0f<< "%"  << endl;
        avg_time.print();
        avg_time.to_JSON(batch, new int[2]{ELEMENTS_PER_TH, TOKENS_PER_BLOCK});
    }
    if  (kernel_id > 5 && kernel_id <= 10){
        cout << "|| Single run of selected kernel " << kernel_id << " ||" << endl;
        single_run(
            (half*)d_input, (half*)d_output,
            (half*)d_scale, (half*)d_bias,
            batch, tokens, embeddings,
            tokens_per_block,
            KERNEL_ID(kernel_id)
        );
        
        CUDA_CHECK(cudaMemcpy(gpu_output.data(), d_output, total_bytes, cudaMemcpyDeviceToHost));
        cout << "Absolute Mean Relative Error: " << compare_results(cpu_x, gpu_output.data()) * 100.0f<< "%"  << endl;
    }

    // - Cleanup
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_scale));
    CUDA_CHECK(cudaFree(d_bias));

    return 0;
}
