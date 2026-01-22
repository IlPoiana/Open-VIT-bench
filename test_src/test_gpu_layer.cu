#include "../gpu_include/gpu_layer.h"
#include "../include/modules.h"
#include <bits/stdc++.h>
#include <curand_kernel.h>

#define BATCH 4

// Returns the MRE of the cpu `y` Tensor and `gpu_y`. Attention! There is a tolerance instroduced to avoid division by zero
float compare_results(Tensor &y, half * gpu_y){
    float tolerance = 1e-3f;
    double avg = 0;
    float gpu_val;
    float total_elem_num = y.get_B() * y.get_N() * y.get_C();
    for(u_int b = 0; b < y.get_B(); b++){
        for(u_int t = 0; t < y.get_N(); t++){
            for(u_int c = 0; c < y.get_C(); c++){
                assert(!isnanf( y.at(b,t,c)));
                assert(!isnanf( __half2float(gpu_y[c + y.get_C() * t + y.get_C() * y.get_N() * b])));
                gpu_val = __half2float(gpu_y[c + y.get_C() * t + y.get_C() * y.get_N() * b]);
                avg += 
                    (
                        (double)abs(y.at(b,t,c) - gpu_val)
                        /
                        (double)max(abs(y.at(b,t,c)), tolerance)
                    )
                    / total_elem_num;
            }
        }
    }
    return float(avg);
}

void cpu_gpu_comparison(){
    /*
        It is possible to modify these parameters,
        but be aware that the loop unrolled method requires MACRO sizes defined 
    */
    u_int total_elements_num = BATCH * EMBEDDINGS_SIZE * TOKENS_NUM;
    cout << "Tensor: [" << BATCH << ","<< TOKENS_NUM << "," << EMBEDDINGS_SIZE << "]" << endl;
    cout << "   || Kernels parameters||     " << "\n"
        << "    1:" <<"\n"
        << "        Elements per thread:" << 2 << "\n"
        << "        Blocks number:      " << BATCH * TOKENS_NUM<< "\n"
        << "        Block Dimension:    " << LAYER_BLOCK_DIM << "\n"
        << "    2:" <<"\n"
        << "        Elements per thread:" << 2 << "\n"
        << "        Blocks number:      " << BATCH * TOKENS_NUM<< "\n"
        << "        Block Dimension:    " << CUB_LAYER_BLOCK_DIM << "\n"
        << "    3:" <<"\n"
        << "        Elements per thread:" << 2 << "\n"
        << "        Tokens per block:   " << TOKENS_PER_BLOCK << "\n"
        << "        Blocks number:      " << (TOKENS_NUM * BATCH) / TOKENS_PER_BLOCK<< "\n"
        << "        Block Dimension:    " << LAYER_BLOCK_DIM << "\n"
        << "    4:" <<"\n"
        << "        Elements per thread:" << 4 << "\n"
        << "        Tokens per block:   " << TOKENS_PER_BLOCK << "\n"
        << "        Blocks number:      " << (TOKENS_NUM * BATCH) / TOKENS_PER_BLOCK<< "\n"
        << "        Block Dimension:    " << LAYER_BLOCK_DIM << "\n"
        << "    5:" <<"\n"
        << "        Elements per thread:" << 4 << "\n"
        << "        Tokens per block:   " << TOKENS_PER_BLOCK << "\n"
        << "        Blocks number:      " << (TOKENS_NUM * BATCH) / TOKENS_PER_BLOCK<< "\n"
        << "        Block Dimension:    " << LAYER_BLOCK_DIM << "\n";
        
    float * x_data = (float *)malloc(sizeof(float) * total_elements_num);
    float b2_data[EMBEDDINGS_SIZE], g_data[EMBEDDINGS_SIZE];
    
    // -Random initialization
    random_device rd;          
    mt19937 gen(rd());         
    uniform_real_distribution<float> dist(-1.0f, 1.0f);

    size_t loop_range = total_elements_num;
    for(size_t i = 0; i < loop_range; i++){
        if(i < EMBEDDINGS_SIZE){
            b2_data[i] = dist(gen);
            g_data[i] = dist(gen);
        }
        if(i < total_elements_num){
            x_data[i] = dist(gen);
        }
    }
    
    double epsilon = 1e-5;
    half * d_x, * d_y, * d_bias, * d_scale;
    half * gpu_y = (half *)malloc(sizeof(half) * total_elements_num);
    half gpu_epsilon = __double2half(epsilon);
    h_tensor gpu_x(x_data,BATCH,EMBEDDINGS_SIZE,1,TOKENS_NUM);
    mtx gpu_bias(b2_data,1,EMBEDDINGS_SIZE);
    mtx gpu_scale(g_data,1,EMBEDDINGS_SIZE);
    
    CUDA_CHECK(cudaMalloc(&d_x, sizeof(half) * total_elements_num));
    CUDA_CHECK(cudaMalloc(&d_y, sizeof(half) * total_elements_num));
    CUDA_CHECK(cudaMalloc(&d_bias, sizeof(half)* EMBEDDINGS_SIZE));
    CUDA_CHECK(cudaMalloc(&d_scale, sizeof(half)* EMBEDDINGS_SIZE));

    CUDA_CHECK(cudaMemcpy(d_x, gpu_x.data, sizeof(half) * total_elements_num, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, gpu_bias.data, sizeof(half) * EMBEDDINGS_SIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scale,gpu_scale.data , sizeof(half) * EMBEDDINGS_SIZE, cudaMemcpyHostToDevice));

    
    // -- 0) CPU reference
    cout<< "CPU reference: " << endl;
    RowVector g(g_data, EMBEDDINGS_SIZE);
    RowVector b2(b2_data, EMBEDDINGS_SIZE);
    
    LayerNorm ln(EMBEDDINGS_SIZE, epsilon, true);
    ln.move_g(g);
    ln.move_b(b2);

    Tensor y(x_data,total_elements_num, BATCH, TOKENS_NUM, EMBEDDINGS_SIZE);
    ln(y);
    float * cpu_y = (float *)malloc(sizeof(float) * total_elements_num);
    cpu_y = y.get_data();

    // -- 1) GPU fused kernel, half precision and sh. mem for reduction
    u_int block_number = BATCH * TOKENS_NUM;
    assert(EMBEDDINGS_SIZE < SH_MEM_DIM);
    
    gpu_layer_norm<<<block_number,LAYER_BLOCK_DIM>>>(EMBEDDINGS_SIZE,d_x,d_y,d_scale,d_bias, gpu_epsilon);
    CHECK_LAUNCH();

    CUDA_CHECK(cudaMemcpy(gpu_y, d_y, sizeof(half) * total_elements_num, cudaMemcpyDeviceToHost));    
    cout << "GPU Naive avg. difference: " << compare_results(y, gpu_y) * 100 << "%" << endl;

    // -- 2) GPU-1 + CUB reduction 
    CUDA_CHECK(cudaMemcpy(d_x, gpu_x.data, sizeof(half) * total_elements_num, cudaMemcpyHostToDevice));
    
    cub_layer_norm<<<block_number, CUB_LAYER_BLOCK_DIM>>>(d_x, d_y, d_scale, d_bias, gpu_epsilon, 1);
    
    CUDA_CHECK(cudaMemcpy(gpu_y, d_y, sizeof(half) * total_elements_num, cudaMemcpyDeviceToHost));
    cout << "GPU block avg. difference: " << compare_results(y, gpu_y) * 100 << "%" << endl;
        
    // -- 3) GPU-2 + multi token per block 
    block_number = (TOKENS_NUM * BATCH) / TOKENS_PER_BLOCK;
    CUDA_CHECK(cudaMemcpy(d_x, gpu_x.data, sizeof(half) * total_elements_num, cudaMemcpyHostToDevice));
    
    cub_layer_norm<<<block_number, CUB_LAYER_BLOCK_DIM>>>(d_x, d_y, d_scale, d_bias, gpu_epsilon, TOKENS_PER_BLOCK);
    
    CUDA_CHECK(cudaMemcpy(gpu_y, d_y, sizeof(half) * total_elements_num, cudaMemcpyDeviceToHost));
    cout << "GPU CUB avg. difference: " << compare_results(y, gpu_y) * 100 << "%" << endl;

    // -- 4) GPU-3 + multi element per thread(not fixed at two)
    CUDA_CHECK(cudaMemcpy(d_x, gpu_x.data, sizeof(half) * total_elements_num, cudaMemcpyHostToDevice));
    assert(BATCH % TOKENS_PER_BLOCK == 0);
    
    multi_elem_cub_ln<<<block_number, CUB_LAYER_MULTI_BLOCK_DIM>>>(d_x, d_y, d_scale, d_bias, gpu_epsilon, TOKENS_PER_BLOCK);
    
    CUDA_CHECK(cudaMemcpy(gpu_y, d_y, sizeof(half) * total_elements_num, cudaMemcpyDeviceToHost));
    cout << "GPU CUB MULTI avg. difference: " << compare_results(y, gpu_y) * 100 << "%" << endl;
    
    // -- 5) GPU-4 + loop unrolling for fixed iterations number loops
    CUDA_CHECK(cudaMemcpy(d_x, gpu_x.data, sizeof(half) * total_elements_num, cudaMemcpyHostToDevice));
    
    unrolled_multi_elem_cub_ln<<<block_number, CUB_LAYER_MULTI_BLOCK_DIM>>>(d_x, d_y, d_scale, d_bias, gpu_epsilon);
    
    CUDA_CHECK(cudaMemcpy(gpu_y, d_y, sizeof(half) * total_elements_num, cudaMemcpyDeviceToHost));
    cout << "GPU UNROLLED CUB MULTI avg. difference: " << compare_results(y, gpu_y) * 100 << "%" << endl;

    // -- 6) GPU-5: single thread cub reduction, multi token per block (dev purpose)
    CUDA_CHECK(cudaMemcpy(d_x, gpu_x.data, sizeof(half) * total_elements_num, cudaMemcpyHostToDevice));
    
    block_number = total_elements_num / (EMBEDDINGS_SIZE *  TOKENS_PER_BLOCK);
    assert((total_elements_num % (EMBEDDINGS_SIZE *  TOKENS_PER_BLOCK))== 0);
    
    cub_single_layer_norm<<<block_number, EMBEDDINGS_SIZE>>>(d_x, d_y, d_scale, d_bias, gpu_epsilon, TOKENS_PER_BLOCK);
    
    CUDA_CHECK(cudaMemcpy(gpu_y, d_y, sizeof(half) * total_elements_num, cudaMemcpyDeviceToHost));
    cout << "GPU SINGLE LAYER NORM avg. difference: " << compare_results(y, gpu_y) * 100 << "%" << endl;


    cudaFree(d_x); cudaFree(d_y); cudaFree(d_bias); cudaFree(d_scale);
    free(gpu_y); free(x_data);
    return;
}

int main() {
    cpu_gpu_comparison();
    
    return 0;
}