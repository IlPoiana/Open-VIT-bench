#include "../gpu_include/gpu_layer.h"
#include "../include/modules.h"
#include <bits/stdc++.h>
#include <curand_kernel.h>

#define BATCH 8

__global__ void generate_reference(float * x){
    u_int idx = blockDim.x * blockIdx.x + threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init(0, /*subsequence*/ idx, /*offset*/ 0, &state);
    if(idx < BATCH*EMBEDDINGS_SIZE * TOKENS_NUM)
        x[idx] = (curand_uniform(&state) * 2) - 1.0f;
        // x[idx] = 1.0f;
}

float to_percentage(float x, float range = 2.0f){
    return (x / range) * 100;
}

void gpu_layer_norm(
    u_int B, u_int T,u_int C,
    half * x_data, half * y,
    half * bias, half * scale, half epsilon
){
    half * d_x, * d_y, * d_bias, * d_scale;
    CUDA_CHECK(cudaMalloc(&d_x, sizeof(half) * B * T * C));
    CUDA_CHECK(cudaMalloc(&d_y, sizeof(half) * B * T * C));
    CUDA_CHECK(cudaMalloc(&d_bias, sizeof(half)* C));
    CUDA_CHECK(cudaMalloc(&d_scale, sizeof(half)* C));

    CUDA_CHECK(cudaMemcpy(d_x, x_data, sizeof(half) * B * T * C, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, bias, sizeof(half) * C, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scale,scale , sizeof(half) * C, cudaMemcpyHostToDevice));

    u_int block_number = B * T;
    assert(C < SH_MEM_DIM);  
    gpu_layer_norm<<<block_number,LAYER_BLOCK_DIM>>>(C,d_x,d_y,d_scale,d_bias, epsilon);
    CHECK_LAUNCH();
    CUDA_CHECK(cudaMemcpy(y, d_y, sizeof(half) * B * T * C, cudaMemcpyDeviceToHost));


    cudaFree(d_x); cudaFree(d_y); cudaFree(d_bias); cudaFree(d_scale);  

}

void cpu_gpu_comparison(){
    cout << "GPU Test Modules" << endl;
    // B = 3, T = 7, C = 4
    const size_t B = 3;
    const size_t T = 7;
    const size_t C = 4;

    vit_float x_data[3*7*4] = {
        -0.886269, 0.853940, -0.236085, -0.469522,
        -0.586181, 0.459059, -0.009484, -0.088119,
        0.879872, 0.126726, 0.619740, 0.510598,
        0.109471, 0.387463, -0.848270, 0.208436,
        -0.009994, 0.422368, 0.836362, 0.025621,
        -0.726769, 0.423409, -0.677464, -0.135067,
        0.984270, 0.213376, 0.219383, -0.203643,

         0.782612, 0.104527, 0.037466, 0.634141,
        0.410488, -0.903342, -0.450407, 0.221993,
        -0.470002, -0.506979, 0.069336, 0.186896,
        -0.722772, -0.628325, 0.117727, 0.201875,
        -0.009994, 0.422368, 0.836362, 0.025621,
        -0.726769, 0.423409, -0.677464, -0.135067,
        0.984270, 0.213376, 0.219383, -0.203643,

        -0.880071, 0.165039, -0.600531, -0.426892,
        -0.420394, -0.907864, -0.033296, -0.437744,
        0.394329, -0.289080, -0.666751, -0.737631,
        0.014464, -0.295355, -0.917562, 0.409367,
        -0.009994, 0.422368, 0.836362, 0.025621,
        -0.726769, 0.423409, -0.677464, -0.135067,
        0.984270, 0.213376, 0.219383, -0.203643
    };
    
    

    Tensor x(x_data, B*T*C, B, T, C);
    

    // LayerNorm Test

    vit_float g_data[C] = {-0.017925, 0.550008, -0.043667, -0.032563};
    RowVector g(g_data, C);
    

    vit_float b2_data[C] = {-0.354565, 0.738078, -0.155186, 0.531506};
    RowVector b2(b2_data, C);
    
    
    double epsilon = 0.00001;

    h_tensor x_half(x_data,B,C,1,T);
    mtx bias(b2_data,1,C);
    mtx scale(g_data,1,C);
    half y_half[B*T*C];

    gpu_layer_norm(
        B,T,C,
        x_half.data,y_half,
        bias.data,scale.data,
        __double2half(epsilon)
    );
    

    cout << "### x" << endl;
    x.print();
    cout << "### g" << endl;
    g.print();
    cout << "### b2" << endl;
    b2.print();

    LayerNorm ln(C, epsilon, true);
    ln.move_g(g);
    ln.move_b(b2);

    Tensor y(x_data,B*T*C, B, T, C);
    ln(y);
    cout << "Reference CPU" << endl;
    y.print();
    
    float gpu_y[B*T*C];

    cout << "GPU Layer Norm" <<endl;
    for(u_int b = 0; b < B; b++){
        cout<< "B:" << b << "[" << endl;
        for(u_int t = 0; t < T; t++){
            cout << "[";
            for(u_int c = 0; c < C; c++){
                gpu_y[(b*T*C) + (t * C) + c] = __half2float( y_half[(b*T*C) + (t * C) + c]);
                cout << " " << gpu_y[(b*T*C) + (t * C) + c] << " ";
            }
            cout << "]"<<endl;
        }
        cout << "]" << endl;
    }

    float avg_difference = 0;
    for(u_int b = 0; b < B; b++){
        for(u_int t = 0; t < T; t++){
            for(u_int c = 0; c < C; c++){
                avg_difference += abs(abs( y.at(b,t,c)) - abs(gpu_y[(b*T*C) + (t * C) + c])); 
            }
        }
    }
    avg_difference /= (B *T *C);
    cout << avg_difference << endl;
    
    return;
}

void gpu_comparison(){
    u_int total_elements_num = BATCH * EMBEDDINGS_SIZE * TOKENS_NUM;
    float difference = 0.0f;
    
    float * x_data = (float *)malloc(sizeof(float) * total_elements_num);
    float b2_data[EMBEDDINGS_SIZE] = {0.0f}, g_data[EMBEDDINGS_SIZE];
    for(u_int i = 0;i < EMBEDDINGS_SIZE; i ++){
        g_data[i] = 1.0f; 
    }

    cout << "Tensor: [" << BATCH << ","<< TOKENS_NUM << "," << EMBEDDINGS_SIZE << "]" << endl;


    //Initialize x_data
    cout << "initializing reference" << endl;
    float * d_ref;
    cudaMalloc(&d_ref, sizeof(float) * total_elements_num);
    u_int block_size = 512,blocks_num = (total_elements_num / block_size) + 1; 
    generate_reference<<<blocks_num, block_size>>>(d_ref);
    CHECK_LAUNCH();
    cudaMemcpy(x_data, d_ref, sizeof(float) * total_elements_num, cudaMemcpyDeviceToHost);

    double epsilon = 0.00001;
    
    // -- 0) CPU reference
    cout<< "CPU reference: " << endl;
    RowVector g(g_data, EMBEDDINGS_SIZE);
    RowVector b2(b2_data, EMBEDDINGS_SIZE);
    
    LayerNorm ln(EMBEDDINGS_SIZE, epsilon, true);
    ln.move_g(g);
    ln.move_b(b2);

    Tensor y(x_data,total_elements_num, BATCH, TOKENS_NUM, EMBEDDINGS_SIZE);
    ln(y);
    // y.print();
    float * cpu_y = (float *)malloc(sizeof(float) * total_elements_num);
    cpu_y = y.get_data();

    // -- 1) GPU fused kernel, half precision and sh. mem for reduction
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

    u_int block_number = BATCH * TOKENS_NUM;
    assert(EMBEDDINGS_SIZE < SH_MEM_DIM);
    // cout << "starting gpu ln"<<endl;  
    gpu_layer_norm<<<block_number,LAYER_BLOCK_DIM>>>(EMBEDDINGS_SIZE,d_x,d_y,d_scale,d_bias, gpu_epsilon);
    CHECK_LAUNCH();
    // cout << "finished ln"<<endl;
    CUDA_CHECK(cudaMemcpy(gpu_y, d_y, sizeof(half) * total_elements_num, cudaMemcpyDeviceToHost));
    difference = result_check_fp16(gpu_y,cpu_y, total_elements_num);
    cout << "GPU Naive avg. difference: " << to_percentage(difference) << endl;

    // -- 2) GPU-1 + CUB reduction 
    CUDA_CHECK(cudaMemcpy(d_x, gpu_x.data, sizeof(half) * total_elements_num, cudaMemcpyHostToDevice));
    
    cub_layer_norm<<<block_number, CUB_LAYER_BLOCK_DIM>>>(d_x, d_scale, d_bias, gpu_epsilon, 1);
    
    CUDA_CHECK(cudaMemcpy(gpu_y, d_x, sizeof(half) * total_elements_num, cudaMemcpyDeviceToHost));
    difference = result_check_fp16(gpu_y, cpu_y, total_elements_num);
    cout << "GPU block avg. difference: " << to_percentage(difference) << endl;
        
    // -- 3) GPU-2 + multi token per block 
    block_number = (TOKENS_NUM * BATCH) / TOKENS_PER_BLOCK;
    CUDA_CHECK(cudaMemcpy(d_x, gpu_x.data, sizeof(half) * total_elements_num, cudaMemcpyHostToDevice));
    
    cub_layer_norm<<<block_number, CUB_LAYER_BLOCK_DIM>>>(d_x, d_scale, d_bias, gpu_epsilon);
    
    CUDA_CHECK(cudaMemcpy(gpu_y, d_x, sizeof(half) * total_elements_num, cudaMemcpyDeviceToHost));
    difference = result_check_fp16(gpu_y, cpu_y, total_elements_num);
    cout << "GPU CUB avg. difference: " << to_percentage(difference) << endl;

    // -- 4) GPU-3 + multi element per thread(not fixed at two)
    CUDA_CHECK(cudaMemcpy(d_x, gpu_x.data, sizeof(half) * total_elements_num, cudaMemcpyHostToDevice));
    
    multi_elem_cub_ln<<<block_number, CUB_LAYER_MULTI_BLOCK_DIM>>>(d_x, d_scale, d_bias, gpu_epsilon);
    
    CUDA_CHECK(cudaMemcpy(gpu_y, d_x, sizeof(half) * total_elements_num, cudaMemcpyDeviceToHost));
    difference = result_check_fp16(gpu_y, cpu_y, total_elements_num);
    cout << "GPU CUB MULTI avg. difference: " << to_percentage(difference) << endl;
    // -- 5) GPU-4 + loop unrolling for fixed iterations number loops
    CUDA_CHECK(cudaMemcpy(d_x, gpu_x.data, sizeof(half) * total_elements_num, cudaMemcpyHostToDevice));
    
    unrolled_multi_elem_cub_ln<<<block_number, CUB_LAYER_MULTI_BLOCK_DIM>>>(d_x, d_scale, d_bias, gpu_epsilon);
    
    CUDA_CHECK(cudaMemcpy(gpu_y, d_x, sizeof(half) * total_elements_num, cudaMemcpyDeviceToHost));
    difference = result_check_fp16(gpu_y, cpu_y, total_elements_num);
    cout << "GPU UNROLLED CUB MULTI avg. difference: " << to_percentage(difference) << endl;

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_bias); cudaFree(d_scale);
    return;
}

int main() {
    test_type test = GPU_COMPARISON;
    if(test == CPU_COMPARISON){
        cpu_gpu_comparison();
    }
    else{
        gpu_comparison();
    }

    return 0;
}