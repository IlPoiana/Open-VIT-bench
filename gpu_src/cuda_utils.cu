#include "../gpu_include/cuda_utils.h"

#include <assert.h>
#include <iostream>
#include <algorithm>

conv_kernel_shape::conv_kernel_shape(){
    W=0;
    H=0;
    w_stride=0;
    h_stride=0;
    in_channels=0;
    out_channels=0;
}

conv_kernel_shape::conv_kernel_shape(int * array_shape){
    in_channels=array_shape[0];
    out_channels=array_shape[1];
    H=array_shape[2];
    W=array_shape[3];
    h_stride=array_shape[4];
    w_stride=array_shape[5];
}


__global__ void add_strided(half * x, half * val_array, u_int N) {
    u_int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        x[i] += val_array[i];
    }
    return;
}

float result_check_fp16(half * x, half * reference, size_t n){
    float sum = 0.0;
    for (size_t i = 0; i < n; ++i) sum += fabs(__half2float( reference[i]) - __half2float(x[i]));
    return sum / (float)n; 
}

float result_check_fp16(half * x, float * reference, size_t n){
    float sum = 0.0;
    for (size_t i = 0; i < n; ++i) sum += fabs(reference[i] - __half2float(x[i]));
    return sum / (float)n; 
}

// initialize the half device array with random (Set seed) values, every value generated (between -1.0 and 1.0) is then scaled by `scale`
__global__ void generate_reference(half * d_x, u_int total_n, float scale, u_long seed){
    u_int idx = blockDim.x * blockIdx.x + threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, /*subsequence*/ idx, /*offset*/ 0, &state);
    if(idx < total_n)
        d_x[idx] = __float2half(((curand_uniform(&state) * 2) - 1.0f) * scale);
}

// initialize the float device array with random (Set seed) values, every value generated (between -1.0 and 1.0) is then scaled by `scale`
__global__ void generate_reference(float * d_x, u_int total_n, float scale, u_long seed){
    u_int idx = blockDim.x * blockIdx.x + threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, /*subsequence*/ idx, /*offset*/ 0, &state);
    if(idx < total_n)
        d_x[idx] = ((curand_uniform(&state) * 2) - 1.0f) * scale;
}

void rand_init(float * h_out, u_int n, float rand_scale, u_long seed){
    u_int blocks_n = (n / 256) + 1;
    float * d_buffer; cudaMalloc(&d_buffer, sizeof(float) * n);
    generate_reference<<<blocks_n, 256>>>(d_buffer, n, rand_scale, seed); 
    CUDA_CHECK(cudaMemcpy(h_out,d_buffer,sizeof(float) * n,cudaMemcpyDeviceToHost));    // GPU Single Stream
}

void rand_init(half * h_out, u_int n, float rand_scale, u_long seed){
    u_int blocks_n = (n / 256) + 1;
    half * d_buffer; cudaMalloc(&d_buffer, sizeof(half) * n);
    generate_reference<<<blocks_n, 256>>>(d_buffer, n, rand_scale, seed); 
    CUDA_CHECK(cudaMemcpy(h_out,d_buffer,sizeof(half) * n,cudaMemcpyDeviceToHost));    // GPU Single Stream
}

void linearize(float * data, float * linearized_data, picture_shape input_img, conv_kernel_shape kernel)
{
    int B = input_img.B;
    int C = input_img.C;
    int H = input_img.H;
    int W = input_img.W;
    int P_H = kernel.H;
    int P_W = kernel.W;
    int stride_h = kernel.h_stride;
    int stride_w = kernel.w_stride;

    //kernel.get_H() = -get_W() = 16 which is also the stride
    assert( (H-P_H) % stride_h  == 0);
    int out_h = ( (H-P_H) / stride_h ) + 1;

    assert( (W-P_W) % stride_w == 0);
    int out_w = ( (W-P_W) / stride_w ) + 1;

    // std::cout << "Total number of patches:" << out_h * out_w << "\nout_h" << out_h << " | out_w" << out_w <<std::endl;

    u_int d_offset = 0;
    u_int h_offset = 0;
    for(u_int b = 0; b < B; ++b){
        //linearize the patches and their channels
        for(u_int patch_h = 0; patch_h < out_h; ++patch_h){
            for(u_int patch_w = 0;patch_w < out_w; ++patch_w){

                for(u_int c = 0; c < C; c++){
                    for(u_int h = 0; h < P_H; h++){
                        //std::copy(src + 3, src + 3 + 4, dest);
                        h_offset = h * W + (patch_h * stride_h * W) + (patch_w * stride_w) + (c * W * H + b * C * H * W);
                        std::copy(data + h_offset, data + h_offset + P_W, linearized_data + d_offset);
                        d_offset += P_W; 
                    
                    }
                }
            }
        }
    }}