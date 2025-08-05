#include "../gpu_include/cuda_utils.h"

#include <assert.h>
#include <iostream>
#include <algorithm>

conv_kernel_shape::conv_kernel_shape(){
    W=0;
    H=0;
    w_stride=0;
    h_stride=0;
}

__global__ void addScalarKernel(float* array, float val, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        array[idx] += val;
    }
}

benchmark_time::benchmark_time(std::vector<float> pre, float& k): kernel(k){
    preprocess = std::move(pre);
}

void print_time(benchmark_time time){
    for(u_int i = 0; i < time.preprocess.size(); i++){
        printf("preprocess time[%d]: %f\n", i,time.preprocess.at(i));        
    }
    printf("kernel elapsed time: %f\n", time.kernel);
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

    std::cout << "Total number of patches:" << out_h * out_w << "\nout_h" << out_h << " | out_w" << out_w <<std::endl;

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