#pragma once
#include <vector>
#include <string>
#include <assert.h>
#include <cuda_fp16.h> 
#include <iostream>
#include "./helpers.h"

// CUDA API error checking

// #define CUDA_CHECK(err)                                                                            \
// do {                                                                                           \
//     cudaError_t err_ = (err);                                                                  \
//     if (err_ != cudaSuccess) {                                                                 \
//         std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                     \
//         throw std::runtime_error("CUDA error");                                                \
//     }                                                                                          \
// } while (0)

#define CUBLAS_CHECK(err)                                                                          \
do {                                                                                           \
    cublasStatus_t err_ = (err);                                                               \
    if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
        std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
        throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)

#define CHECK_LAUNCH() do {                                             \
  CUDA_CHECK(cudaPeekAtLastError()); /* catch launch param errors */    \
  CUDA_CHECK(cudaDeviceSynchronize()); /* catch async runtime errors */ \
} while(0)

#define WARM_UP 5
#define RUNS_N 10
#define MAX_STREAMS_CONV2D 32

using namespace std;

__device__ __forceinline__ half h_rsqrt(half x) {
    return __float2half( rsqrtf(__half2float(x)) );
}

__device__ __forceinline__ half h_div(half a, half b) {
    return __float2half( __half2float(a) / __half2float(b) );
}

__device__ __forceinline__ half h_mul(half a, half b) {
    return __float2half( __half2float(a) * __half2float(b) );
}
__device__ __forceinline__ half h_sub(half a, half b) {
    return __float2half( __half2float(a) - __half2float(b) );
}
__device__ __forceinline__ half h_add(half a, half b) {
    return __float2half( __half2float(a) + __half2float(b) );
}

__device__ __forceinline__ __half h_tanh(__half x) {
    return __float2half(tanhf(__half2float(x))); 
}



struct picture_shape {
    int B;  // Batch
    int C;  // Channels
    int H;  // Height
    int W;  // Width

    picture_shape(int b, int c, int h, int w) : B(b), C(c), H(h), W(w) {}
};

struct conv_kernel_shape {
    int W;  // Batch
    int H;  // Channels
    int w_stride;  // Height
    int h_stride;  // Width
    int in_channels;
    int out_channels;

    conv_kernel_shape(int w, int h, int s_w, int s_h) : W(w), H(h), w_stride(s_w), h_stride(s_h) {}
    conv_kernel_shape(int w, int h, int s_w, int s_h, int in_ch, int out_ch) : W(w), H(h), w_stride(s_w), h_stride(s_h),
    in_channels(in_ch), out_channels(out_ch) {}
    conv_kernel_shape(int * array_shape);
    conv_kernel_shape();  
};

struct benchmark_time
{
    vector<float> preprocess;
    float kernel;

    benchmark_time(vector<float> pre, float &k);
    benchmark_time();
};


// Converts n floats at `in` into n halves at `out` (round-to-nearest-even)
inline void f32_to_f16(const float* in, __half* out, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        out[i] = __float2half_rn(in[i]);   // host-available intrinsic
    }
}

float result_check_fp16(half * x, half * reference, size_t n);
float result_check_fp16(half * x, float * reference, size_t n);


void print_time(benchmark_time time);
void print_json_time(benchmark_time time, const vector<string>& preprocess_names);

__global__ void addScalarKernel(float* array, float val, int N);

void linearize(float * data, float * linearized_data, picture_shape input_img, conv_kernel_shape kernel);