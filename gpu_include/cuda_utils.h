#pragma once
#include <vector>
#include <string>
#include <assert.h>
#include <cuda_fp16.h> 
#include <iostream>
#include "./helpers.h"
//For generate reference
#include <bits/stdc++.h>
#include <curand_kernel.h>

#define CUBLAS_CHECK(err)                                                         \
do {                                                                              \
    cublasStatus_t err_ = (err);                                                 \
    if (err_ != CUBLAS_STATUS_SUCCESS) {                                          \
        char buf[256];                                                            \
        std::snprintf(buf, sizeof(buf),                                          \
            "cublas error %d at %s:%d", (int)err_, __FILE__, __LINE__);          \
        throw std::runtime_error(buf);                                            \
    }                                                                             \
} while (0)

#define CHECK_LAUNCH() do {                                             \
  CUDA_CHECK(cudaPeekAtLastError()); /* catch launch param errors */    \
  CUDA_CHECK(cudaDeviceSynchronize()); /* catch async runtime errors */ \
} while(0)

#define MAX_STREAMS_CONV2D 32
#define WORKSPACE_SIZE 128 << 20 // 32Mb

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

// Converts n floats at `in` into n halves at `out` (round-to-nearest-even)
inline void f16_to_f32(const half* in, float* out, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        out[i] = __half2float(in[i]);   // host-available intrinsic
    }
}

float result_check_fp16(half * x, half * reference, size_t n);
float result_check_fp16(half * x, float * reference, size_t n);

__global__ void generate_reference(float * d_x, u_int total_n, float scale = 1.0f, u_long seed = 0);
__global__ void generate_reference(half * d_x, u_int total_n, float scale = 1.0f, u_long seed = 0);
void rand_init(float * h_out, u_int n, float rand_scale, u_long seed);

void print_time(benchmark_time time);
void print_json_time(benchmark_time time, const vector<string>& preprocess_names);
inline string yesno(bool b){ return b ? "true" : "false"; };

__global__ void add_strided(half * x, half * val_array, u_int N);

void linearize(float * data, float * linearized_data, picture_shape input_img, conv_kernel_shape kernel);