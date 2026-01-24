#pragma once
#include "../include/datatypes.h"
#include "cuda_utils.h"
// #include <cuda_fp16.h>

#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cublasLt.h>

// represent a f16 matrix with data pointer and row and col parameters
struct mtx {
    __half * data;
    u_int16_t row_n;
    u_int16_t col_n;

    mtx(float * f32_data, u_int16_t row, u_int16_t col);
    mtx(u_int row, u_int col);
    ~mtx();
};

// B,C,H,W
struct h_tensor {
    __half * data;
    u_int16_t B;
    u_int16_t C;
    u_int16_t H;
    u_int16_t W;

    h_tensor(float * f32_data, u_int16_t batch, u_int16_t channels, u_int16_t height, u_int16_t width);

    ~h_tensor();
};
