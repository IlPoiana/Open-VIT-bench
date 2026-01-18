#pragma once
#include "./gpu_datatypes.h"
// #include "./cuda_utils.h"
#define MLP_DATA_TYPE CUDA_R_16F// Half
#define MLP_COMPUTE_DATA_TYPE CUBLAS_COMPUTE_32F_PEDANTIC //to avoid using tensor cores
// #define MLP_COMPUTE_DATA_TYPE CUBLAS_COMPUTE_16F_PEDANTIC //to avoid using tensor cores
#define MLP_WORKSPACE_SIZE WORKSPACE_SIZE //4194304 //4MB
#ifndef MLP_BLOCK_DIM
#define MLP_BLOCK_DIM 256
#endif

#define SQRT_2_PI_fp16 hsqrt(__float2half(M_2_PIf32))
#include <iostream>

struct cublasLt_matmul_desc {
    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatrixLayout_t xDesc; 
    cublasLtMatrixLayout_t fcDesc;
    cublasLtMatrixLayout_t cDesc; 
    cublasLtMatrixLayout_t yDesc; 
    float alpha;
    float beta;

    cublasLt_matmul_desc();

    void destroy_descriptors();
};

struct mlp_dimensions {
    u_int B;
    u_int T;
    u_int C;
    u_int K;
    u_int M;

    mlp_dimensions(u_int _B, u_int _T,u_int _C,u_int _K,u_int _M);
};

void create_mlp_descriptors(
    cublasLtHandle_t &handle,
    cublasLt_matmul_desc * matmul, void * d_workspace, cublasLtMatmulAlgo_t * algo,
    mlp_dimensions dimensions,
    bool fused = true
);

//Create the descriptors for cublasLt matmul op
void create_cublasLt_linlay_desc(
    u_int B, u_int T, u_int C, u_int K,
    cublasLt_matmul_desc & matmul
);

//Returns the algorithm used in the cublasLt matmul
cublasLtMatmulAlgo_t fetch_matmul_algos(cublasLtHandle_t &handle,cublasLt_matmul_desc &matmul, void ** d_workspace,  bool initialize_workspace = true);


void strided_linear_layer(
    cublasLtHandle_t & handle, cudaStream_t & stream,
    u_int B, u_int T, u_int K, u_int stride_val,
    cublasLt_matmul_desc &matmul,cublasLtMatmulAlgo_t &algo,void * d_workspace,
    void * d_x, void * d_fc, void * d_b, 
    void * d_y, bool gelu
);


void gpu_mlp(
    cublasLtHandle_t & handle, cudaStream_t & stream,
    u_int B, u_int T, u_int C, u_int K,u_int M,
    void * d_x, void * d_fc1, void * d_h,void * d_b1, void * d_fc2, void * d_b2, 
    void * d_y
);

void gpu_mlp(
    cublasLtHandle_t & handle, cudaStream_t & stream,
    u_int B, u_int T, u_int K,u_int M,
    cublasLt_matmul_desc * matmul,cublasLtMatmulAlgo_t * algo,void * d_workspace,
    void * d_x, void * d_fc1, void * d_h,void * d_b1, void * d_fc2, void * d_b2, 
    void * d_y, int stride_val = 2
);

void fused_gpu_mlp(
    cublasLtHandle_t & handle, cudaStream_t & stream,
    u_int B, u_int T, u_int C, u_int K,u_int M,
    void * d_x, void * d_fc1, void * d_h,void * d_b1, void * d_fc2, void * d_b2, 
    void * d_y
);

void fused_gpu_mlp(
    cublasLtHandle_t & handle, cudaStream_t & stream,
    cublasLt_matmul_desc * matmul,cublasLtMatmulAlgo_t * algo,void * d_workspace,
    void * d_x, void * d_fc1, void * d_h,void * d_b1, void * d_fc2, void * d_b2, 
    void * d_y
);

void bias_matrix(half * h_b, half * h_b_mtx, u_int row, u_int col);

/*
FUNCTIONS USED IN DEV PHASE
*/

void GEMM(
    cublasLtHandle_t &handle, cudaStream_t & stream,
    u_int B, u_int T, u_int C, u_int K,
    void * d_x, void * d_fc, void * d_b, 
    void * y
);

__global__ void bias_GELU(half * d_x, half * d_bias, u_int bias_length, u_int N);

void linear_layer(
    cublasLtHandle_t & handle, cudaStream_t & stream,
    u_int B, u_int T, u_int C, u_int K,
    void * d_x, void * d_fc, void * d_b, 
    void * d_y, bool gelu = true
);

void cuBLAS_test(cublasLtHandle_t & handle, cudaStream_t & stream);
