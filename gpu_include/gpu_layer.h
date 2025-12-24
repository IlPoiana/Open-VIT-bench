#pragma once
#include "./gpu_datatypes.h"
#include <cub/cub.cuh>

#define TOKENS_PER_BLOCK 4
#define EMBEDDINGS_SIZE 768
#define TOKENS_NUM  196 //is 197
#define TOKENS_NUM_VIT 197
#define ELEMENTS_PER_TH 4

#define LAYER_BLOCK_DIM 512
#define SH_MEM_DIM 2*LAYER_BLOCK_DIM //this is the dimension of the shared mem
    
#define CUB_LAYER_BLOCK_DIM EMBEDDINGS_SIZE / 2 // (384 * 2 = 768)a multiple of 768 [384, 192, 96, 48, 24, 12, 6, 3] 
#define CUB_LAYER_MULTI_BLOCK_DIM EMBEDDINGS_SIZE / ELEMENTS_PER_TH

/*
DEVICE KERNELS/FUNCTIONS
*/

template <typename T>
__device__ void type_dev_block_reduction(T * x_sh, u_int arr_size, u_int idx);

__device__ void dev_block_layer_norm(
    u_int C, u_int idx,u_int global_idx,               // N = B*T (flattened), C = channels
    half * x_data, half * out,      //All device pointers
    half * scale, half * bias,
    half epsilon
);

__device__ void cub_dev_block_ln(
    u_int idx,u_int global_idx,               // N = B*T (flattened), C = channels
    half * x_data, half * out,      //All device pointers
    half * scale, half * bias,
    half epsilon
);

/*
HOST KERNELS
*/

__global__ void gpu_layer_norm(
    u_int C, half * x_data, half * out,   //All device pointers   
    half * scale, half * bias,      
    half epsilon
);

__global__ void multi_block_layer_norm(
    u_int C, u_int tokens_n, u_int tokens_block_n,   //All device pointers
    half * x_data,      
    half * scale, half * bias,      
    half epsilon
);

__global__ void cub_layer_norm(
    half * x_data,      //All device pointers
    half * scale, half * bias,      
    half epsilon
);

__global__ void cub_layer_norm(
    half * x_data,      
    half * scale, half * bias,      //All device pointers
    half epsilon, 
    u_int tokens_per_block
);

__global__ void cub_single_layer_norm(
    half * x_data, half * out,     // device pointers
    half * scale, half * bias,      // device pointers
    half epsilon, 
    u_int tokens_per_block
);

__global__ void multi_elem_cub_ln(
    half * x_data,      
    half * scale, half * bias,      //All device pointers
    half epsilon,
    u_int tokens_per_block
);

__global__ void unrolled_multi_elem_cub_ln(
    half * x_data,      
    half * scale, half * bias,      //All device pointers
    half epsilon
);
