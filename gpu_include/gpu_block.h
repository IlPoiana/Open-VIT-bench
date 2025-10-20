#include "./cudnn_attention.h"
#include "./gpu_mlp.h"
#include "./gpu_layer.h"
#include "../gpu_src/gpu_block.cu"

#define RESIDUAL_BLOCK_DIM 256
#define RESIDUAL_ELEM_PER_THREAD 4


//One thread for each element
__global__ void residual_test(half * d_x, half * d_y, u_int N);


//The transpose of the input data could be fused here
__global__ void residual(half * d_x, half * d_y, u_int N);