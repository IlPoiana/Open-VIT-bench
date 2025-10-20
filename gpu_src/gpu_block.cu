#include "../gpu_include/gpu_block.h"

//A method that initialize all the mlp and attention descriptors


//One thread for each element
__global__ void residual_test(half * d_x, half * d_y, u_int N){
    u_int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(g_idx < N){ 
        d_y[i] += d_x[i];
    }
    return;
}


//The transpose of the input data could be fused here
__global__ void residual(half * d_x, half * d_y, u_int N){
    u_int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    u_int start = g_idx * RESIDUAL_ELEM_PER_THREAD;
    u_int end = start + RESIDUAL_ELEM_PER_THREAD;
    if(g_idx < N){ //This shouldn't be necessary 
        #pragma unroll
        for (size_t i = start; i < end; i++)
        {
            d_y[i] += d_x[i];
        }
        
    }
    return;

}
