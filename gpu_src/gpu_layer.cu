#include "../gpu_include/gpu_layer.h"
    
template <typename T>
__device__ void type_dev_block_reduction(T * x_sh, u_int arr_size, u_int idx){
    for(u_int limit = ((arr_size + 1) >> 1); limit > 0; limit >>= 1) {// n N +1 = 6 => limit = 3
        
        if((arr_size & 1) == 0){ 
            if(idx < limit) {
                x_sh[idx] += x_sh[idx + limit];
                
            }
            arr_size >>= 1; 
            limit += 1; 
        }
        else{ // 0 to 1 
            if(idx < limit - 1){ 
                x_sh[idx] += x_sh[idx + limit];
            }
            arr_size >>= 1; arr_size += 1; 
            limit += 1; 
        }

        __syncthreads();

        if(arr_size < 2)
            break;
                    
    }
}

__device__ void sm50_dev_block_ln(
    u_int C, u_int idx,u_int global_idx,               // N = B*T (flattened), C = channels
    half * x_data, half * out,      // device pointers
    half * scale, half * bias,
    half epsilon
){
    __shared__ half mean;
    __shared__ half variance;
    
    __shared__ half x_sh[SH_MEM_DIM]; // n_elems < block size in this case
    __shared__ half x_buff[SH_MEM_DIM]; // n_elems < block size in this case
    half c = __uint2half_rn(C);
    u_int stride = (C + 1) >> 1;

    if(idx < stride){
        // load 2 elements per thread, to increase threads occupancy!
        x_sh[idx] = x_data[global_idx];
        x_buff[idx] = x_sh[idx];

        if((C & 1) || (idx < C - 1)){ // load it if the array is even or you aren't the last thread
            x_sh[idx + stride] = x_data[global_idx + stride];
            x_buff[idx + stride] = x_sh[idx + stride];
        }
        __syncthreads();
        
        // x_mean[0] will have the value
        // mean --> reduction + scalar division
        type_dev_block_reduction<half>(x_buff, C, idx); 
        if(idx == 0)
            mean = x_buff[0] / c;
            
        // reusing the buffer in sh mem
        x_buff[idx] = x_sh[idx]; 
        if((C & 1) || (idx < C - 1)){ 
            x_buff[idx + stride] = x_sh[idx + stride];
        }
        __syncthreads();
            
            
        // variance --> scalar per elem ops + reduction + scalar division
        x_buff[idx] = (x_buff[idx] - mean) * (x_buff[idx] - mean);
        if((C & 1) || (idx < C - 1)){ 
            x_buff[idx + stride] = (x_buff[idx + stride] - mean) * (x_buff[idx + stride] - mean);
        }
        __syncthreads();
        type_dev_block_reduction<half>(x_buff,C,idx);
        if(idx == 0)
            variance = x_buff[0] / c;
        __syncthreads();
        
        // Normalize --> per elem ops
        out[global_idx] = (((x_sh[idx] - mean) * hrsqrt( variance + epsilon)) * scale[idx]) + bias[idx] ;        
        if((C & 1) || (idx < C - 1)){ 
            out[global_idx + stride] = (((x_sh[idx + stride] - mean) * hrsqrt( variance + epsilon)) * scale[idx + stride]) + bias[idx+ stride] ;
        }
        
    }
} 

// 1) Two elements per thread, hand written fused layer norm
__device__ void dev_block_layer_norm(
    u_int C, u_int idx,u_int global_idx,               // N = B*T (flattened), C = channels
    half * x_data, half * out,      // device pointers
    half * scale, half * bias,
    half epsilon
){
    __shared__ half mean;
    __shared__ half variance;
    
    half x_sh[2];
    __shared__ half x_buff[SH_MEM_DIM]; // n_elems < block size in this case
    half c = __uint2half_rn(C);
    u_int stride = (C + 1) >> 1;

    if(idx < stride){
        // load 2 elements per thread, to increase threads occupancy!
        x_sh[0] = x_data[global_idx];
        x_buff[idx] = x_sh[0];

        if((C & 1) || (idx < C - 1)){ // load it if the array is even or you aren't the last thread
            x_sh[1] = x_data[global_idx + stride];
            x_buff[idx + stride] = x_sh[1];
        }
    }
    __syncthreads();
    type_dev_block_reduction<half>(x_buff, C, idx); 
    if(idx < stride){
        // mean --> reduction + scalar division
        
        if(idx == 0)
            mean = x_buff[0]/ c;

        // reusing the buffer in sh mem
        x_buff[idx] = x_sh[0]; 

        if((C & 1) || (idx < C - 1)){ 
            x_buff[idx + stride] = x_sh[1];
        }
    }

    __syncthreads();
            
        
    // variance --> scalar per elem ops + reduction + scalar division        
    if(idx < stride){
        x_buff[idx] = (x_buff[idx] - mean) * (x_buff[idx] - mean);
        if((C & 1) || (idx < C - 1)){ 
            x_buff[idx + stride] = (x_buff[idx + stride] - mean) * (x_buff[idx + stride] - mean);
        }
        
    }
    __syncthreads();
    type_dev_block_reduction<half>(x_buff,C,idx);
    
    if(idx < stride){
        if(idx == 0)
            variance = x_buff[0] / c;
    }

    __syncthreads();
        
    // Normalize --> per elem ops
    if(idx < stride){
        out[global_idx] = (((x_sh[0] - mean) * hrsqrt( variance + epsilon)) * scale[idx]) + bias[idx] ;        

        if((C & 1) || (idx < C - 1)){ 
            out[global_idx + stride] = (((x_sh[1] - mean) * hrsqrt( variance + epsilon)) * scale[idx + stride]) + bias[idx+ stride] ;
        }
    }
        
    
} 

/*
Enhanced version which uses CUB reduction for better performances
- Elements per thread and threads number fixed(2 elements per thread, 384 threads per block) for encoder block LN 
- Using shared memory from the main loop to fetch only once the bias and scale variables(multi token per block)
Requirements:
- CUB_LAYER_BLOCK_DIM have to be EMBEDDINGS_SIZE / 2 (also embeddings size have to be even)
*/
__device__ void cub_dev_block_ln(
    u_int idx,u_int global_idx,               // N = B*T (flattened), C = channels
    half * x_data, half * out,      // device pointers
    half * scale, half * bias,
    half epsilon, 
    cub::BlockReduce<half, CUB_LAYER_BLOCK_DIM> &BlockReduce
    
){
    // using BlockReduce = cub::BlockReduce<half, CUB_LAYER_BLOCK_DIM>;
    // __shared__ typename BlockReduce::TempStorage cub_shared_storage;
    half th_data[2];
    half x_buff[2]; // used to stored the fetched data
    half th_aggregate; // the result is shared across all the threads

    __shared__ half mean;
    __shared__ half variance;
    
    half c = __int2half_rn(EMBEDDINGS_SIZE);
    u_int stride = CUB_LAYER_BLOCK_DIM;

    if(idx < stride){
        // load 2 elements per thread, to increase threads occupancy!
        th_data[0] = x_data[global_idx];
        x_buff[0] = th_data[0];
        
        th_data[1] = x_data[global_idx + stride];
        x_buff[1] = th_data[1];

        th_aggregate = BlockReduce.Sum(th_data);

    }
    if(idx == 0)
        mean = th_aggregate / c;
    __syncthreads(); // suggested by the doc
    
    if(idx < stride){
        
        th_data[0] = (x_buff[0] - mean) * (x_buff[0] - mean);
        th_data[1] = (x_buff[1] - mean) * (x_buff[1] - mean);
        th_aggregate = BlockReduce.Sum(th_data);
        
    }
    if(idx == 0)
        variance = th_aggregate / c;

    __syncthreads();

    if(idx < stride){
    
        out[global_idx] = (((x_buff[0] - mean) * hrsqrt( variance + epsilon)) * scale[0]) + bias[0] ;        
        out[global_idx + stride] = (((x_buff[1] - mean) * hrsqrt( variance + epsilon)) * scale[1]) + bias[1] ;
        
    }  
}

//One element per thread, supposing that the launched kernel has blockDim equals to the channel size
__device__ void cub_dev_block_ln(
    u_int idx,u_int global_idx,               // N = B*T (flattened), C = channels
    half * x_data, half * out,      // device pointers
    float scale, float bias,
    float &mean, float &variance,
    float epsilon, 
    cub::BlockReduce<float, EMBEDDINGS_SIZE> &BlockReduce
    
){
    float th_data, x_buff;
    float th_aggregate; // the result is shared across all the threads
    
    int c = EMBEDDINGS_SIZE;

    th_data = __half2float(x_data[global_idx]);
    x_buff = th_data;

    th_aggregate = BlockReduce.Sum(th_data, EMBEDDINGS_SIZE);

    if(idx == 0)
        mean = th_aggregate / c;
    __syncthreads(); // suggested by the doc
    
    
    th_data = (x_buff - mean) * (x_buff - mean);
    th_aggregate = BlockReduce.Sum(th_data, EMBEDDINGS_SIZE);
        
    if(idx == 0)
        variance = th_aggregate / c;

    __syncthreads();

    
    out[global_idx] = __float2half( (((x_buff - mean) * rsqrtf( variance + epsilon)) * scale) + bias);        
        

}

//Supposing that each thread has the same number of elements to compute!
__device__ void dev_multi_elem_cub_ln(
    u_int idx,u_int global_idx,               // N = B*T (flattened), C = channels
    half * x_data, half * out,      // device pointers
    float * scale, float * bias,
    float epsilon, 
    cub::BlockReduce<float, CUB_LAYER_MULTI_BLOCK_DIM> &BlockReduce
){
    // using BlockReduce = cub::BlockReduce<half, CUB_LAYER_BLOCK_DIM>;
    // __shared__ typename BlockReduce::TempStorage cub_shared_storage;
    float th_data[ELEMENTS_PER_TH];
    float x_buff[ELEMENTS_PER_TH]; // used to stored the fetched data
    float th_aggregate; // the result is shared across all the threads

    __shared__ float mean;
    __shared__ float variance;
    
    float c = EMBEDDINGS_SIZE;
    u_int stride = CUB_LAYER_MULTI_BLOCK_DIM; 
    // u_int offset = global_idx + (idx * (ELEMENTS_PER_TH - 1)); 
    u_int offset;
    if(idx < stride){
    
        for(u_int i = 0; i< ELEMENTS_PER_TH; i++){
            offset = global_idx + stride * i;
            th_data[i] = x_data[offset];
            x_buff[i] = th_data[i];
        }

        th_aggregate = BlockReduce.Sum(th_data);

    }
    if(idx == 0)
        mean = th_aggregate / c;
    __syncthreads(); // suggested by the doc
    
    if(idx < stride){
        for(u_int i = 0; i< ELEMENTS_PER_TH; i++){
            th_data[i] = (x_buff[i] - mean) * (x_buff[i] - mean);
        }
        th_aggregate = BlockReduce.Sum(th_data);
        
    }
    if(idx == 0)
        variance = th_aggregate / c;

    __syncthreads();

    if(idx < stride){
        for(u_int i = 0; i< ELEMENTS_PER_TH; i++){
            offset = global_idx + stride * i;
            out[offset] = (((x_buff[i] - mean) * rsqrt( variance + epsilon)) * scale[i]) + bias[i];        
        }
    }  
    return;
}


//Supposing that each thread has the same number of elements to compute!
__device__ void dev_unrolled_multi_elem_cub_ln(
    u_int idx,u_int global_idx,               // N = B*T (flattened), C = channels
    half * x_data, half * out,      // device pointers
    float * scale, float * bias,
    float epsilon, 
    cub::BlockReduce<float, CUB_LAYER_MULTI_BLOCK_DIM> &BlockReduce
){

    float th_data[ELEMENTS_PER_TH];
    float x_buff[ELEMENTS_PER_TH]; // used to stored the fetched data
    float th_aggregate; // the result is shared across all the threads

    __shared__ float mean;
    __shared__ float variance;
    
    float c = EMBEDDINGS_SIZE;
       

    #pragma unroll
    for(u_int i = 0; i< ELEMENTS_PER_TH; i++){
        th_data[i] = x_data[global_idx + blockDim.x * i];
        x_buff[i] = th_data[i];
    }

    th_aggregate = BlockReduce.Sum(th_data);

    if(idx == 0)
        mean = th_aggregate / c;
    __syncthreads(); // suggested by the doc
    
    #pragma unroll
    for(u_int i = 0; i< ELEMENTS_PER_TH; i++){
        th_data[i] = (x_buff[i] - mean) * (x_buff[i] - mean);
    }
    th_aggregate = BlockReduce.Sum(th_data);
        
    if(idx == 0)
        variance = th_aggregate / c;

    __syncthreads();

    #pragma unroll
    for(u_int i = 0; i< ELEMENTS_PER_TH; i++){
        out[global_idx + blockDim.x * i] = (((x_buff[i] - mean) * rsqrt( variance + epsilon)) * scale[i]) + bias[i] ;        
    }  
    return;
}


/*
One block for token. Features of this approach:
1. Fused Kernel (mean + variance + norm computation)
2. Sh mem usage
3. Thread work more balanced, two elements for thread
IS SUPPOSED THAT C FITS IN ONE BLOCK! (SH_MEM size actually)
*/
__global__ void gpu_layer_norm(
    u_int C,half * x_data, half * out,      // device pointers
    half * scale, half * bias,      // device pointers
    half epsilon
){
    u_int local_idx = threadIdx.x;
    u_int global_idx = blockIdx.x * C + local_idx;
    
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 500) 
        sm50_dev_block_ln( C, local_idx, global_idx,x_data, out, scale, bias, epsilon);

    #else 
        dev_block_layer_norm( C, local_idx, global_idx,x_data, out, scale, bias, epsilon);
    #endif
    return;
}


//More than one token per block to compute(no sh. mem used) DEPRECATED
__global__ void multi_block_layer_norm(
    u_int C, u_int tokens_n,u_int tokens_block_n, 
    half * x_data,      // device pointers
    half * scale, half * bias,      // device pointers
    half epsilon
){
    u_int local_idx = threadIdx.x;
    u_int global_idx = blockIdx.x * C * tokens_block_n + local_idx;

    //Compute the stride between tokens
    // #pragma unroll // This when the token embeddings and the token number is fixed will help a lot
    u_int loop_idx = 0;
    for(u_int token = 0; (token < tokens_block_n); ++token){
        if((blockIdx.x * tokens_block_n + token) < tokens_n){ // This is necessary to avoid some threads finishing in deadlock
            loop_idx = global_idx + C * token;
            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 500) 
                sm50_dev_block_ln( C, local_idx, loop_idx,x_data, x_data, scale, bias, epsilon);
            #else 
                dev_block_layer_norm( C, local_idx,  loop_idx,x_data, x_data, scale, bias, epsilon);
            #endif
        }
    }

}

/**
 * @brief Compute layer norm on x_data, using cub for reduction, 2 elements per thread and multiple tokens per block using sh. mem for scale and bias
 * 
 * @param x_data 
 * @param scale 
 * @param bias 
 * @param epsilon 
 * @return __global__ 
 */
__global__ void cub_layer_norm(
    half * x_data,      // device pointers
    half * scale, half * bias,      // device pointers
    half epsilon
){
    u_int local_idx = threadIdx.x;
    u_int global_idx = blockIdx.x * EMBEDDINGS_SIZE * TOKENS_PER_BLOCK + local_idx;
    
    half th_bias[2], th_scale[2];
    
    th_bias[0] = bias[local_idx];   th_bias[1] = bias[local_idx + blockDim.x];
    th_scale[0] = scale[local_idx]; th_scale[1] = scale[local_idx + blockDim.x];

    using BlockReduce = cub::BlockReduce<half, CUB_LAYER_BLOCK_DIM>;
    __shared__ BlockReduce::TempStorage cub_shared_storage;
    BlockReduce block_reduce(cub_shared_storage);
    //Compute the stride between tokens
    // #pragma unroll // This when the token embeddings and the token number is fixed will help a lot
    u_int loop_idx = 0;
    for(u_int token = 0; (token < TOKENS_PER_BLOCK); ++token){
        loop_idx = global_idx + EMBEDDINGS_SIZE * token;        
        cub_dev_block_ln(
            local_idx, loop_idx,
            x_data, x_data,
            th_scale, th_bias,
            epsilon,
            block_reduce
        );
    }
    

}

__global__ void cub_layer_norm(
    half * x_data, half * y,     // device pointers
    half * scale, half * bias,      // device pointers
    half epsilon, 
    u_int tokens_per_block
){
    u_int local_idx = threadIdx.x;
    u_int global_idx = blockIdx.x * EMBEDDINGS_SIZE * tokens_per_block + local_idx;
    
    half th_bias[2], th_scale[2];
    
    th_bias[0] = bias[local_idx];   th_bias[1] = bias[local_idx + blockDim.x];
    th_scale[0] = scale[local_idx]; th_scale[1] = scale[local_idx + blockDim.x];

    using BlockReduce = cub::BlockReduce<half, CUB_LAYER_BLOCK_DIM>;
    __shared__  __align__(16) BlockReduce::TempStorage cub_shared_storage;
    BlockReduce block_reduce(cub_shared_storage);
    //Compute the stride between tokens
    // #pragma unroll // This when the token embeddings and the token number is fixed will help a lot
    u_int loop_idx = 0;
    for(u_int token = 0; (token < tokens_per_block); ++token){
        loop_idx = global_idx + EMBEDDINGS_SIZE * token;        
        cub_dev_block_ln(
            local_idx, loop_idx,
            x_data, y,
            th_scale, th_bias,
            epsilon,
            block_reduce
        );
    }
    

}

//Single element per thread for CUB, its supposed that the number of blocks launched is equal to the number of elements to compute
__global__ void cub_single_layer_norm(
    half * x_data, half * out,     // device pointers
    half * scale, half * bias,      // device pointers
    float epsilon, 
    u_int tokens_per_block
){
    u_int local_idx = threadIdx.x;
    u_int global_idx = blockIdx.x * blockDim.x * tokens_per_block + local_idx;
    
    float th_bias, th_scale;
    __shared__ float mean;
    __shared__ float variance;

    th_bias  = __half2float(bias[local_idx]);   
    th_scale = __half2float(scale[local_idx]); 

    using BlockReduce = cub::BlockReduce<float, EMBEDDINGS_SIZE>;
    __shared__ __align__(16)BlockReduce::TempStorage cub_shared_storage;
    BlockReduce block_reduce(cub_shared_storage);
    //Compute the stride between tokens
    // #pragma unroll // This when the token embeddings and the token number is fixed will help a lot
    u_int loop_idx = 0;
    for(u_int token = 0; (token < tokens_per_block); ++token){
        loop_idx = global_idx + blockDim.x * token;        
        cub_dev_block_ln(
            local_idx, loop_idx,
            x_data, out,
            th_scale, th_bias,
            mean, variance,
            epsilon,
            block_reduce
        );
        __syncthreads();
    }
}

// Layer Norm, using CUB for reduction, M elements to compute per token per thread and N tokens to compute per block.
__global__ void multi_elem_cub_ln(
    half * x_data, half * y,     // device pointers
    half * scale, half * bias,      // device pointers
    float epsilon, 
    u_int tokens_per_block
){
    u_int local_idx = threadIdx.x;
    u_int global_idx = blockIdx.x * EMBEDDINGS_SIZE * tokens_per_block + local_idx;
    
    float th_bias[ELEMENTS_PER_TH], th_scale[ELEMENTS_PER_TH];
    
    for(u_int i = 0; i < ELEMENTS_PER_TH; i++){
        th_bias[i] = bias[local_idx + i * blockDim.x];         
        th_scale[i] = scale[local_idx+ i * blockDim.x]; 
    }

    using BlockReduce = cub::BlockReduce<float, CUB_LAYER_MULTI_BLOCK_DIM>;
    __shared__ __align__(16) BlockReduce::TempStorage cub_shared_storage;
    BlockReduce block_reduce(cub_shared_storage);
    //Compute the stride between tokens
    u_int loop_idx = 0;
    for(u_int token = 0; (token < tokens_per_block); ++token){
        loop_idx = global_idx + EMBEDDINGS_SIZE * token;        
        dev_multi_elem_cub_ln(
            local_idx, loop_idx,
            x_data, y,
            th_scale, th_bias,
            epsilon,
            block_reduce
        );
    }
    

}

// Unrolled version of multiple element CUB layer norm
__global__ void unrolled_multi_elem_cub_ln(
    half * x_data, half * y,     // device pointers
    half * scale, half * bias,      // device pointers
    float epsilon
){
    u_int local_idx = threadIdx.x;
    u_int global_idx = blockIdx.x * EMBEDDINGS_SIZE * TOKENS_PER_BLOCK + local_idx;
    
    float th_bias[ELEMENTS_PER_TH], th_scale[ELEMENTS_PER_TH];
    
    #pragma unroll
    for(u_int i = 0; i < ELEMENTS_PER_TH; i++){
        th_bias[i] = bias[local_idx + i * CUB_LAYER_MULTI_BLOCK_DIM];         
        th_scale[i] = scale[local_idx+ i * CUB_LAYER_MULTI_BLOCK_DIM]; 
    }

    using BlockReduce = cub::BlockReduce<float, CUB_LAYER_MULTI_BLOCK_DIM>;
    __shared__ __align__(16) BlockReduce::TempStorage cub_shared_storage;
    BlockReduce block_reduce(cub_shared_storage);
    
    //Compute the stride between tokens
    u_int loop_idx = 0;
    #pragma unroll // This can be done when the token embeddings and the token number is fixed
    for(u_int token = 0; (token < TOKENS_PER_BLOCK); ++token){
        loop_idx = global_idx + EMBEDDINGS_SIZE * token;
        dev_unrolled_multi_elem_cub_ln(
            local_idx, loop_idx,
            x_data, y,
            th_scale, th_bias,
            epsilon,
            block_reduce
        );
    }
    

}


