// #include "../include/conv2d.h"
// #include "../include/datatypes.h"
#include "../gpu_include/gpu_conv2d.h"
// #include "../gpu_include/gpu_datatypes.h"

#include <utility>
#include <assert.h>
#include <iostream>

#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

GPUConv2d:: GPUConv2d(
    vit_size _in_channels,
    vit_size _out_channels,
    conv_kernel_shape kernel_shape,
    vit_bool _use_bias
): Conv2d(
        _in_channels, _out_channels, kernel_shape.H,
        kernel_shape.W, kernel_shape.h_stride, kernel_shape.w_stride, _use_bias
    ) 
{   
    std::cout << "Initialized the GPUConv2d class" << std::endl;
    std::cout << "in_channels: "  << get_in_channels()
           << ", out_channels: " << get_out_channels()
           << ", kernel: h"       << get_kernel_h() << "x" << get_kernel_w()
           << ", stride: "       << get_stride_h() << "x" << get_stride_w()
           << ", use_bias: "     << get_use_bias() << std::endl;
}

GPUConv2d:: GPUConv2d(
    cublasHandle_t& in_cublasH, 
    cudaStream_t& in_stream,
    vit_size _in_channels,
    vit_size _out_channels,
    conv_kernel_shape kernel_shape,
    vit_bool _use_bias
): Conv2d(
        _in_channels, _out_channels, kernel_shape.H,
        kernel_shape.W, kernel_shape.h_stride, kernel_shape.w_stride, _use_bias
    )  
{   
    set_stream(in_stream);
    set_handle(in_cublasH);
    bind_handle();
    std::cout << "Initialized the GPUConv2d class" << std::endl;
    std::cout << "in_channels: "  << get_in_channels()
           << ", out_channels: " << get_out_channels()
           << ", kernel: h"       << get_kernel_h() << "x" << get_kernel_w()
           << ", stride: "       << get_stride_h() << "x" << get_stride_w()
           << ", use_bias: "     << get_use_bias() << std::endl;
}

//To do
GPUConv2d:: ~GPUConv2d() {};


void GPUConv2d:: set_d_kernel(vit_float * device_idx){
    d_kernel = device_idx;
}


//Allocate into the GPU the passed kernel
void GPUConv2d::move_kernel(PictureBatch& _kernel){    
    gpu_kernel = std::move(_kernel);
    u_int n_elements= gpu_kernel.get_B() * gpu_kernel.get_C() * gpu_kernel.get_H() * gpu_kernel.get_W();
    std::cout << "kernel dimension" << n_elements << std::endl;
    CUDA_CHECK(cudaMalloc(&d_kernel, sizeof(vit_float) * n_elements));
    CUDA_CHECK(cudaMemcpyAsync(d_kernel, gpu_kernel.get_data(), sizeof(vit_float) * n_elements, cudaMemcpyHostToDevice,
                               stream));
}
    
    
    //Allocate into the GPU the passed bias
void GPUConv2d::move_bias(RowVector& _bias){
    gpu_bias = std::move(_bias);
    u_int n_elements= gpu_bias.get_DIM();
    std::cout << "bias dimension" << n_elements << std::endl;
    CUDA_CHECK(cudaMalloc(&d_bias, sizeof(vit_float) * n_elements));
    CUDA_CHECK(cudaMemcpyAsync(d_bias, gpu_bias.get_data(), sizeof(vit_float) * n_elements, cudaMemcpyHostToDevice,
                               stream)); 
}

/*
set_stream -> create_handle or set_handle() -> bind_handle()
*/
void GPUConv2d::set_stream(cudaStream_t in_stream){
    stream = in_stream;
}

void GPUConv2d::create_handle(){
    CUBLAS_CHECK(cublasCreate(&cublasH));
}


void GPUConv2d::set_handle(cublasHandle_t in_cublasH){
    cublasH = in_cublasH;
}

void GPUConv2d::bind_handle(){
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));
}




//Deallocate the kernel from GPU
void GPUConv2d::kernel_free(){
    CUDA_CHECK(cudaFree(d_kernel));
}

//Deallocate the bias from GPU
void GPUConv2d::bias_free(){
    CUDA_CHECK(cudaFree(d_bias));
}

//free all device resources
void GPUConv2d::cuda_free(){
    kernel_free();
    if(get_use_bias())
        bias_free();
    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));
}

/*
ASSUMING THE INPUT DATA IS ALREADY IN GPU


MODULAR APPROACH:
1. Single channel (one mem copy for each channel of each input...)
2. Single output channel (one mem copy for each output channel...)
3. Patch and Batch are the general case, copy only once at the end
4. No copies just processing

*/


//TO DO
// void GPUConv2d::forward(const PictureBatch& x_in, PictureBatch& x_out) const
// {
// }

// //forward for the actual computation no mem copy to the host)
// void GPUConv2d::forward(GPUPictureBatch x_in, GPUPictureBatch x_out){
// }
//////////////


void GPUConv2d:: test_forward(GPUPictureBatch& x_in, PictureBatch& x_out, u_int level){
    switch (level){
        case 0:
            single_forward(x_in, x_out);
            break;
        case 1:
            unified_forward(x_in, x_out);
            break;
        default:
            memory_forward(x_in, x_out);
            break;
    }
} 

void GPUConv2d:: timed_forward(GPUPictureBatch& x_in, PictureBatch& x_out, u_int level, benchmark_time& time){
    cudaEvent_t start;cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    switch (level){
        case 0:
            CUDA_CHECK(cudaEventRecord(start));
            single_forward(x_in, x_out);
            CUDA_CHECK(cudaEventRecord(stop));
            break;
        case 1:
            CUDA_CHECK(cudaEventRecord(start));
            unified_forward(x_in, x_out);
            CUDA_CHECK(cudaEventRecord(stop));
            break;
        default:
            memory_forward(x_in, x_out);
            break;
    }
    
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time.kernel, start, stop);
    
}


// Single_channel kernel comparison, preserves the cpp implementation memory and sequential implementation
void GPUConv2d::single_forward(GPUPictureBatch& x_in, PictureBatch& x_out) {
    //take x_in * kernel reduced and put it in y, then return y and free memory

    vit_size in_channels = get_in_channels(); //2
    vit_size out_channels = get_out_channels(); //1

    vit_size B = x_in.get_B();
    vit_size C = x_in.get_C();
    vit_size H = x_in.get_H();
    vit_size W = x_in.get_W();
    vit_size P_H = gpu_kernel.get_H();
    vit_size P_W = gpu_kernel.get_W();
    vit_size stride_h = get_stride_h();
    vit_size stride_w = get_stride_w();

    printf("executing the GPUConv2d single channel forward\n");
    
    assert(C == in_channels);
    assert(gpu_kernel.get_C() == out_channels);
    assert(gpu_kernel.get_B() == in_channels);
    if (get_use_bias()) {
        assert(gpu_bias.get_DIM() == out_channels);
    }

    //kernel.get_H() = -get_W() = 16 which is also the stride
    assert( (H-P_H) % stride_h  == 0);
    vit_size out_h = ( (H-P_H) / stride_h ) + 1;

    assert( (W-P_W) % stride_w == 0);
    vit_size out_w = ( (W-P_W) / stride_w ) + 1;

    vit_float val = 0.0;
    vit_size x_stride = 0;
    vit_size k_stride = 0;
    
    PictureBatch y(B, out_channels, out_h, out_w);
    //accumulator vector for intermediate results
    std::vector<vit_float> result;
    
    for (int batch=0;batch<B;++batch) {
        for (int y_c=0;y_c<out_channels;++y_c) {
            // Iterate over all the patches (14x14)
            for (int y_h=0;y_h<out_h;++y_h) {
                for (int y_w=0;y_w<out_w;++y_w) {
                    val = get_use_bias() ? gpu_bias.at(y_c) : 0;
                    

                    // Iterate over all the values of each patch (C=3 x 16 x16)
                    //one mat mul for every input channel and for every output channel (3*768)
                    for(int k_c=0;k_c<C;++k_c) {
                        for(int k_h=0;k_h<P_H;++k_h) {
                            //compute the dot product between each row
                            result.push_back(0.0);
                            //shift the pointers on the gpu to compute
                            x_stride = k_h * W + 
                                stride_w*y_w + W * stride_h *y_h +
                                (k_c*H*W) +
                                (batch*C*H*W);
                            k_stride = k_h * P_W +(k_c*P_H*P_W) + (y_c*C*P_H*P_W);
                            

                            //execute the convolution
                            CUBLAS_CHECK(
                                cublasSdot(
                                    cublasH,
                                    P_W, x_in.get_d_data() + x_stride, 1,
                                    d_kernel + k_stride, 1,
                                    &result[k_c * P_H+ k_h]
                                )
                            );
                
                            
                        }
                    }

                    cudaDeviceSynchronize();
                    //or use cublas for reduce the array
                    for(int res_idx=0;res_idx<result.size();++res_idx) {
                        val += result[res_idx];
                    }
                    
                    // std::cout << "val: " << val << std::endl;
                    y.set(batch, y_c, y_h, y_w, val);
                    result.clear();//zeroing the result vector
                }
            }

        }
    }
    
    x_out = std::move(y);
}


// Sequential forward but with unified kernel computation
void GPUConv2d::unified_forward(GPUPictureBatch& x_in, PictureBatch& x_out) {

    vit_size in_channels = get_in_channels(); //2
    vit_size out_channels = get_out_channels(); //1

    vit_size B = x_in.get_B();
    vit_size C = x_in.get_C();
    vit_size H = x_in.get_H();
    vit_size W = x_in.get_W();
    vit_size P_H = gpu_kernel.get_H();
    vit_size P_W = gpu_kernel.get_W();
    vit_size stride_h = get_stride_h();
    vit_size stride_w = get_stride_w();

    printf("executing the GPUConv2d unified forward\n");
    // x_in.print();
    assert(C == in_channels);
    assert(gpu_kernel.get_C() == out_channels);
    assert(gpu_kernel.get_B() == in_channels);
    if (get_use_bias()) {
        assert(gpu_bias.get_DIM() == out_channels);
    }

    //kernel.get_H() = -get_W() = 16 which is also the stride
    assert( (H-P_H) % stride_h  == 0);
    vit_size out_h = ( (H-P_H) / stride_h ) + 1;

    assert( (W-P_W) % stride_w == 0);
    vit_size out_w = ( (W-P_W) / stride_w ) + 1;

    vit_float val = 0.0;
    vit_size x_stride = 0;
    vit_size k_stride = 0;
    
    vit_float result = 0.0;
    PictureBatch y(B, out_channels, out_h, out_w);

    for (int batch=0;batch<B;++batch) {
        for (int y_c=0;y_c<out_channels;++y_c) {
            // Iterate over all the patches (14x14)
            for (int y_h=0;y_h<out_h;++y_h) {
                for (int y_w=0;y_w<out_w;++y_w) {
                    val = get_use_bias() ? gpu_bias.at(y_c) : 0;
                    //compute the convolution of the patch for the channel
                    //y_w * P_H * P_W * C 
                    x_stride = y_w * P_H * P_W * C + y_h * P_H * P_W * C * out_w + //patches
                        batch * P_H * P_W * C * out_h * out_w;
                    k_stride = P_H * P_W * C * y_c;
                    CUBLAS_CHECK(
                        cublasSdot(
                            cublasH,
                            P_H * P_W * C,
                            x_in.get_d_data() + x_stride,1,
                            d_kernel + k_stride,1,
                            &result
                        )
                    );
                    result += val;
                    y.set(batch, y_c, y_h, y_w, result);
                }
            }
        }
    }
    
    x_out = std::move(y);
}

// fully parallelize forward that copies the result only once
void GPUConv2d::memory_forward(GPUPictureBatch& x_in, PictureBatch& x_out)  {
    
    vit_size in_channels = get_in_channels(); //2
    vit_size out_channels = get_out_channels(); //1

    vit_size B = x_in.get_B();
    vit_size C = x_in.get_C();
    vit_size H = x_in.get_H();
    vit_size W = x_in.get_W();
    vit_size P_H = gpu_kernel.get_H();
    vit_size P_W = gpu_kernel.get_W();
    vit_size stride_h = get_stride_h();
    vit_size stride_w = get_stride_w();

    printf("executing the GPUConv2d memory forward\n");
    // x_in.print();
    assert(C == in_channels);
    assert(gpu_kernel.get_C() == out_channels);
    assert(gpu_kernel.get_B() == in_channels);
    if (get_use_bias()) {
        assert(gpu_bias.get_DIM() == out_channels);
    }

    //kernel.get_H() = -get_W() = 16 which is also the stride
    assert( (H-P_H) % stride_h  == 0);
    vit_size out_h = ( (H-P_H) / stride_h ) + 1;

    assert( (W-P_W) % stride_w == 0);
    vit_size out_w = ( (W-P_W) / stride_w ) + 1;

    //create as many streams as max 32 or the number of patches
    //set as many SM: fetch the SM.
    u_int stream_n = out_h * out_w < 32 ? out_h * out_w : 32;
    cudaStream_t streams[stream_n];
    cublasHandle_t cublas_handle[stream_n];

    streams[0] = stream;
    cublas_handle[0] = cublasH; 
    for (int i = 1; i < stream_n; ++i){
        CUDA_CHECK( cudaStreamCreate(&streams[i]));
        CUBLAS_CHECK(cublasCreate(&cublas_handle[i]));
        CUBLAS_CHECK(cublasSetStream(cublas_handle[i], streams[i]));
    }// Allocate the memory for the computation

    // iterate over all the channels and launch all the kernels
    // vit_float val = 0.0;
    vit_size x_stride = 0;
    vit_size k_stride = 0;
    vit_size res_stride = 0;

    vit_size threads_per_block = 128;
    vit_size blocks_n = ((P_H * P_W * C) / threads_per_block) + 1;
    
    vit_float * d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(vit_float) * B * out_channels * out_h * out_w));  
    vit_size stream_idx = 0;
    for (int batch=0;batch<B;++batch) {
        for (int y_c=0;y_c<out_channels;++y_c) {
            // Iterate over all the patches (14x14)
            for (int y_h=0;y_h<out_h;++y_h) {
                for (int y_w=0;y_w<out_w;++y_w) {
                    if(stream_idx == stream_n)
                        stream_idx = 0;

                    // val = get_use_bias() ? gpu_bias.at(y_c) : 0;
                    
                    //compute the convolution of the patch for the channel
                    x_stride = y_w * P_H * P_W * C + y_h * P_H * P_W * C * out_w + //patches
                        batch * P_H * P_W * C * out_h * out_w;
                    k_stride = P_H * P_W * C * y_c;
                    res_stride = y_c + out_channels * y_w + out_channels * out_w * y_h + 
                        batch * out_channels * out_h * out_w;
                    // printf("Iteration ch:%d - ", y_c);
                    // printf("Iteration patch:[%d,%d] - %d\n", y_w,y_h, res_stride);
                    // if (cublas_handle[stream_idx] == nullptr) {
                    //     std::cerr << "Using null cuBLAS handle at stream " << stream_idx << std::endl;
                    // }

                    CUBLAS_CHECK(
                        cublasSdot(
                            cublas_handle[stream_idx],
                            P_H * P_W * C,
                            x_in.get_d_data() + x_stride,1,
                            d_kernel + k_stride,1,
                            d_result + res_stride
                        )
                    );
                    //bias case, schedule a sum kernel
                    if(get_use_bias()){
                        addScalarKernel<<<threads_per_block,blocks_n,0,streams[stream_idx]>>>( d_result + res_stride,gpu_bias.at(y_c), P_H * P_W * C);
                    }
                    ++stream_idx;
                }
            }
        }
    }

    vit_float * h_y = (vit_float *)malloc(sizeof(vit_float) * B * out_channels * out_h * out_w);
    cudaDeviceSynchronize();
    CUDA_CHECK( cudaMemcpy(h_y,d_result,sizeof(vit_float) * B * out_channels * out_h * out_w, cudaMemcpyDeviceToHost));
    
    // Cleanup
    for (int i = 1; i < stream_n; ++i)
        CUDA_CHECK(cudaStreamDestroy(streams[i]));

    PictureBatch y(h_y,B * out_channels * out_h * out_w,B, out_channels, out_h, out_w);
    x_out = std::move(y);
}

