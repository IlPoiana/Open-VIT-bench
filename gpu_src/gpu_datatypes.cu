#include "../gpu_include/gpu_datatypes.h"

#include <assert.h>
#include <sys/time.h>

GPUPictureBatch::GPUPictureBatch(picture_shape shape): PictureBatch(shape.B,shape.C,shape.H,shape.W){
}

//initialize the batch class and allocate the data on the GPU
GPUPictureBatch::GPUPictureBatch(
     vit_float* _data, vit_size data_dim, picture_shape shape,
     vit_bool linearize_flag, conv_kernel_shape kernel_shape): PictureBatch(shape.B,shape.C,shape.H,shape.W) {   
    assert(data_dim < UINT_MAX);
    //initialize stream
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    
    if(linearize_flag){
        //kernel.get_H() = -get_W() = 16 which is also the stride
        assert( (shape.H-kernel_shape.H) % kernel_shape.h_stride  == 0);
        int out_h = ( (shape.H-kernel_shape.H) / kernel_shape.h_stride ) + 1;

        assert( (shape.W-kernel_shape.W) % kernel_shape.w_stride == 0);
        int out_w = ( (shape.W-kernel_shape.W) / kernel_shape.w_stride ) + 1;
        //Allocate total patch size on the GPU
        vit_size d_size = out_h * out_w * kernel_shape.W * kernel_shape.H * shape.C * shape.B;
        printf("d_size: %d\n", d_size);
        vit_float * linearized_data = (vit_float *)malloc(sizeof(vit_float) * d_size);
        //linearize the input data
        linearize(_data, linearized_data, shape, kernel_shape);
        
        //initialize the memory
        CUDA_CHECK(cudaMalloc(&d_data, sizeof(vit_float) * d_size));
    
        
        //copy on the device the linearized matrix
        CUDA_CHECK(cudaMemcpyAsync(d_data, linearized_data, sizeof(vit_float) * d_size, cudaMemcpyHostToDevice,
        stream));
    }
    else{
        //initialize the memory
        CUDA_CHECK(cudaMalloc(&d_data, sizeof(vit_float) * data_dim));
    
        CUDA_CHECK(cudaMemcpyAsync(d_data, _data, sizeof(vit_float) * data_dim, cudaMemcpyHostToDevice,
        stream));
    }

}

//Initialization with an already existing stream
GPUPictureBatch::GPUPictureBatch(vit_float* _data, vit_size data_dim, picture_shape shape,
     cudaStream_t in_stream, vit_bool linearize_flag, conv_kernel_shape kernel_shape):
     PictureBatch(shape.B,shape.C,shape.H,shape.W) {   
    
    assert(data_dim < UINT_MAX);
    //initialize stream
    set_stream(in_stream);
    
    if(linearize_flag){
        //kernel.get_H() = -get_W() = 16 which is also the stride
        assert( (shape.H-kernel_shape.H) % kernel_shape.h_stride  == 0);
        int out_h = ( (shape.H-kernel_shape.H) / kernel_shape.h_stride ) + 1;

        assert( (shape.W-kernel_shape.W) % kernel_shape.w_stride == 0);
        int out_w = ( (shape.W-kernel_shape.W) / kernel_shape.w_stride ) + 1;
        //Allocate total patch size on the GPU
        vit_size d_size = out_h * out_w * kernel_shape.W * kernel_shape.H * shape.C * shape.B;
        printf("d_size: %d\n", d_size);
        vit_float * linearized_data = (vit_float *)malloc(sizeof(vit_float) * d_size);
        //linearize the input data
        linearize(_data, linearized_data, shape, kernel_shape);
        
        //initialize the memory
        CUDA_CHECK(cudaMalloc(&d_data, sizeof(vit_float) * d_size));
    
        
        //copy on the device the linearized matrix
        CUDA_CHECK(cudaMemcpyAsync(d_data, linearized_data, sizeof(vit_float) * d_size, cudaMemcpyHostToDevice,
        stream));
    }
    else{
        //initialize the memory
        CUDA_CHECK(cudaMalloc(&d_data, sizeof(vit_float) * data_dim));
    
        CUDA_CHECK(cudaMemcpyAsync(d_data, _data, sizeof(vit_float) * data_dim, cudaMemcpyHostToDevice,
        stream));
    }

}

/*
TIME VERSIONS
*/

GPUPictureBatch::GPUPictureBatch(
     vit_float* _data, vit_size data_dim, picture_shape shape,
     vit_bool linearize_flag, conv_kernel_shape kernel_shape,
     benchmark_time& time): PictureBatch(shape.B,shape.C,shape.H,shape.W) {   
    assert(data_dim < UINT_MAX);
    //initialize stream
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    
    if(linearize_flag){
        struct timeval start={0,0};
        struct timeval stop={0,0};
        gettimeofday(&start, (struct timezone*)0);


        assert( (shape.H-kernel_shape.H) % kernel_shape.h_stride  == 0);
        int out_h = ( (shape.H-kernel_shape.H) / kernel_shape.h_stride ) + 1;

        assert( (shape.W-kernel_shape.W) % kernel_shape.w_stride == 0);
        int out_w = ( (shape.W-kernel_shape.W) / kernel_shape.w_stride ) + 1;
        
        //Allocate total patch size on the GPU
        vit_size d_size = out_h * out_w * kernel_shape.W * kernel_shape.H * shape.C * shape.B;
        printf("d_size: %d\n", d_size);
        vit_float * linearized_data = (vit_float *)malloc(sizeof(vit_float) * d_size);
        
        //linearize the input data
        linearize(_data, linearized_data, shape, kernel_shape);

        gettimeofday(&stop, (struct timezone*)0);
        float elapsed_time = ((stop.tv_sec-start.tv_sec)*1.e6+(stop.tv_usec-start.tv_usec));
        time.preprocess.push_back(elapsed_time);
        
        //initialize the memory
        CUDA_CHECK(cudaMalloc(&d_data, sizeof(vit_float) * d_size));
    
        //copy on the device the linearized matrix
        CUDA_CHECK(cudaMemcpyAsync(d_data, linearized_data, sizeof(vit_float) * d_size, cudaMemcpyHostToDevice,
        stream));
    }
    else{
        //initialize the memory
        CUDA_CHECK(cudaMalloc(&d_data, sizeof(vit_float) * data_dim));
    
        CUDA_CHECK(cudaMemcpyAsync(d_data, _data, sizeof(vit_float) * data_dim, cudaMemcpyHostToDevice,
        stream));
    }

}

GPUPictureBatch::GPUPictureBatch(
     vit_float *_data, vit_size data_dim, picture_shape shape,
     cudaStream_t in_stream,
     vit_bool linearize_flag, conv_kernel_shape kernel_shape,
     benchmark_time& time
    ): PictureBatch(shape.B,shape.C,shape.H,shape.W){
    
    assert(data_dim < UINT_MAX);
    //initialize stream
    set_stream(in_stream);
    
    if(linearize_flag){
        struct timeval start={0,0};
        struct timeval stop={0,0};
        gettimeofday(&start, (struct timezone*)0);

        assert( (shape.H-kernel_shape.H) % kernel_shape.h_stride  == 0);
        int out_h = ( (shape.H-kernel_shape.H) / kernel_shape.h_stride ) + 1;

        assert( (shape.W-kernel_shape.W) % kernel_shape.w_stride == 0);
        int out_w = ( (shape.W-kernel_shape.W) / kernel_shape.w_stride ) + 1;
        //Allocate total patch size on the GPU
        vit_size d_size = out_h * out_w * kernel_shape.W * kernel_shape.H * shape.C * shape.B;
        printf("d_size: %d\n", d_size);
        vit_float * linearized_data = (vit_float *)malloc(sizeof(vit_float) * d_size);
        //linearize the input data
        linearize(_data, linearized_data, shape, kernel_shape);
        gettimeofday(&stop, (struct timezone*)0);
        float elapsed_time = ((stop.tv_sec-start.tv_sec)*1.e6+(stop.tv_usec-start.tv_usec));
        time.preprocess.push_back(elapsed_time);
        
        //initialize the memory
        CUDA_CHECK(cudaMalloc(&d_data, sizeof(vit_float) * d_size));
        
        //copy on the device the linearized matrix
        CUDA_CHECK(cudaMemcpyAsync(d_data, linearized_data, sizeof(vit_float) * d_size, cudaMemcpyHostToDevice,
        stream));
    }
    else{
        //initialize the memory
        CUDA_CHECK(cudaMalloc(&d_data, sizeof(vit_float) * data_dim));
    
        CUDA_CHECK(cudaMemcpyAsync(d_data, _data, sizeof(vit_float) * data_dim, cudaMemcpyHostToDevice,
        stream));
    }
}

vit_float *GPUPictureBatch::get_d_data() const
{
    return d_data;
}

void GPUPictureBatch::set_d_data(vit_float *device_idx) {
    d_data = device_idx;
}

void GPUPictureBatch::create_stream(){
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
}

void GPUPictureBatch::set_stream(cudaStream_t& in_stream){
    stream = in_stream; 
}
cudaStream_t GPUPictureBatch::get_stream(){
    return stream;
}

void GPUPictureBatch::data_free(){
    CUDA_CHECK(cudaFree(d_data));
};
void GPUPictureBatch::stream_free(){
    CUDA_CHECK(cudaStreamDestroy(stream));
};
void GPUPictureBatch::cuda_free(){
    data_free();
    stream_free();
};
