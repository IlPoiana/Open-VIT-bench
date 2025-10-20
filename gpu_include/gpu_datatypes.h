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
};

// B,C,H,W
struct h_tensor {
    __half * data;
    u_int16_t B;
    u_int16_t C;
    u_int16_t H;
    u_int16_t W;

    h_tensor(float * f32_data, u_int16_t batch, u_int16_t channels, u_int16_t height, u_int16_t width);
};

//Used in the test_* files to choose which comparison perform
enum test_type{
    CPU_COMPARISON,
    GPU_COMPARISON
};

class GPUPictureBatch : public PictureBatch {
    private:
        vit_float * d_data;
        // cublasHandle_t cublasH;
        cudaStream_t stream;

    public:
        GPUPictureBatch(picture_shape shape);
        GPUPictureBatch(
            vit_float* _data, vit_size data_dim, picture_shape shape,
            vit_bool linearize_flag, conv_kernel_shape kernel_shape
        );
        GPUPictureBatch(
            vit_float* _data, vit_size data_dim, picture_shape shape,
            cudaStream_t in_stream, vit_bool linearize_flag, conv_kernel_shape kernel_shape
        );
        //time versions
        GPUPictureBatch(
            vit_float* _data, vit_size data_dim, picture_shape shape,
            vit_bool linearize_flag, conv_kernel_shape kernel_shape,
            benchmark_time& time
        );
        GPUPictureBatch(
            vit_float* _data, vit_size data_dim, picture_shape shape,
            cudaStream_t in_stream, vit_bool linearize_flag, conv_kernel_shape kernel_shape,
            benchmark_time& time
        );
        GPUPictureBatch(const GPUPictureBatch& pic);
        // GPUPictureBatch(PictureBatch&& pic);
        // ~GPUPictureBatch();

        // GPUPictureBatch& operator= (const PictureBatch& pic) = delete;
        // GPUPictureBatch& operator= (PictureBatch&& pic);

        vit_float * get_d_data() const;
        void set_d_data(vit_float * device_idx);
        void create_stream();
        void set_stream(cudaStream_t& in_stream);
        cudaStream_t get_stream() ;

        //Device memory free
        void data_free();
        void stream_free();
        void cuda_free();
};