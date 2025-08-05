#include "../include/datatypes.h"
#include "cuda_utils.h"

#include <cuda_runtime_api.h>
#include <cublas_v2.h>

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