//cpp types
#include "../include/conv2d.h"
//gpu types
#include "gpu_datatypes.h"

//Sub-class that implement the Conv2d accelerated in the GPU
class GPUConv2d : public Conv2d
{
    private:
        PictureBatch gpu_kernel;
        RowVector gpu_bias; // can be on device or on host
        vit_float * d_kernel;
        vit_float * d_bias; // can be on device or on host
        cublasHandle_t cublasH;
        cudaStream_t stream;

        void single_forward(GPUPictureBatch& x_in, PictureBatch& x_out);
        void unified_forward(GPUPictureBatch& x_in, PictureBatch& x_out);
        void memory_forward(GPUPictureBatch& x_in, PictureBatch& x_out);
        void time_memory_forward(GPUPictureBatch& x_in, PictureBatch& x_out, benchmark_time& time);
        // void memory_timed_forward(GPUPictureBatch& x_in, PictureBatch& x_out);
        
    public:
        GPUConv2d(
            conv_kernel_shape kernel_shape, vit_bool _use_bias=true
        );
        GPUConv2d(
            vit_size _in_channels, vit_size _out_channels, conv_kernel_shape kernel_shape, vit_bool _use_bias=true
        );

        GPUConv2d(
            cublasHandle_t& in_cublasH, cudaStream_t& in_stream,
            vit_size _in_channels, vit_size _out_channels, conv_kernel_shape kernel_shape,
            vit_bool _use_bias=true
        );
        ~GPUConv2d();

        //Get the kernels pointers from the device
        void get_gpu_kernel();
        void get_gpu_bias();
        //Set the device kernels
        void set_d_kernel(vit_float * device_idx);
        //Stream and handle methods
        void set_stream(cudaStream_t in_stream);
        void create_handle();
        void bind_handle();
        void set_handle(cublasHandle_t in_cublasH);
        //Load the kernels
        void move_kernel(PictureBatch& _kernel);
        void move_bias(RowVector& _bias, vit_bool on_gpu = false);
        //Device memory free
        void kernel_free();
        void bias_free();
        void cuda_free();
        //override
        // void forward(const PictureBatch& x_in, PictureBatch& x_out) const;
        //gpu tests
        void test_forward(GPUPictureBatch& x_in, PictureBatch& x_out, u_int level);
        void timed_forward(GPUPictureBatch& x_in, PictureBatch& x_out, u_int level, benchmark_time& time);
        // void forward(GPUPictureBatch x_in, GPUPictureBatch x_out);

};