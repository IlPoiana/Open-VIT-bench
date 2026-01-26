#include "./cudnn_utils.h"

#define CONV_MATH_TYPE CUDNN_DEFAULT_MATH //CUDNN_FMA_MATH
#define CONV_DATA_TYPE CUDNN_DATA_FLOAT
#define CONV_MODE CUDNN_CROSS_CORRELATION // CUDNN_CONVOLUTION
#define CONV_INPUT_DATA_TYPE CUDNN_DATA_HALF

struct convolution_dim{
    u_int batch, channels, height, width, embeddings;
    int Ho, Wo;
    u_int y_height, y_width;

    convolution_dim();
    convolution_dim(convolution_dim &c_dim);
    convolution_dim(u_int b,u_int c,u_int h,u_int w,u_int e,u_int ho,u_int wo);
};

struct convolution_bias_desc{
    cudnnTensorDescriptor_t b_desc;
    cudnnActivationDescriptor_t act_desc;

    convolution_bias_desc();

    convolution_bias_desc(
        cudnnTensorDescriptor_t &b_desc_,
        cudnnActivationDescriptor_t &act_desc_
    );
};

struct convolution_desc{
    cudnnHandle_t handle;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnTensorDescriptor_t x_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnTensorDescriptor_t y_desc;
    cudnnConvolutionFwdAlgo_t algo;
    convolution_bias_desc bias_d;
    void * d_workspace; size_t workspace_size;

    convolution_desc();

    convolution_desc(
        cudnnHandle_t &handle_,
        cudnnConvolutionDescriptor_t &conv_desc_,
        cudnnTensorDescriptor_t &x_desc_,
        cudnnFilterDescriptor_t &w_desc_,
        cudnnTensorDescriptor_t &y_desc_,
        cudnnConvolutionFwdAlgo_t &algo_,
        void * d_workspace_, size_t &workspace_size_
    );

    convolution_desc(
        cudnnHandle_t &handle_,
        cudnnConvolutionDescriptor_t &conv_desc_,
        cudnnTensorDescriptor_t &x_desc_,
        cudnnFilterDescriptor_t &w_desc_,
        cudnnTensorDescriptor_t &y_desc_,
        cudnnConvolutionFwdAlgo_t &algo_,
        convolution_bias_desc &bias_d_,
        void * d_workspace_, size_t &workspace_size_
    );

    void destroy_descriptors();
};

__global__ void transpose_tensor3d(
    half*  src,half*  dst,
    u_int B, u_int C, u_int T
);

//Stride transpose to have less block scheduled and less call overhead
__global__ void transpose_strided_tensor3d(
    half*  src,half*  dst,
    u_int B, u_int C, u_int T
);

//Initialize the cudnn descriptors for a conv2d operation given all the dimensions involved `dim` variable
void init_conv2d_descriptors(
    convolution_desc &desc,
    convolution_dim &dim,
    bool bias = false,
    bool debug = false
);

//Execute the cudnn convolution on d_x data, with d_w filter storing in d_y.
void execute_cudnn_conv2d(
    void * d_x, void * d_w, void * d_y,
    convolution_desc &desc,
    float alpha = 1.0f, float beta = 0.0f
);

//Merge with the upper method
void execute_cudnn_conv2d_bias(
    void * d_x, void * d_w, void * d_y, void * d_b,
    convolution_desc &desc,
    float alpha = 1.0f, float beta = 0.0f
);


