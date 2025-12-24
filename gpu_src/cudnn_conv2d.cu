#include "../gpu_include/cudnn_conv2d.h"

convolution_dim::convolution_dim(){}

convolution_dim::convolution_dim(convolution_dim &c_dim):
    batch(c_dim.batch),
    channels(c_dim.channels),
    height(c_dim.height), width(c_dim.width),
    embeddings(c_dim.embeddings),
    Ho(c_dim.Ho), Wo(c_dim.Wo)
{
    assert(height % Ho == 0); assert(width % Wo == 0);
    y_height = height / Ho;
    y_width = width / Wo;
}

convolution_dim::convolution_dim(
    u_int b,
    u_int c,
    u_int h,u_int w,
    u_int e,
    u_int ho,u_int wo
): 
    batch(b),
    channels(c),
    height(h), width(w),
    embeddings(e),
    Ho(ho), Wo(wo)
{
    assert(height % Ho == 0); assert(width % Wo == 0);
    y_height = height / Ho;
    y_width = width / Wo;
}

convolution_desc::convolution_desc():
    handle(),
    conv_desc(),
    x_desc(),
    w_desc(),
    y_desc(),
    algo(),
    bias_d(),
    d_workspace(nullptr),
    workspace_size(0)
{}

convolution_desc::convolution_desc(
    cudnnHandle_t &handle_,
    cudnnConvolutionDescriptor_t &conv_desc_,
    cudnnTensorDescriptor_t &x_desc_,
    cudnnFilterDescriptor_t &w_desc_,
    cudnnTensorDescriptor_t &y_desc_,
    cudnnConvolutionFwdAlgo_t &algo_,
    void * d_workspace_, size_t &workspace_size_
):
    handle(handle_),
    conv_desc(conv_desc_),
    x_desc(x_desc_),
    w_desc(w_desc_),
    y_desc(y_desc_),
    algo(algo_),
    bias_d(),
    d_workspace(d_workspace_), workspace_size(workspace_size_)
{}

convolution_desc::convolution_desc(
    cudnnHandle_t &handle_,
    cudnnConvolutionDescriptor_t &conv_desc_,
    cudnnTensorDescriptor_t &x_desc_,
    cudnnFilterDescriptor_t &w_desc_,
    cudnnTensorDescriptor_t &y_desc_,
    cudnnConvolutionFwdAlgo_t &algo_,
    convolution_bias_desc &bias_d_,
    void * d_workspace_, size_t &workspace_size_
):
    handle(handle_),
    conv_desc(conv_desc_),
    x_desc(x_desc_),
    w_desc(w_desc_),
    y_desc(y_desc_),
    algo(algo_),
    bias_d(bias_d_),
    d_workspace(d_workspace_), workspace_size(workspace_size_)
{}

void convolution_desc::destroy_descriptors(){
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyTensorDescriptor(bias_d.b_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyActivationDescriptor(bias_d.act_desc);
}

convolution_bias_desc::convolution_bias_desc():
    b_desc(),
    act_desc()
{}

convolution_bias_desc::convolution_bias_desc(
        cudnnTensorDescriptor_t &b_desc_,
        cudnnActivationDescriptor_t &act_desc_
):
    b_desc(b_desc_),
    act_desc(act_desc_)
{}

__global__ void transpose_tensor3d(
    half*  src,half*  dst,
    u_int B, u_int C, u_int T
){
    u_int idx = blockIdx.x * blockDim.x + threadIdx.x;
    u_int N = B * C * T;
    if (idx >= N) return;

    u_int CT = C * T;
    u_int b = idx / CT;
    u_int rem = idx % CT;
    u_int c = rem / T;
    u_int t = rem % T;

    u_int in_idx = idx;
    u_int out_idx = ((b * T + t) * C) + c;

    dst[out_idx] = src[in_idx];
}

//Stride transpose to have less block scheduled and less call overhead
__global__ void transpose_strided_tensor3d(
    half*  src,half*  dst,
    u_int B, u_int C, u_int T
){
    u_int idx = blockIdx.x * blockDim.x + threadIdx.x;
    u_int N = B * C * T;
    if (idx >= N) return;

    u_int in_idx = idx;
    for(u_int stride = 0; in_idx + stride < N;stride += gridDim.x * blockDim.x){
        u_int strided_idx = idx + stride;
        u_int CT = C * T;
        u_int b = strided_idx / CT;
        u_int rem = strided_idx % CT;
        u_int c = rem / T;
        u_int t = rem % T;
        u_int out_idx = ((b * T + t) * C) + c;

        dst[out_idx] = src[strided_idx];
    }
}

//Initialize the cudnn descriptors for a conv2d operation given all the dimensions involved `dim` variable
void init_conv2d_descriptors(
    convolution_desc &desc,
    convolution_dim &dim,
    bool bias,
    bool debug
){
    // 1. Populate the convolution descriptor
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&desc.conv_desc));
    CUDNN_CHECK(cudnnSetConvolutionMathType(desc.conv_desc, CONV_MATH_TYPE)); //Disableing Tensor Core ops
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        desc.conv_desc,
        0,0, //no padding
        dim.Ho,dim.Wo,
        1,1, //no dilation
        CONV_MODE,
        CONV_DATA_TYPE
    ));

    // 2. Create x Tensor descriptor [B,C,H,W]
    cudnnTensorFormat_t x_format = CUDNN_TENSOR_NCHW;
    CUDNN_CHECK(
        cudnnCreateTensorDescriptor(&desc.x_desc)
    );
    CUDNN_CHECK(
        cudnnSetTensor4dDescriptor(
            desc.x_desc,
            x_format, 
            CONV_INPUT_DATA_TYPE,
            dim.batch, dim.channels, dim.height, dim.width
        )
    );

    // 3. Create the w Tensor descriptor
    cudnnTensorFormat_t w_format = CUDNN_TENSOR_NCHW; /*NCHW == KCRS K output C input R filter rows S filter columns*/
    CUDNN_CHECK(
        cudnnCreateFilterDescriptor(&desc.w_desc)
    );
    CUDNN_CHECK(
        cudnnSetFilter4dDescriptor(
            desc.w_desc,
            CONV_INPUT_DATA_TYPE,
            w_format,
            dim.embeddings, dim.channels, dim.Ho, dim.Wo 
        )
    );

    // 4. Create the bias descriptor
    if(bias){
        cudnnTensorFormat_t b_format = CUDNN_TENSOR_NCHW; //TO CHECK
        CUDNN_CHECK(
            cudnnCreateTensorDescriptor(&desc.bias_d.b_desc)
        );
        CUDNN_CHECK(
            cudnnSetTensor4dDescriptor(
                desc.bias_d.b_desc,
                b_format, 
                CONV_INPUT_DATA_TYPE,
                1, dim.embeddings, 1, 1 /*broadcasting*/
            )
        );
    }

    // 5. Create the y Tensor descriptor
    cudnnTensorFormat_t y_format = CUDNN_TENSOR_NCHW;
    CUDNN_CHECK(
        cudnnCreateTensorDescriptor(&desc.y_desc)
    );
    CUDNN_CHECK(
        cudnnSetTensor4dDescriptor(
            desc.y_desc,
            y_format, 
            CONV_INPUT_DATA_TYPE,
            dim.batch, dim.embeddings, dim.y_height, dim.y_width
        )
    );

    // 6. Create che activaction descriptor
    if(bias){
        CUDNN_CHECK(
            cudnnCreateActivationDescriptor(&desc.bias_d.act_desc);
        );
        CUDNN_CHECK(
            cudnnSetActivationDescriptor(
                desc.bias_d.act_desc,
                CUDNN_ACTIVATION_IDENTITY,
                CUDNN_NOT_PROPAGATE_NAN ,
                0.0
            )
        );
    }
    
    // 7. Fetch the algorithm for executing the convolution
    cudnnConvolutionFwdAlgoPerf_t perf_results[10];
    int returned_algo_count = 0;
    CUDNN_CHECK(
        cudnnFindConvolutionForwardAlgorithm(
            desc.handle,
            desc.x_desc, desc.w_desc, desc.conv_desc, desc.y_desc,
            10, &returned_algo_count, perf_results
        )
    );
    if(bias){
        desc.algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM; //required for identity activation
    }
    else{
        desc.algo = perf_results[0].algo;
    }

    // 8. Fetch the workspace size and allocate it 
    size_t fetched_w_size = 0;
    CUDNN_CHECK(
        cudnnGetConvolutionForwardWorkspaceSize(
            desc.handle,
            desc.x_desc, desc.w_desc, desc.conv_desc, desc.y_desc,
            desc.algo, 
            &fetched_w_size
        )
    )
    if(desc.d_workspace == nullptr){
        desc.workspace_size = fetched_w_size;
        CUDA_CHECK(cudaMalloc(&desc.d_workspace, desc.workspace_size));
    }
    else{
        assert(fetched_w_size < desc.workspace_size);
    }

    int y_b, y_c, y_h, y_w;
    CUDNN_CHECK(
        cudnnGetConvolution2dForwardOutputDim(
            desc.conv_desc, desc.x_desc, desc.w_desc,
            &y_b, &y_c, &y_h, &y_w
        )
    );
    if(debug)
        cout << "Y" << endl << "["<< y_b<<"," << y_c <<"," << y_h <<","<< y_w <<"]"<< endl;

    assert(y_b == dim.batch); assert(y_c == dim.embeddings); assert( y_h == dim.y_height ); assert( y_w == dim.y_width );

}

//Execute the cudnn convolution on d_x data, with d_w filter storing in d_y.
void execute_cudnn_conv2d(
    void * d_x, void * d_w, void * d_y,
    convolution_desc &desc,
    float alpha, float beta
){
    CUDNN_CHECK(
        cudnnConvolutionForward(
            desc.handle,
            &alpha,
            desc.x_desc, d_x, /*x*/
            desc.w_desc, d_w, /*w*/
            desc.conv_desc,
            desc.algo, /*algo*/ 
            desc.d_workspace, desc.workspace_size,/*workspace*/
            &beta,
            desc.y_desc, d_y /*y*/
        )
    );
}

/*TO REMOVE merge this into the upper method*/
void execute_cudnn_conv2d_bias(
    void * d_x, void * d_w, void * d_y, void * d_b,
    convolution_desc &desc,
    float alpha, float beta
){
    CUDNN_CHECK(
        cudnnConvolutionBiasActivationForward(
            desc.handle,
            &alpha,
            desc.x_desc, d_x, /*x*/
            desc.w_desc, d_w, /*w*/
            desc.conv_desc,
            desc.algo, /*algo*/ 
            desc.d_workspace, desc.workspace_size,/*workspace*/
            &beta,
            desc.y_desc, d_y, /*y*/
            desc.bias_d.b_desc , d_b,
            desc.bias_d.act_desc ,
            desc.y_desc, d_y /*y*/            
        )
    );
}