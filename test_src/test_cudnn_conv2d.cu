#include "../gpu_include/cudnn_conv2d.h"
#include "../include/conv2d.h"

double compare_results(Tensor &y, half * gpu_y){
    double avg = 0;
    for(u_int b = 0; b < y.get_B(); b++){
        for(u_int t = 0; t < y.get_N(); t++){
            for(u_int c = 0; c < y.get_C(); c++){
                assert(!isnanf( y.at(b,t,c)));
                assert(!isnanf( __half2float(gpu_y[c + y.get_C() * t + y.get_C() * y.get_N() * b])));
                avg += (double)abs(y.at(b,t,c) - __half2float(gpu_y[c + y.get_C() * t + y.get_C() * y.get_N() * b]));
                
            }
        }
    }
    return avg / (double(y.get_B()) * y.get_N() * y.get_C());
}

void cudnn_conv2d_test(half * h_x, half * h_w, half * h_y, convolution_dim dim){
    u_int batch = dim.batch, height =  dim.height, width =  dim.width, channels =  dim.channels, embeddings = dim.embeddings;
    int Ho = dim.Ho, Wo = dim.Wo; //patch size 2x2 => 4 tokens
    assert(height % Ho == 0); assert(width % Wo == 0);
    u_int y_height = dim.y_height, y_width = dim.y_width;
    // 0. Create the cudnn handle
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    // 1. Create the convolution descriptor and populate it
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

    CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc, CONV_MATH_TYPE)); //Disableing Tensor Core ops
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        conv_desc,
        0,0, //no padding
        Ho,Wo,
        1,1, //no dilation
        CONV_MODE,
        CONV_DATA_TYPE
    ));

    // 2. Create x Tensor descriptor [B,C,H,W]
    cudnnTensorDescriptor_t x_desc;
    cudnnTensorFormat_t x_format = CUDNN_TENSOR_NCHW;
    CUDNN_CHECK(
        cudnnCreateTensorDescriptor(&x_desc)
    );
    CUDNN_CHECK(
        cudnnSetTensor4dDescriptor(
            x_desc,
            x_format, 
            CONV_INPUT_DATA_TYPE,
            batch, channels, height, width
        )
    );
    // 3. Create the w Tensor descriptor
    cudnnFilterDescriptor_t w_desc;
    cudnnTensorFormat_t w_format = CUDNN_TENSOR_NCHW; /*NCHW == KCRS K output C input R filter rows S filter columns*/
    CUDNN_CHECK(
        cudnnCreateFilterDescriptor(&w_desc)
    );
    CUDNN_CHECK(
        cudnnSetFilter4dDescriptor(
            w_desc,
            CONV_INPUT_DATA_TYPE,
            w_format,
            embeddings, channels, Ho, Wo 
        )
    );
    // 4. Create the y Tensor descriptor
    cudnnTensorDescriptor_t y_desc;
    cudnnTensorFormat_t y_format = CUDNN_TENSOR_NCHW;
    CUDNN_CHECK(
        cudnnCreateTensorDescriptor(&y_desc)
    );
    CUDNN_CHECK(
        cudnnSetTensor4dDescriptor(
            y_desc,
            y_format, 
            CONV_INPUT_DATA_TYPE,
            batch, embeddings, y_height, y_width
        )
    );
    
    // 5. Fetch the algorithm for executing the convolution
    cudnnConvolutionFwdAlgo_t algo; cudnnConvolutionFwdAlgoPerf_t perf_results[10];
    int returned_algo_count = 0;
    CUDNN_CHECK(
        cudnnFindConvolutionForwardAlgorithm(
            handle,
            x_desc, w_desc, conv_desc, y_desc,
            10, &returned_algo_count, perf_results
        )
    );
    algo = perf_results[0].algo;

    // 6. Fetch the workspace size and allocate it
    void * d_workspace; size_t workspace_size = 0; 
    CUDNN_CHECK(
        cudnnGetConvolutionForwardWorkspaceSize(
            handle,
            x_desc, w_desc, conv_desc, y_desc,
            algo, 
            &workspace_size
        )
    )
    CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));

    // 7. Allocate everything on the device
    void * d_x, * d_w, * d_y;
    CUDA_CHECK(cudaMalloc(&d_x, sizeof(half) * batch * channels * height * width));
    CUDA_CHECK(cudaMalloc(&d_w, sizeof(half) * embeddings * channels * Ho * Wo));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, sizeof(half) * batch * channels * height * width, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w, sizeof(half) * embeddings * channels * Ho * Wo, cudaMemcpyHostToDevice));

    int y_b, y_c, y_h, y_w;
    CUDNN_CHECK(
        cudnnGetConvolution2dForwardOutputDim(
            conv_desc, x_desc, w_desc,
            &y_b, &y_c, &y_h, &y_w
        )
    );
    CUDA_CHECK(cudaMalloc(&d_y, sizeof(half) * y_b * y_c * y_h * y_w));
    cout << "Y" << endl << "["<< y_b<<"," << y_c <<"," << y_h <<","<< y_w <<"]"<< endl;


    // 8. Execute the convolution
    float alpha = 1.0f, beta = 0;
    CUDNN_CHECK(
        cudnnConvolutionForward(
            handle,
            &alpha,
            x_desc, d_x, /*x*/
            w_desc, d_w, /*w*/
            conv_desc,
            algo, /*algo*/ 
            d_workspace, workspace_size,/*workspace*/
            &beta,
            y_desc, d_y /*y*/
        )
    );

    assert(y_b == batch); assert(y_c == embeddings); assert( y_h == y_height ); assert( y_w == y_width );
    void * d_out; CUDA_CHECK(cudaMalloc(&d_out,sizeof(half)* y_b * y_c * y_h* y_w));
    int block_dim = 256, blocks_n = y_b * y_c * y_h* y_w;
    transpose_tensor3d<<<blocks_n, block_dim>>>((half*)d_y,(half*)d_out,y_b,y_c,y_h * y_w);
    // TO REMOVE works
    // int block_dim = 4, blocks_n = 4;
    // transpose_strided_tensor3d<<<blocks_n, block_dim>>>((half*)d_y,(half*)d_out,y_b,y_c,y_h * y_w);
    //-----
    CUDA_CHECK(cudaMemcpy(h_y, d_out,sizeof(half) * batch * embeddings * y_height * y_width ,cudaMemcpyDeviceToHost));

}


void cpu_gpu_comparison(){
    u_int batch = 2, height = 4 ,width = 4 ,channels = 3 ,embeddings = 5;
    int Ho = 2, Wo = 2; //patch size 2x2 => 4 tokens
    convolution_dim dim(
        batch,
        channels,
        height,width,
        embeddings,
        Ho,Wo
    );
    u_int output_elements_number = batch * embeddings * dim.y_height * dim.y_width;
    vector<float> x_f = {
        0,0,0,0, 1,1,1,1, 1,1,1,1, 0,0,0,0,
        0,0,0,0, 1,1,1,1, 1,1,1,1, 0,0,0,0,
        0,0,0,0, 1,1,1,1, 1,1,1,1, 0,0,0,0,

        0,0,0,0, 1,1,1,1, 1,1,1,1, 0,0,0,0,
        0,0,0,0, 1,1,1,1, 1,1,1,1, 0,0,0,0,
        0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0,
    };

    vector<float> w_f = {
        0, 1, 
        0, 1,

        1, 0,
        0, 1,

        1, 0,
        1, 0,

        0, 0, 0, 0, 1,0,0,1, 0,0,0,0,
        0, 0, 0, 0, 1,0,0,1, 0,0,0,0,
        0, 0, 0, 0, 1,0,0,1, 0,0,0,0,
        0, 0, 0, 0, 1,0,0,1, 0,0,0,0,
    };

    float * y_f = (float *)malloc(sizeof(float) * output_elements_number);

    // input is [B,H,W,C] OR [B,C,H,W]
    h_tensor h_x(x_f.data(),batch,channels,height, width);

    // mask is [EMB,C,Ho,Wo]
    h_tensor h_w(w_f.data(),embeddings , channels , Ho , Wo);

    // output is [B,EMB,Y_H,Y_W] with T = (H / Ho) * (W / Wo) = Y_H * Y_W
    vector<half> h_y(output_elements_number);

    // call the gpu method
    cudnn_conv2d_test(h_x.data, h_w.data, h_y.data(), dim);
    
    f16_to_f32(h_y.data(), y_f, output_elements_number);
    Tensor y(y_f, output_elements_number, batch, dim.y_height * dim.y_width,embeddings); 
    
    // -- COMPARISON --
    // CPU reference
    PictureBatch x(x_f.data(), batch * channels * height * width,batch, channels, height, width);
    PictureBatch w(w_f.data(), embeddings * channels * Ho * Wo, embeddings, channels, Ho, Wo);
    x.print();
    w.print();
    
    PictureBatch y_pic(batch, embeddings, dim.y_height, dim.y_width);
    Conv2d cpu_layer(channels,embeddings,Ho, Wo, Ho, Wo, false);
    cpu_layer.move_kernel(w);
    cpu_layer.forward(x,y_pic);
    Tensor cpu_y(batch, dim.y_height * dim.y_width, embeddings);
    y_pic.flatten_to_tensor(cpu_y);

    cout << "CPU" << endl; y_pic.print();
    cout << "CPU flatten" << endl; cpu_y.print();
    cout << "GPU flatten" << endl; y.print();
    cout << "first ten elements: ";
    for(int i = 0; i < 10; i++)
        cout << y_f[i] << " ";
    cout << endl;
    
}

void gpu_comparison(bool bias, bool debug = false){
    // 0. Dimensions definitions
    u_int batch, height ,width ,channels ,embeddings;
    int Ho, Wo;
    if(debug){
        batch = 2, height = 4 ,width = 4 ,channels = 3 ,embeddings = 5;
        Ho = 2, Wo = 2;
    }
    else{
        batch = 8, height = 224 ,width = 224 ,channels = 3 ,embeddings = 768;
        Ho = 16, Wo = 16;
    }
    convolution_dim dim(
        batch,
        channels,
        height,width,
        embeddings,
        Ho,Wo
    );
    u_int input_elements_number = batch * channels * height * width;
    u_int filter_elements_number = embeddings * channels * Ho * Wo;
    u_int output_elements_number = batch * embeddings * dim.y_height * dim.y_width;
    
    cout << "X: [" << batch << ","<< channels << ","<< height << ","<< width << "]" << endl;
    cout << "W: [" << embeddings << ","<< channels << ","<< Ho << ","<< Wo << "]" << endl;
    cout << "Y: [" << batch << ","<< embeddings << ","<< dim.y_height << ","<< dim.y_width << "]" << endl;
    cout << "debug: " << yesno(debug) << endl;
    cout << "bias: " << yesno(bias)  << endl;

    // 1. Random data generation
    float * h_x, * h_w, * h_b,* h_y;
    h_x = (float *)malloc(sizeof(float) * input_elements_number);
    h_w = (float *)malloc(sizeof(float) * filter_elements_number);
    h_b = (float *)malloc(sizeof(float) * embeddings);
    h_y = (float *)malloc(sizeof(float) * output_elements_number);
    half * x_half, * w_half, * b_half,* y_half;
    x_half = (half*)malloc(sizeof(half) * input_elements_number);
    w_half = (half*)malloc(sizeof(half) * filter_elements_number);
    b_half = (half*)malloc(sizeof(half) * embeddings);
    y_half = (half *)malloc(sizeof(half) * output_elements_number);

    void * d_x, * d_w, * d_b,* d_y;
    CUDA_CHECK(cudaMalloc(&d_x, sizeof(float) * input_elements_number)); //float now then reassigned to half
    CUDA_CHECK(cudaMalloc(&d_w, sizeof(float) * filter_elements_number));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(float) * embeddings));
    CUDA_CHECK(cudaMalloc(&d_y, sizeof(half) * output_elements_number));
    u_long seed = std::chrono::high_resolution_clock::now()
        .time_since_epoch()
        .count();
    u_int block_dim = 256, blocks_n = (input_elements_number / block_dim) + 1;
    generate_reference<<<blocks_n, block_dim>>>((float *)d_x,input_elements_number,1.0,seed);
    blocks_n = (filter_elements_number / block_dim) + 1;
    generate_reference<<<blocks_n, block_dim>>>((float *)d_w,filter_elements_number,1.0,seed);
    blocks_n = (embeddings / block_dim) + 1;
    generate_reference<<<blocks_n, block_dim>>>((float *)d_b,embeddings,1.0,seed);


    CUDA_CHECK(cudaMemcpy(h_x, d_x, sizeof(float) * input_elements_number, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_w, d_w, sizeof(float) * filter_elements_number, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b, d_b, sizeof(float) * embeddings, cudaMemcpyDeviceToHost));

    /*Convert to half*/
    cudaFree(d_x); cudaFree(d_w); cudaFree(d_b);
    CUDA_CHECK(cudaMalloc(&d_x, sizeof(half) * input_elements_number));
    CUDA_CHECK(cudaMalloc(&d_w, sizeof(half) * filter_elements_number));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(half) * embeddings));
    f32_to_f16(h_x,x_half, input_elements_number);
    f32_to_f16(h_w,w_half, filter_elements_number);
    f32_to_f16(h_b,b_half, embeddings);
    CUDA_CHECK(cudaMemcpy(d_x,x_half, sizeof(half) * input_elements_number, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w,w_half, sizeof(half) * filter_elements_number, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b,b_half, sizeof(half) * embeddings, cudaMemcpyHostToDevice));

    // 2. CPU reference
    PictureBatch x(h_x, input_elements_number, batch, channels, height, width);
    PictureBatch w(h_w, filter_elements_number, embeddings, channels, Ho, Wo);
    if(debug) {
        x.print();
        w.print();
    }
    PictureBatch y_pic(batch, embeddings, dim.y_height, dim.y_width);
    Tensor y(batch, dim.y_height * dim.y_width , embeddings);

    Conv2d cpu_conv2d(channels,embeddings,Ho,Wo,Ho,Wo,bias);
    if(bias){
        RowVector cpu_bias(h_b,embeddings);
        if(debug) cpu_bias.print();
        cpu_conv2d.move_bias(cpu_bias);
    }
    cpu_conv2d.move_kernel(w);
    cpu_conv2d.forward(x,y_pic);
    if(debug) y_pic.print();
    y_pic.flatten_to_tensor(y);
    if(debug) y.print();
    // 3. GPU Strided   
    /*Initialize the descriptors*/
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));
    
    convolution_desc desc; desc.handle = handle;
    init_conv2d_descriptors(desc, dim, bias,debug);

    /*Execute the convolution*/
    if(bias){ 
        execute_cudnn_conv2d_bias(
            d_x, d_w, d_y, d_b,
            desc
        );
    }
    else{
        execute_cudnn_conv2d(
            d_x, d_w, d_y,
            desc
        );
    }
    if(debug){
        CUDA_CHECK(cudaMemcpy(y_half, d_y, sizeof(half) * output_elements_number, cudaMemcpyDeviceToHost));
        f16_to_f32(y_half, h_y, output_elements_number);
        PictureBatch gpu_y(h_y, output_elements_number, batch, embeddings, dim.y_height , dim.y_width);
        cout << "Pic gpu y: " << endl; gpu_y.print();
    }
    /*Transpose*/
    half * d_out; CUDA_CHECK(cudaMalloc(&d_out, sizeof(half) * output_elements_number));
    block_dim = 256; blocks_n = (output_elements_number / (256 * 4)) + 1;/* We suppose 4 iterations per thread */
    transpose_strided_tensor3d<<<blocks_n, block_dim>>>((half*)d_y,d_out,dim.batch,dim.embeddings,dim.y_height * dim.y_width);
    
    CUDA_CHECK(cudaMemcpy(y_half, d_out, sizeof(half) * output_elements_number, cudaMemcpyDeviceToHost));
    if(debug){
        f16_to_f32(y_half, h_y, output_elements_number);
        Tensor gpu_y(h_y, output_elements_number, batch, dim.y_height * dim.y_width , embeddings);
        cout << "gpu_y: " << endl; gpu_y.print();
    }
    cout << "CPU GPU comparison result: " << compare_results(y, y_half) << endl;
    
    //Tiling (multi stream)
}


int main() {
    test_type test = GPU_COMPARISON;
    
    if(test == CPU_COMPARISON){
        cpu_gpu_comparison();
    }
    else{
        gpu_comparison(true, false);
    }

    return 0;
}