#include "../gpu_include/cudnn_conv2d.h"
#include "../include/conv2d.h"

float compare_results(Tensor &y, half * gpu_y){
    float tolerance = 1e-3f;
    double avg = 0;
    float gpu_val;
    float total_elem_num = y.get_B() * y.get_N() * y.get_C();
    for(u_int b = 0; b < y.get_B(); b++){
        for(u_int t = 0; t < y.get_N(); t++){
            for(u_int c = 0; c < y.get_C(); c++){
                assert(!isnanf( y.at(b,t,c)));
                assert(!isnanf( __half2float(gpu_y[c + y.get_C() * t + y.get_C() * y.get_N() * b])));
                gpu_val = __half2float(gpu_y[c + y.get_C() * t + y.get_C() * y.get_N() * b]);
                avg += 
                    (
                        (double)abs(y.at(b,t,c) - gpu_val)
                        /
                        (double)max(abs(y.at(b,t,c)), tolerance)
                    )
                    / total_elem_num;
            }
        }
    }
    return float(avg);
}


void cpu_gpu_comparison(bool bias, bool debug = false){
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

    
    random_device rd;          
    mt19937 gen(rd());         
    uniform_real_distribution<float> dist(-0.1f, 0.1f);

    size_t loop_range = max(input_elements_number, filter_elements_number);
    for(size_t i = 0; i < loop_range; i++){
        if(i < embeddings){
            h_b[i] = dist(gen);
        }
        if(i < input_elements_number){
            h_x[i] = dist(gen);
        }
        if(i < filter_elements_number){
            h_w[i] = dist(gen);
        }
    }

    f32_to_f16(h_x, x_half, input_elements_number);
    f32_to_f16(h_w, w_half, filter_elements_number);           
    f32_to_f16(h_b, b_half, embeddings); 

    void * d_x, * d_w, * d_b,* d_y;
    CUDA_CHECK(cudaMalloc(&d_x, sizeof(float) * input_elements_number)); //float now then reassigned to half
    CUDA_CHECK(cudaMalloc(&d_w, sizeof(float) * filter_elements_number));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(float) * embeddings));
    CUDA_CHECK(cudaMalloc(&d_y, sizeof(half) * output_elements_number));
    CUDA_CHECK(cudaMemcpy(d_x, x_half, sizeof(float) * input_elements_number, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, w_half, sizeof(float) * filter_elements_number, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b_half, sizeof(float) * embeddings, cudaMemcpyHostToDevice));

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

    // 3. GPU    
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
    int block_dim = 256; int blocks_n = (output_elements_number / (256 * 4)) + 1;/* We suppose 4 iterations per thread */
    transpose_strided_tensor3d<<<blocks_n, block_dim>>>(
        (half*)d_y, d_out,
        dim.batch, dim.embeddings, dim.y_height * dim.y_width
    );
    
    CUDA_CHECK(cudaMemcpy(y_half, d_out, sizeof(half) * output_elements_number, cudaMemcpyDeviceToHost));
    if(debug){
        f16_to_f32(y_half, h_y, output_elements_number);
        Tensor gpu_y(h_y, output_elements_number, batch, dim.y_height * dim.y_width , embeddings);
        cout << "gpu_y: " << endl; gpu_y.print();
    }
    cout << "CPU GPU comparison result: " << compare_results(y, y_half) * 100 << "%" << endl;

    // -Cleanup
    desc.destroy_descriptors();
    
    CUDA_CHECK(cudaFree(d_x)); 
    CUDA_CHECK(cudaFree(d_w)); 
    CUDA_CHECK(cudaFree(d_b)); 
    CUDA_CHECK(cudaFree(d_y)); 
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(desc.d_workspace));
    CUDNN_CHECK(cudnnDestroy(handle));
    free(h_x); free(h_w); free(h_b); free(h_y);
    free(x_half); free(w_half); free(b_half); free(y_half);
}


int main() {
    bool bias = true, debug = false;
    cpu_gpu_comparison(bias, debug);
    
    return 0;
}