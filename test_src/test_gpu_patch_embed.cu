#include "../gpu_include/gpu_patch_embedder.h"
#include "../include/vision_transformer.h"

#define STREAM_N 8

/*
----
This tests includes the patch embedder + position embedder.
Basically the entire part before the encoder blocks
----
*/

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

// Execution equivalent to my implementation of GpuPatchEmbedder, CPU => PatchEmbed + PositionEmbed
Tensor cpu_baseline(
    convolution_dim &conv_dim,
    float * conv_weights,
    float * conv_bias,
    float * positional_embeddings,
    float * x_data,
    bool debug

){
    convolution_dim cd = conv_dim;
    u_int tokens = (cd.y_height * cd.y_width);
    // -Convolution weights and bias
    PictureBatch k(conv_weights, cd.channels* cd.embeddings * cd.Ho * cd.Wo, cd.embeddings, cd.channels , cd.Ho , cd.Wo);
    if(debug){
        cout << "### k" << endl;
        k.print();
    }

    RowVector b(conv_bias, cd.embeddings);
    if(debug) {
        cout << "### b" << endl;
        b.print();
    }   

    // -Convolution layer
    Conv2d c2d(cd.channels,cd.embeddings, cd.Ho, cd.Wo, cd.Ho, cd.Wo, true);
    c2d.move_kernel(k);
    c2d.move_bias(b);
    bool c2d_bias = true, strict_img_size = true, dynamic_img_pad = false, use_norm = false; 
    
    // -Patch Embedder
    PatchEmbed pe(
        cd.height, cd.width, cd.Ho, cd.Wo, cd.channels, cd.embeddings,
        c2d_bias, strict_img_size, dynamic_img_pad, use_norm
    ); //use norm set to true ==> use_pre_norm = false
    pe.move_c2d(c2d);

    // -Positional Embeddings weights
    Matrix pos_emb(positional_embeddings, cd.embeddings * (tokens + 1), (tokens + 1), cd.embeddings);
    if(debug){
        cout << "### positional embeddings" << endl;
        pos_emb.print();
    }
 
    vector<float> cls_tokens_f(cd.embeddings, 0.0f);
    RowVector cls_token(cls_tokens_f.data(),cd.embeddings); // all zeros
    if(debug){
        cout << "### class token" << endl;
        cls_token.print();
    }
    
    // -Vision Transformer, I initialize an entire vit just to use the position embedding function
    VisionTransformer cpu_vit(
        cd.height, cd.width,
        cd.Ho, cd.Wo, cd.channels,
        100, pool_token, cd.embeddings,
        12, 2, 4, true, false, 1.0
    );  
    cpu_vit.move_cls_token(cls_token);
    cpu_vit.move_pos_embed(pos_emb);

    PictureBatch x(x_data, cd.batch * cd.channels * cd.height * cd.width, cd.batch, cd.channels, cd.height, cd.width);
    if(debug){
        cout << "### x" << endl;
        x.print();
    }

    Tensor t;
    pe.forward(x, t);
    if(debug){
        cout << "### t" << endl;
        t.print();
    }

    Tensor y;
    cpu_vit.position_embed(t,y); 
    if(debug){
        cout << "### y" << endl;
        y.print();
    }

    return y;
}

void cpu_gpu_comparison(bool debug){
    u_int batch = 8, channels = 3, height = 224, width = 224;
    u_int Ho = 16, Wo = 16, embeddings = 768;
    if(debug){
        batch = 8; channels = 3; height = 16; width = 16;
        Ho = 4; Wo = 4; embeddings = 10;  
    }
    convolution_dim conv_dim(batch,channels,height,width,embeddings,Ho,Wo);
    u_int tokens = conv_dim.y_height * conv_dim.y_width;
	
    cout << "X: [" << batch << ","<< channels << ","<< height << ","<< width << "]" << endl;
    cout << "W: [" << embeddings << ","<< channels << ","<< Ho << ","<< Wo << "]" << endl;
    cout << "Final embedded tensor: [" << batch << ","<< tokens + 1 << ","<< embeddings << "]" << endl;
    cout << "debug: " << yesno(debug) << endl;
    
    u_int input_pic_elements_num = batch * channels * height * width;
    u_int conv_kernel_elements_num = channels * embeddings * Ho * Wo;
    u_int embedded_elements_num = batch * (tokens + 1) * embeddings;
    
    // -Host memory allocation
    float * h_pic, * h_out;
    float * h_bias, * h_pos_emb, * h_conv_weights;
    h_pic = (float *)malloc(sizeof(float) * input_pic_elements_num);
    h_bias = (float *)malloc(sizeof(float) * embeddings);
    h_pos_emb = (float *)malloc(sizeof(float) * embeddings * (tokens + 1));
    h_conv_weights = (float *)malloc(sizeof(float) * conv_kernel_elements_num);

    h_out = (float *)malloc(sizeof(float) * embedded_elements_num);

    // -Random generation
    /*Host mem for gpu implementation should be pinned!*/
    half * gpu_pic, * gpu_out;
    half * gpu_bias, * gpu_pos_emb, * gpu_conv_weights;
    cudaHostAlloc(&gpu_pic,sizeof(half) * input_pic_elements_num, cudaHostAllocDefault);
    cudaHostAlloc(&gpu_out,sizeof(half) * embedded_elements_num, cudaHostAllocDefault);
    cudaHostAlloc(&gpu_bias,sizeof(half) * embeddings, cudaHostAllocDefault);
    cudaHostAlloc(&gpu_pos_emb,sizeof(half) * embeddings * (tokens + 1), cudaHostAllocDefault);
    cudaHostAlloc(&gpu_conv_weights,sizeof(half) * conv_kernel_elements_num, cudaHostAllocDefault);

    u_long seed = std::chrono::high_resolution_clock::now()
        .time_since_epoch()
        .count();
    rand_init(h_pic, input_pic_elements_num, 1.0f, seed);
    rand_init(h_bias, embeddings, 0.1f, seed);
    rand_init(h_pos_emb, (tokens + 1) * embeddings, 1.0f, seed);
    rand_init(h_conv_weights, conv_kernel_elements_num, 1.0f, seed);
    f32_to_f16(h_pic ,gpu_pic, input_pic_elements_num);
    f32_to_f16(h_bias,gpu_bias, embeddings);
    f32_to_f16(h_pos_emb ,gpu_pos_emb, (tokens + 1) * embeddings);
    f32_to_f16(h_conv_weights,gpu_conv_weights, conv_kernel_elements_num);

    // -- CPU reference -- 
    Tensor y_cpu = cpu_baseline(conv_dim, h_conv_weights, h_bias, h_pos_emb, h_pic, debug);
    
    // -- GPU Single Stream --
    // -CUDA variables and handlers creation
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));
    CUDNN_CHECK(cudnnSetStream(handle, stream));

    // -GpuPatchEmbedder creation
    GpuPatchEmbedder gpu_pe(
      stream,
      handle,
      conv_dim
    );

    // -Data loading and execution
    gpu_pe.load_weights_data(gpu_conv_weights, gpu_bias, gpu_pos_emb, false);
    gpu_pe.load_pics(gpu_pic);
    gpu_pe.forward(gpu_out, false, debug); //Copying on host, vit implementation doesn't need that
    cudaStreamSynchronize(stream);
    if(debug) {
        cout << "gpu_out:" << endl;
        f16_to_f32(gpu_out, h_out, embedded_elements_num);
        Tensor gpu_debug(h_out, embedded_elements_num, batch, tokens + 1, embeddings);
        gpu_debug.print();
    }
    cout << " Comparison CPU/GPU: " << compare_results(y_cpu, gpu_out) * 100 << "%" << endl;

    // -- GPU Multi Stream --

    // -Streams creation & initialization
    cudaStream_t streams[STREAM_N];
    cudnnHandle_t handles[STREAM_N];
    GpuPatchEmbedder sub_gpu_pe[STREAM_N];

    assert(batch % STREAM_N == 0);
    u_int minibatch = batch / STREAM_N; 
    conv_dim.batch = minibatch;
    cout << "minibatch:" << minibatch << endl;
    
    for(int i = 0; i< STREAM_N; i++){
        cudaStreamCreate(&streams[i]);

        CUDNN_CHECK(cudnnCreate(&handles[i]));
        CUDNN_CHECK(cudnnSetStream(handles[i], streams[i]));
        // -Multiple GpuPatchEmbedder instances creation 
        sub_gpu_pe[i] = GpuPatchEmbedder(
            streams[i],
            handles[i],
            conv_dim
        );

        // -Set the pointers to the weights data, testing weights sharing
        sub_gpu_pe[i].set_weights_data(gpu_pe.d_w, gpu_pe.d_bias, gpu_pe.d_pos_emb);

    }   

    // -Copy the respective input data and run asynconously one instance from another
    half * actual_pic = gpu_pic;
    half * actual_out = gpu_out;
    for(int i = 0; i< STREAM_N; i++){
        sub_gpu_pe[i].load_pics(actual_pic); //This should load a minibatch of pics
        sub_gpu_pe[i].forward(debug);
    
        actual_pic += minibatch * channels * height * width; //go to the next minibatch of images 
        actual_out += minibatch * (tokens + 1) * embeddings;
    }
    
    // -Synchronize all the streams
    for (size_t i = 0; i < STREAM_N; i++){
        cudaStreamSynchronize(streams[i]);
    }
    

    // -Compare the results
    cout << " Comparison CPU/GPU Streams: " << compare_results(y_cpu, gpu_out) * 100 << "%" << endl;

    // -Cleanup
    free(h_pic); free(h_bias); free(h_pos_emb); free(h_conv_weights); free(h_out);
    cudaFreeHost(gpu_pic); cudaFreeHost(gpu_out); cudaFreeHost(gpu_bias); cudaFreeHost(gpu_pos_emb); cudaFreeHost(gpu_conv_weights);
    sub_gpu_pe[0].free_weights(); /*Only one istance!*/

}

int main() {
    bool debug = false;
    cpu_gpu_comparison(debug);
    return 0;
}