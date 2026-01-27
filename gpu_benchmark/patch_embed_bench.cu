#include "../gpu_include/gpu_patch_embedder.h"
#include "../include/vision_transformer.h"
#include "../gpu_include/bench_utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>
#include <cstring>
#include <cstdlib> 


struct patch_emb_time{
    float total;          
    float kernel;       
    float transpose;     
    float pos_embeddings;

    patch_emb_time(
        float _total,        
        float _kernel,       
        float _transpose,    
        float _pos_embeddings
    ):
        total(_total),        
        kernel(_kernel),       
        transpose(_transpose),    
        pos_embeddings(_pos_embeddings)
    {}

    void print(){
        cout << "   Total time (ms): " << total         << "ms" << endl;
        cout << "   kernel         : " << kernel        << "ms" << endl;
        cout << "   transpose      : " << transpose     << "ms" << endl;
        cout << "   pos_embeddings : " << pos_embeddings<< "ms" << endl;

    }

    void to_JSON(int batch, int params[]){
        int transpose_stride = params[0];
        int pos_emb_stride   = params[1];
        int block_dim        = params[2];

        cout << "{\n"
            << "\"batch\":" << batch << ",\n"
            << "\"params\": {\n" 
                << "\"transpose_stride\":" << transpose_stride << ",\n"
                << "\"block_dim\":" << block_dim << ",\n"
                << "\"pos_emb_stride\":" << pos_emb_stride << "\n"
            << "},\n"
            << "\"time\": {\n" 
                << "\"total\":" << total << ",\n"
                << "\"kernel\":" << kernel << ",\n"
                << "\"transpose\":" << transpose << ",\n"
                << "\"pos_embeddings\":" << pos_embeddings << "\n"
            << "}\n"
            << "}\n";
    }
};

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

    Conv2d c2d(cd.channels,cd.embeddings, cd.Ho, cd.Wo, cd.Ho, cd.Wo, true);
    c2d.move_kernel(k);
    c2d.move_bias(b);
    bool c2d_bias = true, strict_img_size = true, dynamic_img_pad = false, use_norm = false; 
    PatchEmbed pe(
        cd.height, cd.width, cd.Ho, cd.Wo, cd.channels, cd.embeddings,
        c2d_bias, strict_img_size, dynamic_img_pad, use_norm
    ); //use norm set to true ==> use_pre_norm = false
    pe.move_c2d(c2d);

    Matrix pos_emb(positional_embeddings, cd.embeddings * (tokens + 1), (tokens + 1), cd.embeddings);
    if(debug){
        cout << "### positional embeddings" << endl;
        pos_emb.print();
    }
    // float cls_t[8] = {1.0, 0.0, 2.0, -1.0, 0.0, 0.5, 0.7, 1.0}; 
    // RowVector cls_token(cls_t,embeddings);
    vector<float> cls_tokens_f(cd.embeddings, 0.0f);
    RowVector cls_token(cls_tokens_f.data(),cd.embeddings); // all zeros
    if(debug){
        cout << "### class token" << endl;
        cls_token.print();
    }

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

patch_emb_time full_gpu_pe(
    cudaStream_t & stream,  
    convolution_dim conv_dim,
    half * pics,
    half * conv_weights,
    half * bias,
    half * pos_emb,
    int block_dim, int transpose_stride, int pos_emb_stride,
    void * d_y
){
    cudnnHandle_t handle; CUDNN_CHECK(cudnnCreate(&handle));
    CUDNN_CHECK(cudnnSetStream(handle, stream));

    GpuPatchEmbedder pe(stream, handle, conv_dim); 
    pe.load_weights_data(conv_weights, bias, pos_emb, false);
    pe.load_pics(pics);
    pe.set_kernel_params(block_dim, transpose_stride, pos_emb_stride);
    
    int tokens = conv_dim.y_height * conv_dim.y_width;
    u_int embedded_elements_num = pe.batch * (tokens + 1) * pe.embeddings;
    
    // - Numerical Check
    // gpu patch embedder uses d_t without resetting it (only when defined) so the result have to be taken at the first iter
    pe.forward();
    CUDA_CHECK(cudaMemcpy(d_y, pe.d_x, sizeof(half) * embedded_elements_num, cudaMemcpyDeviceToDevice));
    // ----

    float total = time_kernel(WARM_UP, N, stream,[&]() {
        pe.forward();
    });
    
    float kernel = time_kernel(WARM_UP, N, stream,[&]() {
        execute_cudnn_conv2d_bias(pe.d_pic, pe.d_w, pe.d_out_pic, pe.d_bias, pe.conv_desc);
    });
    float transpose = time_kernel(WARM_UP, N, stream,[&]() {
        transpose_strided_tensor3d<<<pe.transpose_blocks_n, pe.block_dim, 0, stream>>>(
            (half *)pe.d_out_pic,
            (half *)pe.d_t,
            pe.batch,
            pe.embeddings,
            tokens
        );
    });
    float pos_embeddings = time_kernel(WARM_UP, N, stream,[&]() {
        add_pos_embeddings<<<pe.pos_emb_blocks_n, pe.block_dim, 0, stream>>>(
            (half *)pe.d_x,
            (half *)pe.d_pos_emb,
            embedded_elements_num,
            (pe.embeddings * (tokens + 1))
        );
    });

    pe.destroy_descriptors();
    pe.free_weights();
    CUDNN_CHECK(cudnnDestroy(handle));
    return patch_emb_time(total, kernel, transpose, pos_embeddings);
}

void single_run(
    cudaStream_t & stream,  
    convolution_dim conv_dim,
    half * pics,
    half * conv_weights,
    half * bias,
    half * pos_emb,
    int block_dim, int transpose_stride, int pos_emb_stride,
    void * d_y
){
    cudnnHandle_t handle; CUDNN_CHECK(cudnnCreate(&handle));
    CUDNN_CHECK(cudnnSetStream(handle, stream));

    GpuPatchEmbedder pe(stream, handle, conv_dim);
    pe.load_weights_data(conv_weights, bias, pos_emb, false);
    pe.load_pics(pics);
    pe.set_kernel_params(block_dim, transpose_stride, pos_emb_stride);
    
    int tokens = conv_dim.y_height * conv_dim.y_width;
    u_int embedded_elements_num = pe.batch * (tokens + 1) * pe.embeddings;
    
    pe.forward();
    CUDA_CHECK(cudaMemcpy(d_y, pe.d_x, sizeof(half) * embedded_elements_num, cudaMemcpyDeviceToDevice));

    pe.destroy_descriptors();
    pe.free_weights();
    CUDNN_CHECK(cudnnDestroy(handle));

}


int main(int argc, char** argv){
    int kernel              = get_arg(argc, argv, "--kernel", 1);
    int batch               = get_arg(argc, argv, "--batch", 32);
    int embeddings          = get_arg(argc, argv, "--embeddings", 768);
    int block_dim           = get_arg(argc, argv, "--block_dim", 256);
    int transpose_stride    = get_arg(argc, argv, "--transpose_stride", 4);
    int pos_emb_stride      = get_arg(argc, argv, "--pos_emb_stride", 4);
    bool cpu_comparison     = get_arg(argc, argv, "--cpu", 0);

    int channels = 3, height = 224, width = 224, Ho = 16, Wo = 16;
    convolution_dim conv_dim(batch, channels, height, width, embeddings, Ho, Wo);
    int tokens = conv_dim.y_height * conv_dim.y_width;
    cout << "PatchEmbedder Benchmark\n"
    << " input 4D tensor NCHW:" << "["<< batch<< "," << channels << ","<< height<< "," << width << "]" << "\n"
    << " patch size:          " << "[" << Ho << "," << Wo << "]" << "\n"
    << " batch_size:          " << batch << "\n"
    << " tokens:              " << tokens          << "\n"
    << " tokens + cls token:  " << tokens + 1      << "\n"
    << " embeddings:          " << embeddings      << "\n"
    << " block_dim            " << block_dim       << "\n"
    << " transpose_stride     " << transpose_stride<< "\n"
    << " pos_emb_stride       " << pos_emb_stride  << "\n"
    << " warmup_iters:        " << WARM_UP << "\n"
    << " timed_iters:         " << N << "\n";

    
    // -  Memory allocation
    size_t embedded_elements_num    = batch * (tokens + 1) * embeddings;
    size_t input_pic_elements_num   = batch * channels * height * width;
    size_t conv_kernel_elements_num = channels * embeddings * Ho * Wo;
    size_t pos_emb_n = embeddings * (tokens + 1);

    size_t bytes_embedded_elements_num    = sizeof(half) * embedded_elements_num   ;  

    vector<float> h_input       (input_pic_elements_num);
    vector<float> h_conv_weights(conv_kernel_elements_num);    
    vector<float> h_bias        (embeddings);
    vector<float> h_pos_emb     (pos_emb_n);
    
    vector<half> gpu_input       (input_pic_elements_num);
    vector<half> gpu_output      (embedded_elements_num);
    vector<half> gpu_conv_weights(conv_kernel_elements_num);    
    vector<half> gpu_bias        (embeddings);
    vector<half> gpu_pos_emb     (pos_emb_n);

    random_device rd;          
    mt19937 gen(rd());         
    uniform_real_distribution<float> dist(-1.0f, 1.0f);

    size_t loop_range = max(input_pic_elements_num, conv_kernel_elements_num);
    for(size_t i = 0; i < loop_range; i++){
        if(i < input_pic_elements_num){
            h_input[i] = dist(gen);
        }
        if(i < embeddings){
            h_bias[i] = dist(gen);
        }
        if(i < conv_kernel_elements_num){
            h_conv_weights[i] = dist(gen);
        }
        if(i < pos_emb_n){
            h_pos_emb[i] = dist(gen);
        }
    }

    f32_to_f16(h_input.data(), gpu_input.data(), input_pic_elements_num);
    f32_to_f16(h_conv_weights.data(), gpu_conv_weights.data(), conv_kernel_elements_num );
    f32_to_f16(h_bias.data(), gpu_bias.data(), embeddings);
    f32_to_f16(h_pos_emb.data(), gpu_pos_emb.data(), pos_emb_n);

    cudaStream_t stream;  CUDA_CHECK(cudaStreamCreate(&stream));

    void * d_y; CUDA_CHECK(cudaMalloc(&d_y, bytes_embedded_elements_num));
    // - Reference creation
    Tensor cpu_y;
    if(cpu_comparison){
        cpu_y = cpu_baseline(
            conv_dim,
            h_conv_weights.data(),
            h_bias.data(),
            h_pos_emb.data(),
            h_input.data(),
            false
        );
    }
    if(kernel == 0 || kernel == 1){
        cout << "|| Full times ||" << endl;
        patch_emb_time res_time = full_gpu_pe(
            stream,  
            conv_dim,
            gpu_input.data(),
            gpu_conv_weights.data(),
            gpu_bias.data(),
            gpu_pos_emb.data(),
            block_dim, transpose_stride, pos_emb_stride,
            d_y
        );
        if(cpu_comparison){
            CUDA_CHECK(cudaMemcpy(gpu_output.data(), d_y, bytes_embedded_elements_num, cudaMemcpyDeviceToHost));
            cout << "Last iteration comparison with CPU: " << compare_results(cpu_y, gpu_output.data()) * 100.0f<< "%" <<endl;
        }
        res_time.print();
        res_time.to_JSON(batch, new int[3]{transpose_stride, pos_emb_stride, block_dim});

    }
    if(kernel == 0 || kernel == 2){
        cout << "|| Single run ||" << endl;
        single_run(
            stream,  
            conv_dim,
            gpu_input.data(),
            gpu_conv_weights.data(),
            gpu_bias.data(),
            gpu_pos_emb.data(),
            block_dim, transpose_stride, pos_emb_stride,
            d_y
        );
        if(cpu_comparison){
            CUDA_CHECK(cudaMemcpy(gpu_output.data(), d_y, bytes_embedded_elements_num, cudaMemcpyDeviceToHost));
            cout << "Single run comparison with CPU: " << compare_results(cpu_y, gpu_output.data()) * 100.0f<< "%" <<endl;
        }
    }

    // - Cleanup
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return 0;
}


