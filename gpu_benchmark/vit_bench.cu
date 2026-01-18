#include "../gpu_include/gpu_vit.h"
#include "../include/vision_transformer.h"
#include "../gpu_include/bench_utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>
#include <cstring>
#include <cstdlib>

#define CLASS_N 100
#define DEPTH 12
#define HEADS_N 12
#define SCALE 1.0f
#define MLP_HIDDEN 3072
#define FULL_VIT_WARMUP 5
#define FULL_VIT_N 10

struct vit_predictions {
    half * predictions_probabilities;
    int total_samples_n;
    size_t total_probabilities_n;


    vit_predictions(int total_probabilities_n_, int total_samples_):
        total_samples_n(total_samples_),
        total_probabilities_n(total_probabilities_n_)
    {
        predictions_probabilities = (half *)malloc(sizeof(half) * total_probabilities_n_);
    }

    ~vit_predictions(){
        free(predictions_probabilities);
    }

    // `class_prediction` and initialized array of size `total_samples_n`
    void compute_predictions(int * class_prediction){
        float max_val;
        int idx;
        for (int b = 0; b < total_samples_n; b++){
            max_val = -1; //Prevoiusly have done a softmax so all values are [0,1]
            idx = 0;
            for(int cls = 0; cls < CLASS_N; cls++){
                float val = __half2float(predictions_probabilities[b * CLASS_N + cls]);
                if(val > max_val){
                    max_val = val;
                    idx = cls;
                }
            }
            class_prediction[b] = idx;
        }
    }   
};

struct vit_time{
    float class_setup_time;   //GPU
    float pics_load_time;       //GPU
    float forward_time;         //GPU
    float total_time;           //CPU

    vit_time(
        float class_setup_time_ = 0.0f,
        float pics_load_time_ = 0.0f,
        float forward_time_ = 0.0f,
        float total_time_ = 0.0f
    ):
        class_setup_time(class_setup_time_),
        pics_load_time(pics_load_time_),
        forward_time(forward_time_),
        total_time(total_time_)
    {}

    void print(){
        cout << "Class setup(shared & weights alloc + desc creation) : " << class_setup_time << " ms\n"
             << "Pics loading time (GPU)                             : " << pics_load_time << " ms\n"
             << "Forward time (GPU)                                  : " << forward_time << " ms\n"
             << "Total time (CPU)                                    : " << total_time << " ms\n";
    }

    void to_JSON(int streams_n, int batch_n, int minibatch_size, int batch_size, int params[]){
        int transpose_stride    = params[0];
        int add_stride          = params[1];
        int tokens_per_block    = params[2];
        int elements_per_thread = params[3];

        cout << "{\n"
            << "\"streams_n\": " << streams_n << ",\n"
            << "\"batch_n\": " << batch_n << ",\n"            
            << "\"batch_size\": " << batch_size << ",\n"
            << "\"minibatch_size\": " << minibatch_size << ",\n"
            << "\"params\": {\n"
                << "\"transpose_stride   \": " << transpose_stride    << ",\n"
                << "\"add_stride         \": " << add_stride          << ",\n"
                << "\"ln_tokens_per_block\": " << tokens_per_block    << ",\n"
                << "\"ln_elem_per_thread\": "  << elements_per_thread << "\n"
            << "},\n"
            << "\"time\": {\n"
                << "\"class_setup_time\":"   << class_setup_time << ",\n"
                << "\"pics_load_time\":"     << pics_load_time << ",\n"
                << "\"forward_time\":"       << forward_time << ",\n"
                << "\"total_time\":"         << total_time << "\n"
            << "}\n"
        << "}\n";
    }
};

vit_time vit_setup(    
    cudaStream_t stream,cublasLtHandle_t cublas_handle,cudnnHandle_t cudnn_handle,
    convolution_dim &conv_dim, int tokens, int tokens_per_block,
    patch_emb_weights pe_w,
    vector<block_weights> blk_w,
    pred_head_weights ph_w
){
    float avg_setup = time_cpu(WARM_UP, N, [&]() {
        // Create the model and init descriptors
        GpuVit gpu_vit(
            stream, cudnn_handle, cublas_handle,
            conv_dim, tokens, 
            CLASS_N, DEPTH, HEADS_N, SCALE, MLP_HIDDEN,
            false, 
            false, 
            true, 
            false
        );

        gpu_vit.tokens_per_block = tokens_per_block; 

        gpu_vit.allocate_shared_buffers(); //d_x, d_y, d_pred, d_t....
        gpu_vit.create_descriptors(); // all the descriptors
        gpu_vit.allocate_weights();
        gpu_vit.load_weights(pe_w, blk_w, ph_w);
        CUDA_CHECK(cudaStreamSynchronize(stream)); 
    });

    return vit_time(avg_setup);
}

vit_time vit_forward(
    cudaStream_t stream, cublasLtHandle_t cublas_handle, cudnnHandle_t cudnn_handle,
    convolution_dim &conv_dim, int tokens, int tokens_per_block,
    patch_emb_weights pe_w,
    vector<block_weights> blk_w,
    pred_head_weights ph_w,    
    half *gpu_pics, int * gpu_predictions
){
    GpuVit gpu_vit(
        stream, cudnn_handle, cublas_handle,
        conv_dim, tokens, 
        CLASS_N, DEPTH, HEADS_N, SCALE, MLP_HIDDEN,
        false, 
        false, 
        true, 
        false
    );

    gpu_vit.tokens_per_block = tokens_per_block; 

    gpu_vit.allocate_shared_buffers(); //d_x, d_y, d_pred, d_t....
    gpu_vit.create_descriptors(); // all the descriptors
    gpu_vit.allocate_weights();
    gpu_vit.load_weights(pe_w, blk_w, ph_w);
    
    float avg_pics_load = time_kernel(WARM_UP, N, stream, [&]() {
        gpu_vit.load_pics(gpu_pics);
    });

    gpu_vit.forward();
    gpu_vit.compute_predictions();
    for(int i = 0; i < conv_dim.batch; i++)
        gpu_predictions[i] = gpu_vit.ph.class_prediction[i];

    float avg_forward = time_kernel(WARM_UP, N, stream, [&]() {
        gpu_vit.forward();
    });

    return vit_time(0.0f, avg_pics_load, avg_forward);
}

// From the model creation to the final classification, memory efficient approach
vit_time full_vit_multi_prediction(
    cudaStream_t stream, cublasLtHandle_t cublas_handle, cudnnHandle_t cudnn_handle,
    int batch_n,
    convolution_dim &conv_dim, int tokens, int tokens_per_block,
    patch_emb_weights pe_w,
    vector<block_weights> blk_w,
    pred_head_weights ph_w,    
    half *gpu_pics, int * gpu_predictions
){
    size_t batch_elements_n = conv_dim.batch * conv_dim.height * conv_dim.width * conv_dim.channels;
    half * gpu_pics_start = gpu_pics;

    float total_time = time_cpu(FULL_VIT_WARMUP, FULL_VIT_N, [&]() {
        GpuVit gpu_vit(
            stream, cudnn_handle, cublas_handle,
            conv_dim, tokens, 
            CLASS_N, DEPTH, HEADS_N, SCALE, MLP_HIDDEN,
            false, 
            false, 
            true, 
            false
        );
        gpu_vit.tokens_per_block = tokens_per_block; 
        gpu_vit.ph.unmark_host_arr();        
        free(gpu_vit.ph.class_prediction);
        gpu_vit.ph.class_prediction = gpu_predictions;

        gpu_vit.allocate_shared_buffers(); //d_x, d_y, d_pred, d_t....
        gpu_vit.create_descriptors(); // all the descriptors
        gpu_vit.allocate_weights();
        gpu_vit.load_weights(pe_w, blk_w, ph_w);
        for(int iter = 0; iter < batch_n; iter++){
            gpu_vit.load_pics(gpu_pics_start);
            
            gpu_pics_start += batch_elements_n;
            gpu_vit.forward();
            gpu_vit.compute_predictions();
            gpu_vit.ph.class_prediction += conv_dim.batch;
        }
        gpu_pics_start = gpu_pics;
        free(gpu_vit.ph.gpu_x);
        free(gpu_vit.ph.h_x);
    });

    return vit_time(0.0f, 0.0f, 0.0f, total_time);
}

// From the model creation to the final classification, faster memory expensive approach
vit_time full_vit_single_prediction(
    cudaStream_t stream, cublasLtHandle_t cublas_handle, cudnnHandle_t cudnn_handle,
    int batch_n,
    convolution_dim &conv_dim, int tokens, int tokens_per_block,
    patch_emb_weights pe_w,
    vector<block_weights> blk_w,
    pred_head_weights ph_w,    
    half *gpu_pics, int * gpu_predictions
){
    size_t batch_elements_n = conv_dim.batch * conv_dim.height * conv_dim.width * conv_dim.channels;
    half * gpu_pics_start = gpu_pics;

    float total_time = time_cpu(FULL_VIT_WARMUP, FULL_VIT_N, [&]() {  
        GpuVit gpu_vit(
            stream, cudnn_handle, cublas_handle,
            conv_dim, tokens, 
            CLASS_N, DEPTH, HEADS_N, SCALE, MLP_HIDDEN,
            false, 
            false, 
            true, 
            false
        );
        gpu_vit.tokens_per_block = tokens_per_block; 

        vit_predictions predictions(batch_n * conv_dim.batch * CLASS_N, batch_n * conv_dim.batch);
        gpu_vit.ph.unmark_host_arr();        
        free(gpu_vit.ph.gpu_x);
        gpu_vit.ph.gpu_x = predictions.predictions_probabilities;
        gpu_vit.allocate_shared_buffers(); //d_x, d_y, d_pred, d_t....
        gpu_vit.create_descriptors(); // all the descriptors
        gpu_vit.allocate_weights();
        gpu_vit.load_weights(pe_w, blk_w, ph_w);
        for(int iter = 0; iter < batch_n; iter++){
            gpu_vit.load_pics(gpu_pics_start);
            
            gpu_pics_start += batch_elements_n;
            gpu_vit.forward();

            gpu_vit.ph.gpu_x += conv_dim.batch * CLASS_N;
        }
        CUDA_CHECK(cudaStreamSynchronize(stream)); //Maybe substitute with a for stream sync? 
        predictions.compute_predictions(gpu_predictions);
        gpu_pics_start = gpu_pics;

        free(gpu_vit.ph.class_prediction);
        free(gpu_vit.ph.h_x);
        //Not freeing gpu_x cause is already freed by vit_predictions dtor
    });

    return vit_time(0.0f, 0.0f, 0.0f, total_time);
}

vit_time multi_stream_full_vit(
    cudaStream_t stream[], cublasLtHandle_t cublas_handle[], cudnnHandle_t cudnn_handle[],
    int streams_n, int batch_n, int minibatch,
    convolution_dim &conv_dim, int tokens, int tokens_per_block,
    patch_emb_weights pe_w,
    vector<block_weights> blk_w,
    pred_head_weights ph_w,    
    half *gpu_pics, int *gpu_predictions
){
    assert(conv_dim.batch == minibatch);
    half * gpu_pics_start = gpu_pics;
    size_t minibatch_elements_n = conv_dim.batch * conv_dim.height * conv_dim.width * conv_dim.channels;
    
    float total_time = time_cpu(FULL_VIT_WARMUP, FULL_VIT_N, [&]() {  
        vit_predictions predictions(batch_n * conv_dim.batch * CLASS_N, batch_n * conv_dim.batch);
        vector<GpuVit> gpu_vit;
        gpu_vit.reserve(streams_n);
        for(int i = 0; i < streams_n; i++){
            gpu_vit.emplace_back(
                stream[i], cudnn_handle[i], cublas_handle[i],
                conv_dim, tokens, 
                CLASS_N, DEPTH, HEADS_N, SCALE, MLP_HIDDEN,
                false, 
                false, 
                true, 
                false
            );
            gpu_vit[i].tokens_per_block = tokens_per_block; 

            // -Independent buffers
            gpu_vit[i].allocate_shared_buffers(); 
            gpu_vit[i].create_descriptors();
            
            gpu_vit[i].ph.unmark_host_arr();
            free(gpu_vit[i].ph.gpu_x);
            // -Shared weights handling
            if(i == 0){
                gpu_vit[i].allocate_weights();
                gpu_vit[i].load_weights(pe_w, blk_w, ph_w);
            }
            else{
                // -Patch embedder handling
                gpu_vit[i].pe.set_weights_data(
                    gpu_vit[0].pe.d_w,
                    gpu_vit[0].pe.d_bias,
                    gpu_vit[0].pe.d_pos_emb
                );
                // -Blocks handling
                for(int block_idx = 0; block_idx < DEPTH; block_idx++){
                    gpu_vit[i].blocks[block_idx].set_weights_data(
                        gpu_vit[0].blocks[block_idx].d_n1_bias,
                        gpu_vit[0].blocks[block_idx].d_n1_scale,
                        gpu_vit[0].blocks[block_idx].d_n2_bias ,
                        gpu_vit[0].blocks[block_idx].d_n2_scale,
                        gpu_vit[0].blocks[block_idx].d_fc1     ,
                        gpu_vit[0].blocks[block_idx].d_b1_data ,
                        gpu_vit[0].blocks[block_idx].d_b1_mtx  ,
                        gpu_vit[0].blocks[block_idx].d_fc2     ,
                        gpu_vit[0].blocks[block_idx].d_b2_data ,
                        gpu_vit[0].blocks[block_idx].d_b2_mtx  ,
                        gpu_vit[0].blocks[block_idx].fused_desc.dWeights,
                        gpu_vit[0].blocks[block_idx].fused_desc.weightBytes
                    );
                }
                // -Prediction head handling
                gpu_vit[i].ph.set_shared_weights(
                    gpu_vit[0].ph.d_ln_scale    ,
                    gpu_vit[0].ph.d_ln_bias     ,
                    gpu_vit[0].ph.d_head_weights,
                    gpu_vit[0].ph.d_head_bias   
                );
            }
        }

        int vit_instance = 0;
        for(int mini = 0; mini <  batch_n ; mini++){
            gpu_vit[vit_instance].ph.gpu_x =
                predictions.predictions_probabilities + (mini * conv_dim.batch * CLASS_N);

            gpu_vit[vit_instance].load_pics(gpu_pics_start);
            gpu_pics_start += minibatch_elements_n;
            gpu_vit[vit_instance].forward();
            

            if(vit_instance == streams_n - 1)
                vit_instance = 0;
            else
                ++vit_instance;
        }
        CUDA_CHECK(cudaDeviceSynchronize()); //Maybe substitute with a for stream sync? 
        predictions.compute_predictions(gpu_predictions);  


        gpu_pics_start = gpu_pics;
        for(int i = 0; i < streams_n; i++){
            free(gpu_vit[i].ph.h_x);
            free(gpu_vit[i].ph.class_prediction);
        }        
    });

    return vit_time(0.0f, 0.0f, 0.0f, total_time);
} 

/*
Parameters:
- number of streams
- batch
- minibatch 
- patch embedder:
    transpose stride
    position embeddings add stride
- block:
    residual stride
- layer norm:
    tokens per block
    channels stride value
- mlp bias stride
*/

int main(int argc, char** argv){
    int kernel              = get_arg(argc, argv, "--kernel", 3);
    int streams_n           = get_arg(argc, argv, "--streams_n", 2);
    int batch_n             = get_arg(argc, argv, "--batch_n", 2);
    int batch               = get_arg(argc, argv, "--batch", 4);
    int minibatch           = get_arg(argc, argv, "--minibatch", 2);
    /*THESE 3 should be found in isolation in others bench*/
    int transpose_stride    = get_arg(argc, argv, "--transpose_stride", 2);
    int add_stride          = get_arg(argc, argv, "--add_stride", 2);
    int tokens_per_block    = batch < TOKENS_PER_BLOCK ? 1 : TOKENS_PER_BLOCK;
    if(kernel == 5)
        tokens_per_block    = minibatch < TOKENS_PER_BLOCK ? 1 : TOKENS_PER_BLOCK;
    int elements_per_thread = ELEMENTS_PER_TH;
    //----

    int channels = 3, height = 224, width = 224, Ho = 16, Wo = 16,
        embeddings = EMBEDDINGS_SIZE, tokens = (height / Ho) * (width / Wo);
    size_t total_samples = batch_n * batch;
    convolution_dim conv_dim(
        batch, channels, height, width,
        embeddings,
        Ho, Wo
    );

    cout << "Vit Benchmark\n"
        << " streams number:      " << streams_n << "\n"
        << " total samples:       " << total_samples << "\n"
        << " batch number:        " << batch_n << "\n"
        << " minibatch:           " << minibatch << "\n"        
        << " input 4D tensor NCHW:" << "["<< batch<< "," << channels << ","<< height<< "," << width << "]" << "\n"
        << " patch size:          " << "[" << Ho << "," << Wo << "]" << "\n"
        << " Embedded 3D tensor:  " << "["<< batch<< "," << tokens << ","<< embeddings << "]" << "\n"        
        << " tokens + cls token:  " << tokens + 1 << "\n"
        // << " block_dim            " << block_dim       << "\n"
        << " transpose_stride     " << transpose_stride<< "\n"
        << " add_stride           " << add_stride  << "\n"
        << " ln_tokens_per_block  " << tokens_per_block << "\n"
        << " ln elements_per_th   " << elements_per_thread  << "\n"
        << " warmup_iters:        " << WARM_UP << "\n"
        << " timed_iters:         " << N << "\n";

    
    // -  Memory allocation
    size_t total_elements_num = batch_n * batch * channels * height * width;
    size_t input_pic_elements_num   = batch * channels * height * width;
    size_t conv_kernel_elements_num = channels * embeddings * Ho * Wo;

    vector<float> h_input(total_elements_num);
    
    vector<half> gpu_input(total_elements_num);

    vector<int> gpu_predictions(total_samples);

    random_device rd;          
    mt19937 gen(rd());         
    uniform_real_distribution<float> dist(-1.0f, 1.0f);

    size_t loop_range = max(total_elements_num, conv_kernel_elements_num);
    for(size_t i = 0; i < loop_range; i++){
        if(i < total_elements_num){
            h_input[i] = dist(gen);
        }
    }

    f32_to_f16(h_input.data(), gpu_input.data(), total_elements_num);

    cudaStream_t streams[streams_n];
    cudnnHandle_t cudnn_handles[streams_n];
    cublasLtHandle_t cublaslt_handles[streams_n];    
    assert(batch % streams_n == 0);
    
    for(int i = 0; i< streams_n; i++){
        cudaStreamCreate(&streams[i]);

        CUDNN_CHECK(cudnnCreate(&cudnn_handles[i]));
        CUDNN_CHECK(cudnnSetStream(cudnn_handles[i], streams[i]));
        CUBLAS_CHECK(cublasLtCreate(&cublaslt_handles[i]));        
    }   

    // - Reference creation
    const string cvit_path = "./models/vit_1.cvit";
    VisionTransformer vit;
    load_cvit(cvit_path, vit);
    
    PredictionBatch pb_cpu[batch_n];
    vector<PictureBatch> pic; pic.reserve(batch_n);
    float * data_iter = h_input.data();
    for(int i = 0; i < batch_n; i++){
        pic.emplace_back(data_iter, input_pic_elements_num , batch, channels, height, width);
        // vit.forward(pic[i], pb_cpu[i]); // TO UNCOMMENT
        data_iter += input_pic_elements_num;
    }

    patch_emb_weights pe_w;
    vector<block_weights> blk_w;
    pred_head_weights ph_w;
    convert_vit_weights(vit, pe_w, blk_w, ph_w);

    if(kernel == 1){
        cout << "|| GpuVit Setup ||" << endl;
        vit_time avg_setup = vit_setup(
            streams[0], cublaslt_handles[0], cudnn_handles[0],
            conv_dim, tokens, tokens_per_block,
            pe_w, blk_w, ph_w
        );

        avg_setup.print();
        avg_setup.to_JSON(
            1, 1, batch, batch,
            new int[4]{transpose_stride, add_stride, tokens_per_block, elements_per_thread}
        );

    }
    if(kernel == 2){
        cout << "|| GpuVit Forward ||" << endl;
        vit_time avg_forward = vit_forward(
            streams[0], cublaslt_handles[0], cudnn_handles[0],
            conv_dim, tokens, tokens_per_block,
            pe_w, blk_w, ph_w,
            gpu_input.data(), gpu_predictions.data()
        );
        avg_forward.print();
        avg_forward.to_JSON(
            1, 1, batch, batch,
            new int[4]{transpose_stride, add_stride, tokens_per_block, elements_per_thread}
        );

        cout << "Single batch comparison with CPU: " << compare_predictions(pb_cpu[0], gpu_predictions.data()) * 100.0f<< "%" <<endl;
    }
    if(kernel == 3){
        cout << "|| GpuVit full multi predictions ||" << endl;
        vit_time avg_total_time = full_vit_multi_prediction(
            streams[0], cublaslt_handles[0], cudnn_handles[0],
            batch_n,
            conv_dim, tokens, tokens_per_block,
            pe_w, blk_w, ph_w,
            gpu_input.data(), gpu_predictions.data()
        );
        avg_total_time.print();
        avg_total_time.to_JSON(
            1, batch_n, batch, batch,
            new int[4]{transpose_stride, add_stride, tokens_per_block, elements_per_thread}
        );

        float avg_accuracy = 0.0f;
        int * prediction_iter = gpu_predictions.data();
        for(int i = 0; i < batch_n; i++){
            avg_accuracy += compare_predictions(pb_cpu[i], prediction_iter);
            prediction_iter += batch;
        }
        cout << "Full comparison with CPU: " << (avg_accuracy / batch_n) * 100 << "%" <<endl;

    }
    if(kernel == 4){
        cout << "|| GpuVit full single prediction ||" << endl;
        vit_time avg_total_time = full_vit_single_prediction(
            streams[0], cublaslt_handles[0], cudnn_handles[0],
            batch_n,
            conv_dim, tokens, tokens_per_block,
            pe_w, blk_w, ph_w,
            gpu_input.data(), gpu_predictions.data()
        );
        avg_total_time.print();
        avg_total_time.to_JSON(
            1, batch_n, batch, batch,
            new int[4]{transpose_stride, add_stride, tokens_per_block, elements_per_thread}
        );

        float avg_accuracy = 0.0f;
        int * prediction_iter = gpu_predictions.data();
        for(int i = 0; i < batch_n; i++){
            avg_accuracy += compare_predictions(pb_cpu[i], prediction_iter);
            prediction_iter += batch;
        }
        cout << "Full comparison with CPU: " << (avg_accuracy / batch_n) * 100 << "%" <<endl;
    }
    if(kernel == 5){
        cout << "|| GpuVit MultiStream ||" << endl;
        assert(batch % minibatch == 0);
        conv_dim.batch = minibatch;
        vit_time avg_total_time = multi_stream_full_vit(
            streams, cublaslt_handles, cudnn_handles,
            streams_n, batch_n, minibatch,
            conv_dim, tokens, tokens_per_block,
            pe_w, blk_w, ph_w,
            gpu_input.data(), gpu_predictions.data()
        );
        avg_total_time.print();
        avg_total_time.to_JSON(
            streams_n, batch_n, batch, minibatch,
            new int[4]{transpose_stride, add_stride, tokens_per_block, elements_per_thread}
        );

        float avg_accuracy = 0.0f;
        int * prediction_iter = gpu_predictions.data();
        for(int i = 0; i < batch_n; i++){
            avg_accuracy += compare_predictions(pb_cpu[i], prediction_iter);
            prediction_iter += batch;
        }
        cout << "Full comparison with CPU: " << (avg_accuracy / batch_n) * 100 << "%" <<endl;
    }
    if(kernel == 6){
        cout << "|| GpuVit MultiStream Pinned memory buffer ||" << endl;

    }

    // - Cleanup
    for(int i = 0; i< streams_n; i++){
        CUBLAS_CHECK(cublasLtDestroy(cublaslt_handles[i]));
        CUDNN_CHECK(cudnnDestroy(cudnn_handles[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }

    return 0;
}