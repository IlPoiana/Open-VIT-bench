#include "../gpu_include/gpu_proj_head.h"
#include "../include/vision_transformer.h"

u_long new_seed(){
    return std::chrono::high_resolution_clock::now()
        .time_since_epoch()
        .count();
}

void compare_results(PredictionBatch &cpu_pb, half * gpu_pred, vector<int> class_prediction, bool show_predictions = false){
    u_int batch = cpu_pb.get_B(), class_num = cpu_pb.get_CLS();
    vector<float> probabilities_array(batch * class_num);
    double avg = 0; int class_correctly_classified = 0;

    if(show_predictions){
        cout << "cpu predictions:" << endl;
        for(int i = 0; i < batch; i++){
            std::cout << "   B[" << i << "]: class " << cpu_pb.get_prediction_class(i) << ", prob ";
            printf("%7.3f\n", cpu_pb.get_prediction_class_probability(i));
        }
    }
    
    if(show_predictions) {cout << "gpu predictions:"<< endl;}
    f16_to_f32(gpu_pred, probabilities_array.data(), batch * class_num);
    for(int i = 0; i < batch; i++){
        class_prediction[i] = argmax(probabilities_array, i * class_num, (i + 1) * class_num);
        if(show_predictions) {cout << "B[" << i << "] : class " << class_prediction[i] << ", prob " << probabilities_array[i *class_num + class_prediction[i]] << endl;}
        
        if(class_prediction[i] == cpu_pb.get_prediction_class(i)) class_correctly_classified++;
        else{
            RowVector tmp(probabilities_array.data() + (i * class_num), class_num);
            cout << "wrong gpu class: "<< cpu_pb.get_prediction_class(i) << " - " << class_prediction[i] << endl;
            tmp.print();
        }
        avg += abs(probabilities_array[i *class_num + class_prediction[i]] - cpu_pb.get_prediction_class_probability(i)); 
    }
    cout << "gpu class prediction accuracy(cpu reference): " << ((float)class_correctly_classified / batch) * 100.0f << "%" << endl;
    cout << "average difference between cpu/GPU probabilities: " << avg / batch << endl;

    
}

void gpu_comparison(bool debug){
    u_int batch = 256, tokens = 197, embeddings = 768, class_num = 100;
    u_long seed = 0;
    if(debug){
        batch = 2; tokens = 16; embeddings = 16; class_num = 8;  
    }
    
    u_int input_elements_number = batch * tokens * embeddings;
    double epsilon = 1e-4;

    // Random generation
    float * h_x, * h_ln_scale, * h_ln_bias, * h_lin_w, * h_lin_bias;
    float * og_h_x;
    h_x        = (float *)malloc(sizeof(float) * input_elements_number);
    h_ln_scale = (float *)malloc(sizeof(float) * embeddings);
    h_ln_bias  = (float *)malloc(sizeof(float) * embeddings);
    h_lin_w    = (float *)malloc(sizeof(float) * embeddings * class_num);
    h_lin_bias = (float *)malloc(sizeof(float) * class_num);
    og_h_x     = (float *)malloc(sizeof(float) * input_elements_number);

    seed = new_seed();
    rand_init(h_x, input_elements_number, 0.1f, seed);
    seed = new_seed();
    rand_init(h_ln_scale, embeddings, 0.1f, seed);
    seed = new_seed();
    rand_init(h_ln_bias, embeddings, 0.1f, seed);
    seed = new_seed();
    rand_init(h_lin_w, embeddings * class_num, 0.1f, seed);
    seed = new_seed();
    rand_init(h_lin_bias, class_num, 0.1f, seed);
    

    half * gpu_x, * gpu_ln_scale, * gpu_ln_bias, * gpu_lin_w, * gpu_lin_bias;
    gpu_x = (half *)malloc(sizeof(half) * input_elements_number);
    gpu_ln_scale = (half *)malloc(sizeof(half) * embeddings);
    gpu_ln_bias  = (half *)malloc(sizeof(half) * embeddings);
    gpu_lin_w    = (half *)malloc(sizeof(half) * embeddings * class_num);
    gpu_lin_bias = (half *)malloc(sizeof(half) * class_num);
    f32_to_f16(h_x,gpu_x,input_elements_number);
    f32_to_f16(h_ln_scale, gpu_ln_scale, embeddings);
    f32_to_f16(h_ln_bias, gpu_ln_bias, embeddings);
    f32_to_f16(h_lin_w, gpu_lin_w, embeddings * class_num);
    f32_to_f16(h_lin_bias,gpu_lin_bias, class_num);
    for(int i = 0; i< input_elements_number; i++) {og_h_x[i] = h_x[i];}    

    void * d_x, * d_t, * d_y, * d_pred,* d_ln_scale, * d_ln_bias, * d_lin_w, * d_lin_bias;
    CUDA_CHECK(cudaMalloc(&d_x       , sizeof(half) * input_elements_number));
    CUDA_CHECK(cudaMalloc(&d_t       , sizeof(half) * batch * embeddings));
    CUDA_CHECK(cudaMalloc(&d_y       , sizeof(half) * batch * class_num));
    CUDA_CHECK(cudaMalloc(&d_pred    , sizeof(half) * batch * class_num));
    CUDA_CHECK(cudaMalloc(&d_ln_scale, sizeof(half) * embeddings ));
    CUDA_CHECK(cudaMalloc(&d_ln_bias , sizeof(half) * embeddings ));
    CUDA_CHECK(cudaMalloc(&d_lin_w   , sizeof(half) * embeddings * class_num));
    CUDA_CHECK(cudaMalloc(&d_lin_bias, sizeof(half) * class_num ));
    CUDA_CHECK(cudaMemcpy(d_x        , gpu_x        , sizeof(half) * input_elements_number, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ln_scale , gpu_ln_scale , sizeof(half) * embeddings, cudaMemcpyHostToDevice ));
    CUDA_CHECK(cudaMemcpy(d_ln_bias  , gpu_ln_bias  , sizeof(half) * embeddings, cudaMemcpyHostToDevice ));
    CUDA_CHECK(cudaMemcpy(d_lin_w    , gpu_lin_w    , sizeof(half) * embeddings * class_num, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lin_bias , gpu_lin_bias , sizeof(half) * class_num, cudaMemcpyHostToDevice));

    // CPU REFERENCE
    /*
    N = Class_num E = embeddings
    X [B,T,E] 
        -> LN [B,T,E]
            -> Pool [B,1,E]
                -> Linear [B,1,N] 
    */
    Tensor x(h_x, input_elements_number, batch, tokens, embeddings);
    Tensor head_in(batch, 1 , embeddings);
    Tensor head_out(batch,1, class_num);
    Matrix head_w(h_lin_w, embeddings * class_num, class_num, embeddings);
    RowVector ln_g(h_ln_scale, embeddings);
    RowVector ln_bias(h_ln_bias, embeddings);
    RowVector head_bias(h_lin_bias, class_num);

    if(debug){
        cout << "Layer Norm Bias" << endl;ln_bias.print();
        cout << "Layer Norm Scale" << endl;ln_g.print();
        cout << "Head matrix weights" << endl;head_w.print();
        cout << "Head bias" << endl;head_bias.print();
        cout << "x" << endl; x.print();
    }

    LayerNorm ln(embeddings, epsilon, true);
    ln.move_g(ln_g); ln.move_b(ln_bias);
    Linear head(embeddings, class_num, true);
    head.move_A(head_w); head.move_b(head_bias);

    ln(x);
    if(debug){
        cout << "After Ln: " << endl; x.print();
    }
    global_pool_nlc(x, head_in, pool_token, 1, true); //num_prefix_tokens = 1 (cls token)
    if(debug){
        cout << "Pool: " << endl; head_in.print();
    }
    head(head_in, head_out);
    if(debug){
        cout << "Head: " << endl; head_out.print();
    }

    PredictionBatch pb(head_out);
    if(debug){
        cout << "Prediction batch: " << endl; pb.print();
    }
    
    // GPU 
    cudaStream_t stream(0);
    cublasLtHandle_t cublas_handle;
    CUBLAS_CHECK(cublasLtCreate(&cublas_handle));
    cudnnHandle_t cudnn_handle;
    CUDNN_CHECK(cudnnCreate(&cudnn_handle));

    void * d_workspace; cudaMalloc(&d_workspace, (size_t) MLP_WORKSPACE_SIZE);

    // -- GPU class
    // f32_to_f16(og_h_x,gpu_x,input_elements_number);
    CUDA_CHECK(cudaMemcpy(d_x, gpu_x, sizeof(half) * input_elements_number, cudaMemcpyHostToDevice));

    GpuPredictionHead gpu_ph(
        batch,tokens, embeddings, class_num,
        cudnn_handle, cublas_handle,
        stream, d_workspace, false
    );

    gpu_ph.set_shared_buffers(d_x, d_t, d_y, d_pred);
    gpu_ph.set_shared_weights(d_ln_scale, d_ln_bias, d_lin_w, d_lin_bias);
    
    gpu_ph.tokens_per_block = debug ? 1 : 32;
    gpu_ph.stride_val = debug ? 1 : 32;

    gpu_ph.forward(debug);
    // cudaStreamSynch(...) wait until all the streams have finished

    gpu_ph.mark_shared_buffers(); //Using device buffers previously allocated
    gpu_ph.mark_shared_weights(); //Using precedentely allocated weights
    gpu_ph.destroy_descriptors();

    // -- COMPARISON --
    
    cout << "gpu class" << endl; 
    CUDA_CHECK(cudaMemcpyAsync(gpu_x, gpu_ph.d_pred, sizeof(half) * batch * class_num, cudaMemcpyDeviceToHost));
    
    compare_results(pb, gpu_x, gpu_ph.class_prediction);

}

int main() {
    bool debug = false;
    gpu_comparison(debug);

    return 0;
}