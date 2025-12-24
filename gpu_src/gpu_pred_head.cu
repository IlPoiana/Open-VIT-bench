#include "../gpu_include/gpu_pred_head.h"

pred_head_weights::pred_head_weights():
    ln_scale(),
    ln_bias(),
    head_weights(),
    head_bias()
{}

pred_head_weights::pred_head_weights(
    half * _ln_scale,   
    half * _ln_bias,    
    half * _head_weights,
    half * _head_bias   
){
    ln_scale = _ln_scale;  
    ln_bias = _ln_bias;
    head_weights = _head_weights;
    head_bias = _head_bias;
}

void softmax_desc::destroy_descriptors(){
    cudnnDestroyTensorDescriptor(x_desc);
}

int argmax(vector<float>& vec, int begin, int end){
    if (vec.empty()) return -1;
    auto vec_begin = vec.begin() + begin;
    auto vec_end = vec.begin() + end;
    return std::distance(vec_begin, std::max_element(vec_begin, vec_end));
}

// `dimensions` will be [B,1,E,CLS_NUM,0]
void create_ph_desc(
    cublasLtHandle_t &cublas_handle,
    mlp_dimensions dimensions,
    cublasLt_matmul_desc &matmul, cublasLtMatmulAlgo_t &cublas_algo,
    softmax_desc &softmax,
    void * d_workspace
){
    u_int B = dimensions.B, C = dimensions.C ,K = dimensions.K;

    create_cublasLt_linlay_desc(
        B,1,C,K,
        matmul
    );

    cublas_algo = fetch_matmul_algos(cublas_handle, matmul, &d_workspace, false);

    cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
    cudnnDataType_t dtype = CUDNN_DATA_HALF;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&softmax.x_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        softmax.x_desc,
        format, dtype,
        B,
        K,
        1,
        1
    ));
    
}

//Initialize the object istance and descriptors, allocate unique pointers 
GpuPredictionHead::GpuPredictionHead(
    u_int batch_,
    u_int tokens_,
    u_int embeddings_,
    u_int class_num_,
    cudnnHandle_t &cudnn_handle_,
    cublasLtHandle_t &cublas_handle_,
    cudaStream_t &stream_
):
    batch(batch_) ,
    tokens(tokens_) ,
    embeddings(embeddings_),
    class_num(class_num_ ),
    cublas_handle(cublas_handle_),
    cudnn_handle(cudnn_handle_),
    stream(stream_),
    block_dim(embeddings_)
{
    probabilities_array = vector<float>(batch * class_num);
    class_prediction = vector<int>(batch);
    input_elements_number = batch * tokens * embeddings;
    
    gpu_x = (half *)malloc(sizeof(half) * input_elements_number);
    h_x = (float *)malloc(sizeof(float) * input_elements_number);
}

GpuPredictionHead::~GpuPredictionHead(){
    if(destroy_shared_buffers){
        cudaFree(d_x);
        cudaFree(d_t);
        cudaFree(d_y);
        cudaFree(d_pred);
        cudaFree(d_workspace);
    }
    if(destroy_shared_weights){
        free_weights();
    }
    free(gpu_x);
    free(h_x);
}

void GpuPredictionHead::mark_shared_buffers(){
    destroy_shared_buffers = true;
}

void GpuPredictionHead::mark_shared_weights(){
    destroy_shared_weights = true;
}

void GpuPredictionHead::free_weights(){
    cudaFree(d_ln_scale);
    cudaFree(d_ln_bias );
    cudaFree(d_head_weights);
    cudaFree(d_head_bias);
    destroy_shared_weights = false;
}

void GpuPredictionHead::init_descriptors(){
    mlp_dimensions dim(batch, 1, embeddings, class_num, 0);
    create_ph_desc(
        cublas_handle,
        dim,
        matmul,
        algo,
        softmax,
        d_workspace
    );
}

void GpuPredictionHead::destroy_descriptors(){
    matmul.destroy_descriptors();
    softmax.destroy_descriptors();
}


void GpuPredictionHead::allocate_weights(){
    CUDA_CHECK(cudaMallocAsync(&d_ln_scale,     sizeof(half) * embeddings               , stream));
    CUDA_CHECK(cudaMallocAsync(&d_ln_bias,      sizeof(half) * embeddings               ,stream));
    CUDA_CHECK(cudaMallocAsync(&d_head_weights, sizeof(half) * embeddings * class_num, stream));
    CUDA_CHECK(cudaMallocAsync(&d_head_bias,    sizeof(half) * class_num                , stream));
}

void GpuPredictionHead::load_weights(
    half * h_ln_scale_,   
    half * h_ln_bias_,    
    half * h_head_weights_,
    half * h_head_bias_  
){
    CUDA_CHECK(cudaMemcpyAsync(d_ln_scale , h_ln_scale_, sizeof(half) * embeddings , cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_ln_bias  , h_ln_bias_, sizeof(half) * embeddings , cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_head_weights, h_head_weights_, sizeof(half) * embeddings * class_num, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_head_bias , h_head_bias_, sizeof(half) * class_num, cudaMemcpyHostToDevice, stream));
}

void GpuPredictionHead::set_shared_weights(
    void * d_ln_scale_,   
    void * d_ln_bias_,    
    void * d_head_weights_,
    void * d_head_bias_  
){
    d_ln_scale    = d_ln_scale_;   
    d_ln_bias     = d_ln_bias_;    
    d_head_weights = d_head_weights_;
    d_head_bias   = d_head_bias_;
}

void GpuPredictionHead::set_shared_buffers(
    void * d_x_,        
    void * d_t_,        
    void * d_y_,        
    void * d_pred_,
    void * d_workspace_    
){
    d_x = d_x_;
    d_t =    d_t_;   
    d_y =    d_y_;   
    d_pred = d_pred_;
    assert(d_workspace == nullptr);
    d_workspace = d_workspace_;
}


void GpuPredictionHead::compute_predictions(){
    cudaStreamSynchronize(stream);
    f16_to_f32(gpu_x, probabilities_array.data(), batch * class_num);
    for(int i = 0; i < batch; i++){
        class_prediction[i] = argmax(probabilities_array, i * class_num, (i + 1) * class_num);
    }
}

void GpuPredictionHead::forward(bool debug){
    if(debug) tokens_per_block = 1;

    /*layer norm*/
    blocks_num = (input_elements_number / (block_dim * tokens_per_block));
    assert(input_elements_number % (block_dim * tokens_per_block) == 0);
    cub_single_layer_norm<<<blocks_num, block_dim, 0, stream>>>(
        (half*)d_x, (half*)d_x,
        (half*)d_ln_scale, (half*)d_ln_bias,
        __double2half(epsilon),
        tokens_per_block
    );
    if(debug){
        cout << "gpu layer norm:  " << endl;
        CUDA_CHECK(cudaMemcpy(gpu_x, d_x, sizeof(half) * input_elements_number, cudaMemcpyDeviceToHost));
        f16_to_f32(gpu_x,h_x, input_elements_number);
        Tensor tmp(h_x, input_elements_number, batch, tokens, embeddings);
        tmp.print();
    }

    /*pool*/
    half * pool_iterator = (half *)d_t;
    half * tokens_iterator = (half *)d_x;
    for(int i = 0; i < batch; i++){
        cudaMemcpyAsync(pool_iterator, tokens_iterator, sizeof(half) * embeddings, cudaMemcpyDeviceToDevice); //Using default stream
        pool_iterator += embeddings;
        tokens_iterator += embeddings * tokens;
    }
    if(debug){
        cout << "pool:  " << endl;
        CUDA_CHECK(cudaMemcpy(gpu_x, d_t, sizeof(half) * batch * embeddings, cudaMemcpyDeviceToHost));
        f16_to_f32(gpu_x,h_x, batch * embeddings);
        Tensor tmp(h_x, batch * embeddings, batch, 1, embeddings);
        tmp.print();
    }

    // /*matmul*/ CHECK THE WEIGHTS LOADING
    strided_linear_layer(
        cublas_handle,stream,
        batch,  1, class_num, stride_val,
        matmul, algo, d_workspace,
        d_t, d_head_weights, d_head_bias, d_y,
        false
    );

    if(debug){
        cout << "linear:  " << endl;
        CUDA_CHECK(cudaMemcpy(gpu_x, d_y, sizeof(half) * batch * class_num, cudaMemcpyDeviceToHost));
        f16_to_f32(gpu_x,h_x, batch * class_num);
        Tensor tmp(h_x, batch * class_num, batch, 1, class_num);
        tmp.print();
    }


    /*cuDNN softmax*/
    CUDNN_CHECK(
        cudnnSoftmaxForward(
            cudnn_handle,
            softmax.algo,
            softmax.mode,
            &alpha, softmax.x_desc, d_y, 
            &beta,softmax.x_desc, d_pred
        )
    )

    if(debug){
        cout << "softmax:  " << endl;
        CUDA_CHECK(cudaMemcpy(gpu_x, d_pred, sizeof(half) * batch * class_num, cudaMemcpyDeviceToHost));
        f16_to_f32(gpu_x,h_x, batch * class_num);
        Tensor tmp(h_x, batch * class_num, batch, 1, class_num);
        tmp.print();
    }

    /*Find the max of each elements of the batch*/
    CUDA_CHECK(cudaMemcpyAsync(gpu_x, d_pred, sizeof(half) * batch * class_num, cudaMemcpyDeviceToHost));
    compute_predictions();
}
