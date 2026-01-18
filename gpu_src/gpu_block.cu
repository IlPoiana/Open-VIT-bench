#include "../gpu_include/gpu_block.h"

block_weights::block_weights(
    half * _n1_bias , half * _n1_scale,
    half * _n2_bias , half * _n2_scale,
    half * _q , half * _k , half * _v , half * _p ,
    half * _qb, half * _kb, half * _vb, half * _pb,
    half * _fc1, half * _b1_data,
    half * _fc2, half * _b2_data
){
    n1_bias = _n1_bias ; n1_scale = _n1_scale;
    n2_bias = _n2_bias ; n2_scale = _n2_scale;
    attn_w.d_q  = _q ; attn_w.d_k  = _k ; attn_w.d_v  = _v ; attn_w.d_o  = _p ;
    attn_w.d_qb = _qb; attn_w.d_kb = _kb; attn_w.d_vb = _vb; attn_w.d_ob = _pb;
    fc1 = _fc1; b1_data = _b1_data;
    fc2 = _fc2; b2_data = _b2_data;
}

//One thread for each element
__global__ void residual_test(half * d_x, half * d_y, u_int N){
    u_int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(g_idx < N){ 
        d_y[g_idx] += d_x[g_idx];
    }
    return;
}

//More elements per thread, the scale is applied to the input(for now only in float)
__global__ void residual_strided(half * d_x, half * d_y, u_int N, float scale) {
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        float out = __half2float(d_y[i]);
        out += scale * __half2float (d_x[i]);
        d_y[i] = __float2half(out);
    }
    return;
}

//Strided version for apply a scale factor to an array, scale in float
__global__ void gpu_scale(half * d_x, half * d_y,u_int N, float scale){
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        d_y[i] = scale * __half2float(d_x[i]);
    }
    return;
}

GpuBlock& GpuBlock::operator=(GpuBlock&& other) noexcept{
    batch      = other.batch     ;
    tokens     = other.tokens    ;
    channels   = other.channels  ;
    k_channels = other.k_channels;

    kernel_type = other.kernel_type;
    scale     = other.scale    ; 
    epsilon   = other.epsilon  ; 
    num_heads = other.num_heads; 

    d_x = other.d_x;
    d_t = other.d_t;
    d_y = other.d_y;
    d_h = other.d_h;

    d_n1_bias  = other.d_n1_bias ; 
    d_n1_scale = other.d_n1_scale;
    d_n2_bias  = other.d_n2_bias ;
    d_n2_scale = other.d_n2_scale;

    d_fc1     = other.d_fc1    ;
    d_b1_data = other.d_b1_data;
    d_b1_mtx  = other.d_b1_mtx ;
    d_fc2     = other.d_fc2    ;
    d_b2_data = other.d_b2_data;
    d_b2_mtx  = other.d_b2_mtx ;

    mlp_alpha = other.mlp_alpha;
    mlp_beta = other.mlp_beta;

    rand_scale = other.rand_scale;
    input_elements_number  = other.input_elements_number ;
    hidden_elements_number = other.hidden_elements_number;
    destroy_shared_buffers = other.destroy_shared_buffers; 
    destroy_shared_weights = other.destroy_shared_weights; 

    set_descriptors(
        other.stream,
        other.ltHandle,
        other.cudnnHandle,
        other.fused_desc,
        other.matmul,
        other.algo,
        other.transposeDesc,
        other.d_workspace_mlp,
        other.mlp_out_desc,
        other.res_in_desc
    );

    return *this;
}

GpuBlock::GpuBlock(GpuBlock&& other) noexcept{
    batch      = other.batch     ;
    tokens     = other.tokens    ;
    channels   = other.channels  ;
    k_channels = other.k_channels;

    kernel_type = other.kernel_type;
    scale     = other.scale    ; 
    epsilon   = other.epsilon  ; 
    num_heads = other.num_heads; 

    d_x = other.d_x;
    d_t = other.d_t;
    d_y = other.d_y;
    d_h = other.d_h;

    d_n1_bias  = other.d_n1_bias ; 
    d_n1_scale = other.d_n1_scale;
    d_n2_bias  = other.d_n2_bias ;
    d_n2_scale = other.d_n2_scale;

    d_fc1     = other.d_fc1    ;
    d_b1_data = other.d_b1_data;
    d_b1_mtx  = other.d_b1_mtx ;
    d_fc2     = other.d_fc2    ;
    d_b2_data = other.d_b2_data;
    d_b2_mtx  = other.d_b2_mtx ;

    mlp_alpha = other.mlp_alpha;
    mlp_beta = other.mlp_beta;

    rand_scale = other.rand_scale;
    input_elements_number  = other.input_elements_number ;
    hidden_elements_number = other.hidden_elements_number;
    destroy_shared_buffers = other.destroy_shared_buffers; 
    destroy_shared_weights = other.destroy_shared_weights; 

    set_descriptors(
        other.stream,
        other.ltHandle,
        other.cudnnHandle,
        other.fused_desc,
        other.matmul,
        other.algo,
        other.transposeDesc,
        other.d_workspace_mlp,
        other.mlp_out_desc,
        other.res_in_desc
    );


    // - host allocation
    h_q  = (float *)malloc(sizeof(float) * channels * channels);
    h_k  = (float *)malloc(sizeof(float) * channels * channels);
    h_v  = (float *)malloc(sizeof(float) * channels * channels);
    h_p  = (float *)malloc(sizeof(float) * channels * channels);
    h_qb = (float *)malloc(sizeof(float) * channels);
    h_kb = (float *)malloc(sizeof(float) * channels);
    h_vb = (float *)malloc(sizeof(float) * channels);
    h_pb = (float *)malloc(sizeof(float) * channels);

    h_debug_out = (half*)malloc(sizeof(half) * input_elements_number);
}

GpuBlock::GpuBlock(
    cudaStream_t     &_stream,
    cudnnHandle_t    &_cudnn_handle,
    cublasLtHandle_t &_cublas_handle,
    u_int B_, u_int T_, u_int C_, u_int K_,
    bool kernel_type_,
    double epsilon_, float scale_, int num_heads_,
    bool initialize_descriptors,
    bool allocate_weights
): 
    stream(_stream),
    cudnnHandle(_cudnn_handle),
    ltHandle(_cublas_handle),
    batch(B_), tokens(T_), channels(C_), k_channels(K_),
    kernel_type(kernel_type_), 
    epsilon(epsilon_), scale(scale_), num_heads(num_heads_)
{
    // 0. Initialize all the descriptors
    assert(k_channels % num_heads == 0);
    if(initialize_descriptors){
        // 0.1 cuBLASLt MLP descriptors
        mlp_dimensions mdim(batch, tokens, channels, k_channels, channels);
        CUDA_CHECK(cudaMallocAsync(&d_workspace_mlp, (size_t)MLP_WORKSPACE_SIZE, stream));
        create_mlp_descriptors(ltHandle, matmul, d_workspace_mlp, algo, mdim, kernel_type);
    

        // 0.2 Optional transpose descriptors if kernel_type == true (like your code)
        if (kernel_type) {
            cublasOperation_t op = CUBLAS_OP_T;

            CUBLAS_CHECK(cublasLtMatrixTransformDescCreate(&transposeDesc, CUDA_R_32F));
            CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&mlp_out_desc, CUDA_R_16F, /*rows*/batch*tokens, /*cols*/channels, /*ld*/batch*tokens));
            CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&res_in_desc, CUDA_R_16F, /*rows*/channels, /*cols*/batch*tokens, /*ld*/channels));
            CUBLAS_CHECK(cublasLtMatrixTransformDescSetAttribute(
                transposeDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &op, sizeof(op)
            ));
        }
    }

    // 1. Allocate main activation buffers on device
    input_elements_number = batch * tokens * channels;
    hidden_elements_number = batch * tokens * k_channels;

    // 2. Attention variables
    h_q  = (float *)malloc(sizeof(float) * channels * channels);
    h_k  = (float *)malloc(sizeof(float) * channels * channels);
    h_v  = (float *)malloc(sizeof(float) * channels * channels);
    h_p  = (float *)malloc(sizeof(float) * channels * channels);
    h_qb = (float *)malloc(sizeof(float) * channels);
    h_kb = (float *)malloc(sizeof(float) * channels);
    h_vb = (float *)malloc(sizeof(float) * channels);
    h_pb = (float *)malloc(sizeof(float) * channels);

    // 3. LayerNorm params
    if(allocate_weights){
        CUDA_CHECK(cudaMallocAsync(&d_n1_bias,  sizeof(half)*channels, stream));
        CUDA_CHECK(cudaMallocAsync(&d_n1_scale, sizeof(half)*channels, stream));
        CUDA_CHECK(cudaMallocAsync(&d_n2_bias,  sizeof(half)*channels, stream));
        CUDA_CHECK(cudaMallocAsync(&d_n2_scale, sizeof(half)*channels, stream));
    }
    // 4. MLP weights/biases
    if(allocate_weights){

        size_t bytes_fc1 = sizeof(half)*k_channels*channels;
        size_t bytes_fc2 = sizeof(half)*channels*k_channels;
        size_t bytes_b1  = sizeof(half)*k_channels;
        size_t bytes_b2  = sizeof(half)*channels;
        size_t bytes_b1_mtx = sizeof(half)*hidden_elements_number;
        size_t bytes_b2_mtx = sizeof(half)*input_elements_number;

        CUDA_CHECK(cudaMallocAsync(&d_fc1,     bytes_fc1, stream));
        CUDA_CHECK(cudaMallocAsync(&d_b1_data, bytes_b1 , stream));
        CUDA_CHECK(cudaMallocAsync(&d_fc2,     bytes_fc2, stream));
        CUDA_CHECK(cudaMallocAsync(&d_b2_data, bytes_b2 , stream));
        if(kernel_type) {
            CUDA_CHECK(cudaMallocAsync(&d_b1_mtx,  bytes_b1_mtx, stream));
            CUDA_CHECK(cudaMallocAsync(&d_b2_mtx,  bytes_b2_mtx, stream));
        }
    }

    // 5. host debug buffer for pulling results back
    h_debug_out = (half*)malloc(sizeof(half) * input_elements_number);

}

// ---- dtor ----
GpuBlock::~GpuBlock() {
    // cout << "destructor called!" << endl;
    if(destroy_shared_buffers){
        free_buffers();
    }
    
    if(destroy_shared_weights){
        free_weights();
    }
    free_host_buffers();
}

void GpuBlock::free_host_buffers(){
    free(h_q);
    free(h_k);
    free(h_v);
    free(h_p);
    free(h_qb);
    free(h_kb);
    free(h_vb);
    free(h_pb);

    // free host scratch
    free(h_debug_out);
}

void GpuBlock::init_attn_descriptor(bool load_attn_weights){
    vector<half> 
        q_half(channels*channels,0),
        k_half(channels*channels,0),
        v_half(channels*channels,0),
        p_half(channels*channels,0);
    mtx 
        qb_half(h_qb,1,channels),
        kb_half(h_kb,1,channels),
        vb_half(h_vb,1,channels),
        pb_half(h_pb,1,channels);

    transposeHostF32toHalf(h_q,q_half); // Missing implementation!
    transposeHostF32toHalf(h_k,k_half);
    transposeHostF32toHalf(h_v,v_half);
    transposeHostF32toHalf(h_p,p_half);


    //cuDNN Attention descriptors
    attn_dimensions_gpu adim(batch,tokens,channels,channels);
    fused_desc.hiWin = std::vector<int>(tokens, 0);

    attn_data_gpu h_attn_weights(
        q_half.data(), k_half.data(), v_half.data(), p_half.data(),
        qb_half.data, kb_half.data, vb_half.data, pb_half.data
    );

    initialize_attn_descriptors(
        cudnnHandle,
        h_attn_weights,
        adim,
        fused_desc,
        num_heads,
        load_attn_weights
    );
}

    //Initialize the attention descriptor given the weight matrices
void GpuBlock::init_attn_descriptor(
    float * h_q_, float * h_k_, float * h_v_, float * h_p_,
    float * h_qb_, float * h_kb_, float * h_vb_, float * h_pb_
){

    h_q = h_q_;
    h_k = h_k_;
    h_v = h_v_;
    h_p = h_p_;
    h_qb= h_qb_;
    h_kb= h_kb_;
    h_vb= h_vb_;
    h_pb= h_pb_;

    this->init_attn_descriptor();
}

void GpuBlock::init_descriptors(){
    // -Attention
    attn_dimensions_gpu adim(batch,tokens,channels,channels);
    fused_desc.hiWin = std::vector<int>(tokens, 0);
    attn_data_gpu<half> h_attn_weights; //not used in this function

    initialize_attn_descriptors(
        cudnnHandle,
        h_attn_weights,
        adim,
        fused_desc,
        num_heads,
        false
    );

    // - MLP 
    mlp_dimensions mdim(batch, tokens, channels, k_channels, channels);
    if(d_workspace_mlp == nullptr)
        CUDA_CHECK(cudaMallocAsync(&d_workspace_mlp, (size_t)MLP_WORKSPACE_SIZE, stream));
    create_mlp_descriptors(ltHandle, matmul, d_workspace_mlp, algo, mdim, kernel_type);


    // - Optional transpose descriptors if kernel_type == true
    if (kernel_type) {
        cublasOperation_t op = CUBLAS_OP_T;

        CUBLAS_CHECK(cublasLtMatrixTransformDescCreate(&transposeDesc, CUDA_R_32F));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&mlp_out_desc, CUDA_R_16F, /*rows*/batch*tokens, /*cols*/channels, /*ld*/batch*tokens));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&res_in_desc, CUDA_R_16F, /*rows*/channels, /*cols*/batch*tokens, /*ld*/channels));
        CUBLAS_CHECK(cublasLtMatrixTransformDescSetAttribute(
            transposeDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &op, sizeof(op)
        ));
    }
}

void GpuBlock::destroy_descriptors(bool weights, bool workspace){
    matmul[0].destroy_descriptors();
    matmul[1].destroy_descriptors();
    fused_desc.destroy_descriptors(weights, workspace);
}



void GpuBlock::allocate_weights(){
    size_t bytes_fc1 = sizeof(half)*k_channels*channels;        
    size_t bytes_fc2 = sizeof(half)*channels*k_channels;        
    size_t bytes_b1  = sizeof(half)*k_channels;                 
    size_t bytes_b2  = sizeof(half)*channels;                   
    size_t bytes_b1_mtx = sizeof(half)*hidden_elements_number;
    size_t bytes_b2_mtx = sizeof(half)*input_elements_number;

    cudaMallocAsync(&d_n1_bias ,sizeof(half) *  channels, stream);
    cudaMallocAsync(&d_n1_scale,sizeof(half) *  channels, stream);
    cudaMallocAsync(&d_n2_bias ,sizeof(half) *  channels, stream);
    cudaMallocAsync(&d_n2_scale,sizeof(half) *  channels, stream);
    cudaMallocAsync(&d_fc1     ,bytes_fc1, stream);
    cudaMallocAsync(&d_b1_data ,bytes_b1, stream);
    cudaMallocAsync(&d_fc2     ,bytes_fc2, stream);
    cudaMallocAsync(&d_b2_data ,bytes_b2, stream);
    if(kernel_type){
        cudaMallocAsync(&d_b1_mtx, bytes_b1_mtx, stream);
        cudaMallocAsync(&d_b2_mtx, bytes_b2_mtx, stream);
    }

    allocate_attn_weights(cudnnHandle, stream, fused_desc);
}

void GpuBlock::load_weights(
    half * _n1b_data, half * _n1g_data,
    half * _n2b_data, half * _n2g_data,
    half * _A1_data, half * _b1_data,
    half * _A2_data, half * _b2_data,
    attn_data_gpu<half> attn_w
){
    //Memcpy all the device vectors (layer norm and mlp)
    CUDA_CHECK(cudaMemcpyAsync(d_n1_scale, _n1g_data, sizeof(half) * channels,cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_n1_bias , _n1b_data, sizeof(half) * channels,cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_n2_scale, _n2g_data, sizeof(half) * channels,cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_n2_bias , _n2b_data, sizeof(half) * channels,cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaMemcpyAsync(d_fc1     , _A1_data,sizeof(half) * channels * k_channels, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_fc2     , _A2_data,sizeof(half) * channels * k_channels, cudaMemcpyHostToDevice, stream));
    
    if(kernel_type){
        vector<half> b1_half_mtx(hidden_elements_number);
        vector<half> b2_half_mtx(input_elements_number);

        bias_matrix(_b1_data,b1_half_mtx.data(),k_channels, batch * tokens);
        bias_matrix(_b2_data,b2_half_mtx.data(),channels, batch * tokens);

        CUDA_CHECK(cudaMemcpyAsync(d_b1_mtx , b1_half_mtx.data(),sizeof(half) * hidden_elements_number, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_b2_mtx , b2_half_mtx.data(),sizeof(half) * input_elements_number, cudaMemcpyHostToDevice, stream));
    }
    else{
        CUDA_CHECK(cudaMemcpyAsync(d_b1_data , _b1_data,sizeof(half) * k_channels, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_b2_data , _b2_data,sizeof(half) * channels, cudaMemcpyHostToDevice, stream));
    }

    attn_dimensions_gpu attn_dim(batch,tokens,channels, channels);

    load_attn_weights(
        cudnnHandle,
        stream,
        attn_w,
        attn_dim,
        fused_desc
    );
}



void GpuBlock::random_data(bool attn_init, bool input){
    cout << "ln" << endl;
    //Layer norm
    populate_rand(d_n1_bias ,channels);
    populate_rand(d_n1_scale,channels); //TO CHECK channels
    populate_rand(d_n2_bias ,channels);
    populate_rand(d_n2_scale,channels);
    cout << "mlp" << endl;
    //Mlp
    populate_rand(d_fc1    , channels * k_channels);
    populate_rand(d_b1_data, k_channels);
    populate_rand(d_fc2    , k_channels * channels);
    populate_rand(d_b2_data, channels);
    
    if(kernel_type){
        vector<half> h_b1(k_channels);
        vector<half> h_b2(channels);
        vector<half> h_b1_mtx(hidden_elements_number);
        vector<half> h_b2_mtx(input_elements_number);

        CUDA_CHECK(cudaMemcpy(h_b1.data(),d_b1_data, sizeof(half) * k_channels, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_b2.data(),d_b2_data, sizeof(half) * channels, cudaMemcpyDeviceToHost));

        bias_matrix(h_b1.data(), h_b1_mtx.data(), k_channels, batch * tokens);
        bias_matrix(h_b2.data(), h_b2_mtx.data(), channels, batch * tokens);

        CUDA_CHECK(cudaMemcpy(d_b1_mtx,h_b1_mtx.data(), sizeof(half) * hidden_elements_number, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b2_mtx,h_b2_mtx.data(), sizeof(half) * input_elements_number, cudaMemcpyHostToDevice));

        // populate_rand(d_b1_mtx , hidden_elements_number);
        // populate_rand(d_b2_mtx , input_elements_number);
    }

    cout << "attn" << endl;
    //Attention
    if(!attn_init){
        populate_rand(h_q, channels * channels);
        populate_rand(h_k, channels * channels);
        populate_rand(h_v, channels * channels);
        populate_rand(h_p, channels * channels);
        populate_rand(h_qb, channels);
        populate_rand(h_kb, channels);
        populate_rand(h_vb, channels);
        populate_rand(h_pb, channels);
        this->init_attn_descriptor();
    }
    if(input){
        cout << "input" << endl;
        //Input data
        populate_rand(d_x, input_elements_number);
    }
}

void GpuBlock::forward(bool debug, u_int tokens_per_block){
    u_int ln_blocks_n = (batch * tokens) / tokens_per_block;
    assert(((batch * tokens) % tokens_per_block) == 0);

    //Variables init
    half gpu_epsilon = __float2half(epsilon);

    //-Layer Norm    
    cub_single_layer_norm<<<ln_blocks_n,channels,0,stream>>>((half *)d_x, (half *)d_y,(half *)d_n1_scale, (half *)d_n1_bias, gpu_epsilon, tokens_per_block);
    
    /*TO REMOVE*/
    // {
    //     cudaMemcpyAsync(h_debug_out, d_n2_scale, sizeof(half) * channels,cudaMemcpyDeviceToHost,stream);
    //     cudaStreamSynchronize(stream);
    //     float * f_debug = (float *)calloc(channels,sizeof(float));
    //     f16_to_f32(h_debug_out, f_debug, channels);
    //     RowVector y(f_debug, channels);
    //     cout << "ln2 scale after ln1" << endl; y.print();
    //     cudaMemcpyAsync(h_debug_out, d_n2_bias, sizeof(half) * channels,cudaMemcpyDeviceToHost,stream);
    //     cudaStreamSynchronize(stream);
    //     f16_to_f32(h_debug_out, f_debug, channels);
    //     RowVector y2(f_debug, channels);
    //     cout << "ln2 bias after ln1" << endl; y2.print();
    // }//----

    if(debug) {cout << "ln1" << endl; print_debug();}
    //-Attention
    attention_device( 
        cudnnHandle,
        d_y, d_t,
        fused_desc
    );
     /*TO REMOVE
    // cout << "ELEMENTS CHECK" << endl;
    // cout << input_elements_number << endl;
    // cout << "d_x "<< d_x << endl;
    // cout << "d_t "<< d_t << endl;
    // cout << "d_h "<< d_h << endl;
    // cout << "d_y "<< d_y << endl;
    */
    if(debug) {cout << "attn" << endl; print_debug();}

    /*TO REMOVE*/
    // {
    //     cudaMemcpyAsync(h_debug_out, d_n2_scale, sizeof(half) * channels,cudaMemcpyDeviceToHost,stream);
    //     cudaStreamSynchronize(stream);
    //     float * f_debug = (float *)calloc(channels,sizeof(float));
    //     f16_to_f32(h_debug_out, f_debug, channels);
    //     RowVector y(f_debug, channels);
    //     cout << "ln2 scale after attn" << endl; y.print();
    //     cudaMemcpyAsync(h_debug_out, d_n2_bias, sizeof(half) * channels,cudaMemcpyDeviceToHost,stream);
    //     cudaStreamSynchronize(stream);
    //     f16_to_f32(h_debug_out, f_debug, channels);
    //     RowVector y2(f_debug, channels);
    //     cout << "ln2 bias after attn" << endl; y2.print();
    // }//----

    //-Residual
    residual_strided<<<tokens,channels,0,stream>>>((half*)d_t,(half*)d_x, input_elements_number, scale);
    if(debug) {cout << "residual1" << endl; print_debug();}

    //-Layer Norm
    cub_single_layer_norm<<<ln_blocks_n,channels,0,stream>>>((half *)d_x, (half *)d_y, (half *)d_n2_scale, (half *)d_n2_bias, gpu_epsilon, tokens_per_block);
    if(debug) {cout << "ln2" << endl; print_debug();}

    //-MLP
    if(kernel_type)
    {
        fused_gpu_mlp(
            ltHandle,stream,
            matmul, algo, d_workspace_mlp,
            d_y, d_fc1, d_h, d_b1_mtx, d_fc2, d_b2_mtx,d_t
        );

        //Transpose
        cublasLtMatrixTransform(
            ltHandle, transposeDesc,
            &mlp_alpha, d_t, mlp_out_desc,
            &mlp_beta, nullptr, nullptr,
            d_y, res_in_desc, stream
        );
        if(debug) {cout << "fused_mlp" << endl; print_debug();}

        //-Residual
        /*Toy inefficient example to see residual striding work, should be B elements per thread*/
        residual_strided<<<tokens,channels,0,stream>>>((half*)d_y,(half*)d_x,input_elements_number, scale);
        if(debug) {cout << "residual 2" << endl; print_debug();}

    }
    else{ //not fused but without the transpose
        gpu_mlp(
            ltHandle,stream,
            batch,tokens,k_channels,channels,
            matmul, algo, d_workspace_mlp,
            d_y, d_fc1, d_h,d_b1_data, d_fc2,d_b2_data,d_t
        );
        if(debug) {cout << "gpu_mlp" << endl; print_debug();}

        //-Residual
        /*Toy inefficient example to see residual striding work, should be B elements per thread*/
        residual_strided<<<tokens,channels,0,stream>>>((half*)d_t,(half*)d_x,input_elements_number, scale);
        if(debug) {cout << "residual 2" << endl; print_debug();}

    }

    if(debug){
        CUDA_CHECK(cudaMemcpy(h_debug_out,d_x, sizeof(half) * input_elements_number, cudaMemcpyDeviceToHost));
        print_h_out();
    }

}


void GpuBlock::forward(float * h_x, bool debug, u_int tokens_per_block){
    u_int input_elements_dim = batch*tokens*channels;
    
    // upload x_data -> d_x as half
    std::vector<half> x_half(input_elements_dim);
    f32_to_f16(h_x, x_half.data(), input_elements_dim);
    CUDA_CHECK(cudaMemcpy(d_x, x_half.data(), input_elements_dim * sizeof(half), cudaMemcpyHostToDevice));

    forward(debug, tokens_per_block);
}

void GpuBlock::forward(half * h_x, bool debug, u_int tokens_per_block){
    u_int input_elements_dim = batch*tokens*channels;

    CUDA_CHECK(cudaMemcpy(d_x, h_x, input_elements_dim * sizeof(half), cudaMemcpyHostToDevice));

    forward(debug, tokens_per_block);
}

// ---- Setters ----

void GpuBlock::set_buffers(
    void * _d_x,      
    void * _d_t,      
    void * _d_y,      
    void * _d_h,  
    void * _d_workspace_mlp  
){
    d_x = _d_x;
    d_t = _d_t;
    d_y = _d_y;
    d_h = _d_h;
    assert(d_workspace_mlp == nullptr);
    d_workspace_mlp = _d_workspace_mlp;
    fused_desc.dWork = _d_workspace_mlp;
    fused_desc.workBytes = WORKSPACE_SIZE;
}

/*
Copy on the host device the block weights, converting to half
*/
void GpuBlock::set_data(
    float* n1b_data,
    float* n1g_data,
    float* n2b_data,
    float* n2g_data,

    float* q_data,
    float* k_data,
    float* v_data,
    float* p_data,   // O proj

    float* qb_data,
    float* kb_data,
    float* vb_data,
    float* pb_data,

    float* A1_data,  // fc1 weights KxC
    float* b1_data,  // fc1 bias   K
    float* A2_data,  // fc2 weights MxK
    float* b2_data   // fc2 bias   M
)
{
    vector<half> n1g_half(channels);
    vector<half> n1b_half(channels);
    vector<half> n2g_half(channels);
    vector<half> n2b_half(channels);

    vector<half> A1_half(channels * k_channels);
    vector<half> b1_half(k_channels);
    vector<half> A2_half(channels * k_channels);
    vector<half> b2_half(channels);

    f32_to_f16(n1g_data, n1g_half.data(), channels);
    f32_to_f16(n1b_data, n1b_half.data(), channels);
    f32_to_f16(n2g_data, n2g_half.data(), channels);
    f32_to_f16(n2b_data, n2b_half.data(), channels);

    f32_to_f16(A1_data, A1_half.data(), channels * k_channels);
    f32_to_f16(b1_data, b1_half.data(), k_channels);
    f32_to_f16(A2_data, A2_half.data(), channels * k_channels);
    f32_to_f16(b2_data, b2_half.data(), channels);

    //Memcpy all the device vectors (layer norm and mlp)
    CUDA_CHECK(cudaMemcpy(d_n1_scale, n1g_half.data(), sizeof(half) * channels,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_n1_bias , n1b_half.data(), sizeof(half) * channels,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_n2_scale, n2g_half.data(), sizeof(half) * channels,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_n2_bias , n2b_half.data(), sizeof(half) * channels,cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_fc1     , A1_half.data(),sizeof(half) * channels * k_channels, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fc2     , A2_half.data(),sizeof(half) * channels * k_channels, cudaMemcpyHostToDevice));
    
    if(kernel_type){
        vector<half> b1_half_mtx(hidden_elements_number);
        vector<half> b2_half_mtx(input_elements_number);

        bias_matrix(b1_half.data(),b1_half_mtx.data(),k_channels, batch * tokens);
        bias_matrix(b2_half.data(),b2_half_mtx.data(),channels, batch * tokens);

        CUDA_CHECK(cudaMemcpy(d_b1_mtx , b1_half_mtx.data(),sizeof(half) * hidden_elements_number, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b2_mtx , b2_half_mtx.data(),sizeof(half) * input_elements_number, cudaMemcpyHostToDevice));
    }
    else{
        CUDA_CHECK(cudaMemcpy(d_b1_data , b1_half.data(),sizeof(half) * k_channels, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b2_data , b2_half.data(),sizeof(half) * channels, cudaMemcpyHostToDevice));
    }

    //Deepcopy all the host vectors (attention)
    for(u_int i = 0; i < channels * channels; i++){
        h_q [i] =  q_data[i]; 
        h_k [i] =  k_data[i]; 
        h_v [i] =  v_data[i]; 
        h_p [i] =  p_data[i]; 
        if(i < channels){
            h_qb[i] =  qb_data[i];
            h_kb[i] =  kb_data[i];
            h_vb[i] =  vb_data[i];
            h_pb[i] =  pb_data[i];
        }
    }
}

void GpuBlock::set_weights_data(
    void* d_n1_bias_, void* d_n1_scale_, void* d_n2_bias_, void* d_n2_scale_, 
    void* d_fc1_    , void* d_b1_data_, void* d_b1_mtx_ ,
    void* d_fc2_    , void* d_b2_data_, void* d_b2_mtx_ ,
    void* d_attn_weights, size_t weights_bytes
){
    d_n1_bias  = d_n1_bias_ ;
    d_n1_scale = d_n1_scale_;
    d_n2_bias  = d_n2_bias_ ;
    d_n2_scale = d_n2_scale_;
    d_fc1      = d_fc1_     ;
    d_b1_data  = d_b1_data_ ;
    d_b1_mtx   = d_b1_mtx_  ;
    d_fc2      = d_fc2_     ;
    d_b2_data  = d_b2_data_ ;
    d_b2_mtx   = d_b2_mtx_  ;
    fused_desc.dWeights = d_attn_weights;
    fused_desc.weightBytes = weights_bytes;
}

// Replace matmul descriptors (e.g. autotuned algos)
void GpuBlock::set_matmul_descriptors(
    const cublasLt_matmul_desc newMatmul[2],
    const cublasLtMatmulAlgo_t newAlgo[2],
    void* newWorkspace
){
    matmul[0] = newMatmul[0];
    matmul[1] = newMatmul[1];
    algo[0]   = newAlgo[0];
    algo[1]   = newAlgo[1];
    d_workspace_mlp = newWorkspace;
}

void GpuBlock::set_rand_scale(float _scale){
    rand_scale = _scale;
}

//Call this method to destroy the shared device pointer and descriptors between block, should be called before the destructor call.
void GpuBlock::mark_shared_buffers(){
    destroy_shared_buffers = true;
}

void GpuBlock::mark_shared_weights(){
    destroy_shared_weights = true;
}



//Call this method to destroy the shared device pointer and descriptors between block, should be called before the destructor call.
void GpuBlock::free_buffers(){
    cudaFree(d_x);
    cudaFree(d_t);
    cudaFree(d_y);
    cudaFree(d_h);
    cudaFree(d_workspace_mlp);
    destroy_shared_buffers = false;
}

void GpuBlock::free_weights(){
    CUDA_CHECK(cudaFree(d_n1_bias) );
    CUDA_CHECK(cudaFree(d_n1_scale));
    CUDA_CHECK(cudaFree(d_n2_bias) );
    CUDA_CHECK(cudaFree(d_n2_scale));
    
    CUDA_CHECK(cudaFree(d_fc1)    );
    CUDA_CHECK(cudaFree(d_b1_data));
    CUDA_CHECK(cudaFree(d_b1_mtx) );
    CUDA_CHECK(cudaFree(d_fc2)    );
    CUDA_CHECK(cudaFree(d_b2_data));
    CUDA_CHECK(cudaFree(d_b2_mtx) );
    
    // destroy transpose descs if created
    if (transposeDesc) cublasLtMatrixTransformDescDestroy(transposeDesc);
    if (mlp_out_desc)  cublasLtMatrixLayoutDestroy(mlp_out_desc);
    if (res_in_desc)   cublasLtMatrixLayoutDestroy(res_in_desc);

    CUDA_CHECK(cudaFree(fused_desc.dWeights));

    destroy_shared_weights = false;
}

//Initialize the block descriptors 
void GpuBlock::set_descriptors(
    cudaStream_t stream_,
    cublasLtHandle_t ltHandle_,
    cudnnHandle_t cudnnHandle_,
    attn_cuDNN_descriptors fused_desc_, /*MHA cuDNN descriptors*/
    cublasLt_matmul_desc matmul_[2], cublasLtMatmulAlgo_t algo_[2], /*MLP descriptors*/
    cublasLtMatrixTransformDesc_t transposeDesc_, /*MLP Transpose descriptors*/
    void * d_mlp_workspace_,
    cublasLtMatrixLayout_t mlp_out_desc_, cublasLtMatrixLayout_t res_in_desc_     
)
{
    stream = stream_;
    ltHandle = ltHandle_;
    cudnnHandle = cudnnHandle_;
    fused_desc = fused_desc_;
    set_matmul_descriptors(matmul_, algo_, d_mlp_workspace_);
    transposeDesc = transposeDesc_;
    mlp_out_desc = mlp_out_desc_;
    res_in_desc = res_in_desc_;
}

//Copy of the descriptors
void GpuBlock::get_descriptors(
    cudaStream_t &_stream,
    cublasLtHandle_t &_ltHandle,
    cudnnHandle_t &_cudnnHandle,
    attn_cuDNN_descriptors &_fused_desc, /*MHA cuDNN descriptors*/
    cublasLt_matmul_desc (&_matmul)[2], cublasLtMatmulAlgo_t (&_algo)[2], /*MLP descriptors*/
    cublasLtMatrixTransformDesc_t &_transposeDesc, /*MLP Transpose descriptors*/
    cublasLtMatrixLayout_t &_mlp_out_desc, cublasLtMatrixLayout_t &_res_in_desc 
){
    _stream = stream;
    _ltHandle = ltHandle;
    _cudnnHandle = cudnnHandle;
    _fused_desc = fused_desc;
    _matmul[0] = matmul[0];
    _matmul[1] = matmul[1];
    _algo[0]   = algo[0];
    _algo[1]   = algo[1];
    _transposeDesc = transposeDesc;
    _mlp_out_desc = mlp_out_desc;
    _res_in_desc = res_in_desc; 

}

float GpuBlock::get_rand_scale(){
    return rand_scale;
}

u_int GpuBlock::get_hidden_elements_number(){
    return hidden_elements_number;
}


void GpuBlock::to_CPU(Block &cpu_block, bool debug){
    //Convert the device weights to float host
    float * n1_bias, *n1_scale,
            * n2_bias, *n2_scale,
            * fc1_data,* b1_data,
            * fc2_data,* b2_data;
    vector<half>  h_n1_bias(channels), h_n1_scale(channels),
            h_n2_bias(channels), h_n2_scale(channels),
            h_fc1(channels * k_channels), h_b1_data(k_channels),
            h_fc2(channels * k_channels), h_b2_data(channels);          
    if(debug) cout << "var" << endl;
    size_t channel_bytes = sizeof(half) * channels;
    size_t k_channels_bytes = sizeof(half) * k_channels;
    size_t mtx_bytes = sizeof(half) * channels * k_channels;
    
    if(debug) cout << "float malloc" << endl;
    n1_bias = (float *)malloc(channel_bytes * 2); n1_scale= (float *)malloc(channel_bytes * 2);
    n2_bias = (float *)malloc(channel_bytes * 2); n2_scale= (float *)malloc(channel_bytes * 2);
    fc1_data = (float *)malloc(mtx_bytes * 2); fc2_data = (float *)malloc(mtx_bytes * 2);
    b1_data = (float *)malloc(k_channels_bytes * 2);  b2_data = (float *)malloc(channel_bytes * 2);
    if(debug) cout << "ln memcpy" << endl;
    cudaMemcpy(h_n1_bias.data(), d_n1_bias, channel_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_n1_scale.data(), d_n1_scale, channel_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_n2_bias.data(), d_n2_bias, channel_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_n2_scale.data(), d_n2_scale, channel_bytes, cudaMemcpyDeviceToHost);
    if(debug) cout << "mlp memcpy" << endl;
    cudaMemcpy(h_fc1.data(), d_fc1, mtx_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fc2.data(), d_fc2, mtx_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b1_data.data(), d_b1_data, k_channels_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b2_data.data(), d_b2_data, channel_bytes, cudaMemcpyDeviceToHost);
    if(debug) cout << "ln conversion" << endl;
    f16_to_f32(h_n1_bias.data(), n1_bias, channels);
    f16_to_f32(h_n1_scale.data(), n1_scale, channels);
    f16_to_f32(h_n2_bias.data(), n2_bias, channels);
    f16_to_f32(h_n2_scale.data(), n2_scale, channels);
    if(debug) cout << "mlp conversion" << endl;
    f16_to_f32(h_fc1.data(),fc1_data,  channels * k_channels);
    f16_to_f32(h_fc2.data(),fc2_data,  channels * k_channels);
    f16_to_f32(h_b1_data.data(), b1_data, k_channels);
    f16_to_f32(h_b2_data.data(), b2_data, channels);


    //-Attention
    Matrix q(h_q, channels * channels, channels, channels);
    if(debug) {
        cout << "### q" << endl;
        q.print();
    }

    Matrix k(h_k, channels*channels, channels, channels);
    if(debug) {
        cout << "### k" << endl;
        k.print();
    }
    
    Matrix v(h_v, channels*channels, channels, channels);
    if(debug){ 
        cout << "### v" << endl;
        v.print();
    }
    RowVector qb(h_qb, channels);
    if(debug){ 
        cout << "### qb" << endl;
        qb.print();
    }
    RowVector kb(h_kb, channels);
    if(debug){ 
        cout << "### kb" << endl;
        kb.print();
    }
    RowVector vb(h_vb, channels);
    if(debug){ 
        cout << "### vb" << endl;
        vb.print();
    }
    Matrix p(h_p, channels*channels, channels, channels);
    if(debug){ 
        cout << "### p" << endl;
        p.print();
    }
    RowVector pb(h_pb, channels);
    if(debug){ 
        cout << "### pb" << endl;
        pb.print();
    }
    Attention attn(channels, num_heads, false, false);

    Linear q_gen(channels, channels, true);
    q_gen.move_A(q);
    q_gen.move_b(qb);
    Linear k_gen(channels, channels, true);
    k_gen.move_A(k);
    k_gen.move_b(kb);
    Linear v_gen(channels, channels, true);
    v_gen.move_A(v);
    v_gen.move_b(vb);
    Linear proj(channels, channels, true);
    proj.move_A(p);
    proj.move_b(pb);

    attn.move_qkv_gen(q_gen, k_gen, v_gen);
    attn.move_proj(proj);

    // Mlp Initialization
    Matrix A1(fc1_data, k_channels * channels, k_channels, channels);
    if(debug){ 
        cout << "### A1" << endl;
        A1.print();
    }
    RowVector b1(b1_data, k_channels);
    if(debug){ 
        cout << "### b1" << endl;
        b1.print();
    }
    Matrix A2(fc2_data, channels * k_channels, channels, k_channels);
    if(debug){ 
        cout << "### A2" << endl;
        A2.print();
    }
    RowVector b2(b2_data, channels);
    if(debug){ 
        cout << "### b2" << endl;
        b2.print();
    }
    Linear fc1(channels, k_channels, true);
    fc1.move_A(A1);
    fc1.move_b(b1);
    Linear fc2(k_channels, channels, true);
    fc2.move_A(A2);
    fc2.move_b(b2);

    Mlp mlp(channels, k_channels, channels, GELU, true, false);

    mlp.move_fc1(fc1);
    mlp.move_fc2(fc2);
    if(debug) cout << fc1.get_in_features()<< " x " << fc1.get_out_features() << endl;
    if(debug) cout << fc2.get_in_features()<< " x " << fc2.get_out_features() << endl;

    RowVector n1g(n1_scale, channels);
    if(debug){ 
        cout << "### n1g" << endl;
        n1g.print();
    }
    RowVector n1b(n1_bias, channels);
    if(debug){ 
        cout << "### n1b" << endl;
        n1b.print();
    }
    RowVector n2g(n2_scale, channels);
    if(debug){ 
        cout << "### n2g" << endl;
        n2g.print();
    }
    RowVector n2b(n2_bias, channels);
    if(debug){ 
        cout << "### n2b" << endl;
        n2b.print();
    }
    LayerNorm block_n1(channels, epsilon, true);
    block_n1.move_g(n1g);
    block_n1.move_b(n1b);
    LayerNorm block_n2(channels, epsilon, true);
    block_n2.move_g(n2g);
    block_n2.move_b(n2b);

    // Block Initialization
    cpu_block.move_attn(attn);
    cpu_block.move_mlp(mlp);
    cpu_block.move_norm1(block_n1);
    cpu_block.move_norm2(block_n2);
}

void GpuBlock::print_h_out(){
    float * f_debug = (float *)calloc(input_elements_number,sizeof(float));
    f16_to_f32(h_debug_out, f_debug, input_elements_number);
    Tensor y(f_debug, input_elements_number, batch, tokens, channels);
    y.print();
}

// helper: pull d_x (B*T*C) back to host_half buffer for debug
void GpuBlock::download_x(float * h_x) {
    size_t bytes_xBTc = sizeof(half) * batch * tokens * channels;
    half * tmp = (half *)malloc(sizeof(half) * batch * tokens * channels); 

    CUDA_CHECK(cudaMemcpy(tmp, d_x, bytes_xBTc, cudaMemcpyDeviceToHost));
    f16_to_f32(tmp,h_x, batch * tokens * channels);
}


/*  
-- PRIVATE METHODS --
*/

void GpuBlock::print_debug(){
    float * h_xty_gpu = (float *)malloc(input_elements_number * sizeof(float));
    float * h_h_gpu = (float *)malloc(hidden_elements_number * sizeof(float));
    half * h_xty_half = (half *)malloc(input_elements_number * sizeof(half));
    half * h_h_half = (half *)malloc(hidden_elements_number * sizeof(half));


    CUDA_CHECK(cudaMemcpy(h_xty_half, d_x, sizeof(half) * input_elements_number, cudaMemcpyDeviceToHost));
    f16_to_f32(h_xty_half, h_xty_gpu, input_elements_number);
    Tensor out_x(h_xty_gpu,input_elements_number,batch, tokens, channels);
    cout << "d_x | ";out_x.print();

    CUDA_CHECK(cudaMemcpy(h_xty_half, d_t, sizeof(half) * input_elements_number, cudaMemcpyDeviceToHost));
    f16_to_f32(h_xty_half, h_xty_gpu, input_elements_number);
    Tensor out_t(h_xty_gpu,input_elements_number,batch, tokens, channels);
    cout << "d_t | ";out_t.print();

    CUDA_CHECK(cudaMemcpy(h_h_half, d_h, sizeof(half) * hidden_elements_number, cudaMemcpyDeviceToHost));
    f16_to_f32(h_h_half, h_h_gpu, hidden_elements_number);
    Tensor out_h(h_h_gpu, hidden_elements_number,batch, tokens, k_channels);
    cout << "d_h | ";out_h.print();

    CUDA_CHECK(cudaMemcpy(h_xty_half, d_y, sizeof(half) * input_elements_number, cudaMemcpyDeviceToHost));
    f16_to_f32(h_xty_half, h_xty_gpu, input_elements_number);
    Tensor out_y(h_xty_gpu,input_elements_number,batch, tokens, channels);
    cout << "d_y | ";out_y.print();

}

//Transpose a square channels X channels mtx from float to half 
void GpuBlock::transposeHostF32toHalf(float* src, vector<half>& dst){
    for(int r = 0; r < channels; r++){
        for(int c = 0; c < channels; c++){
            dst[c * channels + r] = __float2half(src[r*channels + c]);
        }
    }
}

//Initialize randomly the passed DEVICE vector with `dim` size (half type), scaled by `rand_scale`
void GpuBlock::populate_rand(void * d_var, u_int dim){
    u_int blocks_n = (dim / 256) + 1;
    half * d_buffer; cudaMalloc(&d_buffer, sizeof(half) * dim);
    generate_reference<<<blocks_n, 256>>>(d_buffer, dim, rand_scale);
    CUDA_CHECK(cudaMemcpy(d_var,d_buffer,sizeof(half) * dim,cudaMemcpyDeviceToDevice));
    
}

//Initialize randomly the passed HOST vector with `dim` size (float type), , scaled by `rand_scale`
void GpuBlock::populate_rand(float * h_var, u_int dim){
    u_int blocks_n = (dim / 256) + 1;
    float * d_buffer; cudaMalloc(&d_buffer, sizeof(float) * dim);
    generate_reference<<<blocks_n, 256>>>(d_buffer, dim, rand_scale); 
    CUDA_CHECK(cudaMemcpy(h_var,d_buffer,sizeof(float) * dim,cudaMemcpyDeviceToHost));
    
}

GpuBlock::GpuBlock(
    u_int B_, u_int T_, u_int C_, u_int K_,
    bool kernel_type_,
    double epsilon_, float scale_, int num_heads_,
    float rand_scale_
): batch(B_), tokens(T_), channels(C_), k_channels(K_),
    kernel_type(kernel_type_), 
    epsilon(epsilon_), scale(scale_), num_heads(num_heads_), rand_scale(rand_scale_)
{
    /* -- Initialize all the descriptors -- */
    assert(channels % num_heads == 0);
    // 1. Create stream
    CUDA_CHECK(cudaStreamCreate(&stream));

    // 2. Create cuBLASLt handle
    CUBLAS_CHECK(cublasLtCreate(&ltHandle));

    // 3. Create cuDNN handle
    CUDNN_CHECK(cudnnCreate(&cudnnHandle));
    CUDNN_CHECK(cudnnSetStream(cudnnHandle, stream));

    // 4. Allocate main activation buffers on device
    input_elements_number = batch * tokens * channels;
    hidden_elements_number = batch * tokens * k_channels;
    size_t bytes_input = sizeof(half) * input_elements_number;
    size_t bytes_hidden  = sizeof(half) * hidden_elements_number;

    CUDA_CHECK(cudaMalloc(&d_x, bytes_input));  CUDA_CHECK(cudaMemset(d_x, 0, bytes_input));
    CUDA_CHECK(cudaMalloc(&d_t, bytes_input));  CUDA_CHECK(cudaMemset(d_t, 0, bytes_input));
    CUDA_CHECK(cudaMalloc(&d_y, bytes_input));  CUDA_CHECK(cudaMemset(d_y, 0, bytes_input));
    CUDA_CHECK(cudaMalloc(&d_h, bytes_hidden)); CUDA_CHECK(cudaMemset(d_h, 0, bytes_hidden));
    

    // 5. Attention variables
    h_q  = (float *)malloc(sizeof(float) * channels * channels);
    h_k  = (float *)malloc(sizeof(float) * channels * channels);
    h_v  = (float *)malloc(sizeof(float) * channels * channels);
    h_p  = (float *)malloc(sizeof(float) * channels * channels);
    h_qb = (float *)malloc(sizeof(float) * channels);
    h_kb = (float *)malloc(sizeof(float) * channels);
    h_vb = (float *)malloc(sizeof(float) * channels);
    h_pb = (float *)malloc(sizeof(float) * channels);

    // 6. LayerNorm params
    CUDA_CHECK(cudaMalloc(&d_n1_bias,  sizeof(half)*channels));
    CUDA_CHECK(cudaMalloc(&d_n1_scale, sizeof(half)*channels));
    CUDA_CHECK(cudaMalloc(&d_n2_bias,  sizeof(half)*channels));
    CUDA_CHECK(cudaMalloc(&d_n2_scale, sizeof(half)*channels));

    // 7. MLP weights/biases
    size_t bytes_fc1 = sizeof(half)*k_channels*channels;
    size_t bytes_fc2 = sizeof(half)*channels*k_channels;
    size_t bytes_b1  = sizeof(half)*k_channels;
    size_t bytes_b2  = sizeof(half)*channels;
    size_t bytes_b1_mtx = sizeof(half)*hidden_elements_number;
    size_t bytes_b2_mtx = sizeof(half)*input_elements_number;

    CUDA_CHECK(cudaMalloc(&d_fc1,     bytes_fc1));
    CUDA_CHECK(cudaMalloc(&d_b1_data, bytes_b1));
    CUDA_CHECK(cudaMalloc(&d_fc2,     bytes_fc2));
    CUDA_CHECK(cudaMalloc(&d_b2_data, bytes_b2));
    if(kernel_type) {
        CUDA_CHECK(cudaMalloc(&d_b1_mtx,  bytes_b1_mtx));
        CUDA_CHECK(cudaMalloc(&d_b2_mtx,  bytes_b2_mtx));
    }
    // 8. cuBLASLt MLP descriptors
    mlp_dimensions mdim(batch, tokens, channels, k_channels, channels);
    CUDA_CHECK(cudaMalloc(&d_workspace_mlp, (size_t)MLP_WORKSPACE_SIZE));
    create_mlp_descriptors(ltHandle, matmul, d_workspace_mlp, algo, mdim, kernel_type);


    // 9. Optional transpose descriptors if kernel_type == true (like your code)
    if (kernel_type) {
        cublasOperation_t op = CUBLAS_OP_T;

        CUBLAS_CHECK(cublasLtMatrixTransformDescCreate(&transposeDesc, CUDA_R_32F));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&mlp_out_desc, CUDA_R_16F, /*rows*/batch*tokens, /*cols*/channels, /*ld*/batch*tokens));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&res_in_desc, CUDA_R_16F, /*rows*/channels, /*cols*/batch*tokens, /*ld*/channels));
        CUBLAS_CHECK(cublasLtMatrixTransformDescSetAttribute(
            transposeDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &op, sizeof(op)
        ));
    }


    // 10. host debug buffer for pulling results back
    h_debug_out = (half*)malloc(sizeof(half) * input_elements_number);

}

/*
`initialize_descriptors`: if true, it will create and initialize the stream, handles and mlp descriptors
*/
GpuBlock::GpuBlock(
    u_int B_, u_int T_, u_int C_, u_int K_,
    void * d_x_, void * d_h_, void * d_t_, void * d_y_,
    bool kernel_type_,
    double epsilon_, float scale_, int num_heads_,
    float rand_scale_, 
    bool initialize_descriptors,
    bool allocate_weights
): batch(B_), tokens(T_), channels(C_), k_channels(K_),
    d_x(d_x_), d_h(d_h_), d_t(d_t_), d_y(d_y_),
    kernel_type(kernel_type_), 
    epsilon(epsilon_), scale(scale_), num_heads(num_heads_), rand_scale(rand_scale_)
{
    // 0. Initialize all the descriptors
    assert(channels % num_heads == 0);
    if(initialize_descriptors){
        // 1. Create stream
        CUDA_CHECK(cudaStreamCreate(&stream));

        // 2. Create cuBLASLt handle
        CUBLAS_CHECK(cublasLtCreate(&ltHandle));

        // 3. Create cuDNN handle
        CUDNN_CHECK(cudnnCreate(&cudnnHandle));
        CUDNN_CHECK(cudnnSetStream(cudnnHandle, stream));

        // 8. cuBLASLt MLP descriptors
        mlp_dimensions mdim(batch, tokens, channels, k_channels, channels);
        CUDA_CHECK(cudaMalloc(&d_workspace_mlp, (size_t)MLP_WORKSPACE_SIZE));
        create_mlp_descriptors(ltHandle, matmul, d_workspace_mlp, algo, mdim, kernel_type);
    

        // 9. Optional transpose descriptors if kernel_type == true (like your code)
        if (kernel_type) {
            cublasOperation_t op = CUBLAS_OP_T;

            CUBLAS_CHECK(cublasLtMatrixTransformDescCreate(&transposeDesc, CUDA_R_32F));
            CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&mlp_out_desc, CUDA_R_16F, /*rows*/batch*tokens, /*cols*/channels, /*ld*/batch*tokens));
            CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&res_in_desc, CUDA_R_16F, /*rows*/channels, /*cols*/batch*tokens, /*ld*/channels));
            CUBLAS_CHECK(cublasLtMatrixTransformDescSetAttribute(
                transposeDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &op, sizeof(op)
            ));
        }
    }

    // 1. Allocate main activation buffers on device
    input_elements_number = batch * tokens * channels;
    hidden_elements_number = batch * tokens * k_channels;

    // 2. Attention variables
    h_q  = (float *)malloc(sizeof(float) * channels * channels);
    h_k  = (float *)malloc(sizeof(float) * channels * channels);
    h_v  = (float *)malloc(sizeof(float) * channels * channels);
    h_p  = (float *)malloc(sizeof(float) * channels * channels);
    h_qb = (float *)malloc(sizeof(float) * channels);
    h_kb = (float *)malloc(sizeof(float) * channels);
    h_vb = (float *)malloc(sizeof(float) * channels);
    h_pb = (float *)malloc(sizeof(float) * channels);

    // 3. LayerNorm params
    if(allocate_weights){
        CUDA_CHECK(cudaMalloc(&d_n1_bias,  sizeof(half)*channels));
        CUDA_CHECK(cudaMalloc(&d_n1_scale, sizeof(half)*channels));
        CUDA_CHECK(cudaMalloc(&d_n2_bias,  sizeof(half)*channels));
        CUDA_CHECK(cudaMalloc(&d_n2_scale, sizeof(half)*channels));
    }
    // 4. MLP weights/biases
    size_t bytes_fc1 = sizeof(half)*k_channels*channels;        
    size_t bytes_fc2 = sizeof(half)*channels*k_channels;        
    size_t bytes_b1  = sizeof(half)*k_channels;                 
    size_t bytes_b2  = sizeof(half)*channels;                   
    size_t bytes_b1_mtx = sizeof(half)*hidden_elements_number;
    size_t bytes_b2_mtx = sizeof(half)*input_elements_number;
    if(allocate_weights){
        CUDA_CHECK(cudaMalloc(&d_fc1,     bytes_fc1));
        CUDA_CHECK(cudaMalloc(&d_b1_data, bytes_b1));
        CUDA_CHECK(cudaMalloc(&d_fc2,     bytes_fc2));
        CUDA_CHECK(cudaMalloc(&d_b2_data, bytes_b2));
        if(kernel_type) {
            CUDA_CHECK(cudaMalloc(&d_b1_mtx,  bytes_b1_mtx));
            CUDA_CHECK(cudaMalloc(&d_b2_mtx,  bytes_b2_mtx));
        }
    }


    // 5. host debug buffer for pulling results back
    h_debug_out = (half*)malloc(sizeof(half) * input_elements_number);

}


GpuBlock::GpuBlock(
    u_int B_, u_int T_, u_int C_, u_int K_,
    bool kernel_type_,
    double epsilon_, float scale_,
    //Layer norm
    float* n1b_data,
    float* n1g_data,
    float* n2b_data,
    float* n2g_data,
    //Attention
    float* q_data,
    float* k_data,
    float* v_data,
    float* p_data,   // O proj
    float* qb_data,
    float* kb_data,
    float* vb_data,
    float* pb_data,
    //Mlp
    float* A1_data,  // fc1 weights KxC
    float* b1_data,  // fc1 bias   K
    float* A2_data,  // fc2 weights MxK
    float* b2_data,   // fc2 bias   M

    int num_heads_,
    float rand_scale_,
    bool initialize_descriptors
):
batch(B_), tokens(T_), channels(C_), k_channels(K_),
kernel_type(kernel_type_), 
epsilon(epsilon_), scale(scale_), num_heads(num_heads_), rand_scale(rand_scale_)
{
    assert(channels % num_heads == 0);

    // 0. Allocate main activation buffers on device
    input_elements_number = batch * tokens * channels;
    hidden_elements_number = batch * tokens * k_channels;

    /* -- Initialize all the descriptors -- */
    if(initialize_descriptors){
        // 1. Create stream
        CUDA_CHECK(cudaStreamCreate(&stream));

        // 2. Create cuBLASLt handle
        CUBLAS_CHECK(cublasLtCreate(&ltHandle));

        // 3. Create cuDNN handle
        CUDNN_CHECK(cudnnCreate(&cudnnHandle));
        CUDNN_CHECK(cudnnSetStream(cudnnHandle, stream));

        // 5. cuBLASLt MLP descriptors
        mlp_dimensions mdim(batch, tokens, channels, k_channels, channels);
        CUDA_CHECK(cudaMalloc(&d_workspace_mlp, (size_t)MLP_WORKSPACE_SIZE));
        create_mlp_descriptors(ltHandle, matmul, d_workspace_mlp, algo, mdim, kernel_type);
    

        // 6. Optional transpose descriptors if kernel_type == true (like your code)
        if (kernel_type) {
            cublasOperation_t op = CUBLAS_OP_T;

            CUBLAS_CHECK(cublasLtMatrixTransformDescCreate(&transposeDesc, CUDA_R_32F));
            CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&mlp_out_desc, CUDA_R_16F, /*rows*/batch*tokens, /*cols*/channels, /*ld*/batch*tokens));
            CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&res_in_desc, CUDA_R_16F, /*rows*/channels, /*cols*/batch*tokens, /*ld*/channels));
            CUBLAS_CHECK(cublasLtMatrixTransformDescSetAttribute(
                transposeDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &op, sizeof(op)
            ));
        }
    }

    // 1. Attention variables
    h_q  = (float *)malloc(sizeof(float) * channels * channels);
    h_k  = (float *)malloc(sizeof(float) * channels * channels);
    h_v  = (float *)malloc(sizeof(float) * channels * channels);
    h_p  = (float *)malloc(sizeof(float) * channels * channels);
    h_qb = (float *)malloc(sizeof(float) * channels);
    h_kb = (float *)malloc(sizeof(float) * channels);
    h_vb = (float *)malloc(sizeof(float) * channels);
    h_pb = (float *)malloc(sizeof(float) * channels);

    // 2. LayerNorm params
    CUDA_CHECK(cudaMalloc(&d_n1_bias,  sizeof(half)*channels));
    CUDA_CHECK(cudaMalloc(&d_n1_scale, sizeof(half)*channels));
    CUDA_CHECK(cudaMalloc(&d_n2_bias,  sizeof(half)*channels));
    CUDA_CHECK(cudaMalloc(&d_n2_scale, sizeof(half)*channels));

    // 3. MLP weights/biases
    size_t bytes_fc1 = sizeof(half)*k_channels*channels;
    size_t bytes_fc2 = sizeof(half)*channels*k_channels;
    size_t bytes_b1  = sizeof(half)*k_channels;
    size_t bytes_b2  = sizeof(half)*channels;
    size_t bytes_b1_mtx = sizeof(half)*hidden_elements_number;
    size_t bytes_b2_mtx = sizeof(half)*input_elements_number;

    CUDA_CHECK(cudaMalloc(&d_fc1,     bytes_fc1));
    CUDA_CHECK(cudaMalloc(&d_b1_data, bytes_b1));
    CUDA_CHECK(cudaMalloc(&d_fc2,     bytes_fc2));
    CUDA_CHECK(cudaMalloc(&d_b2_data, bytes_b2));
    if(kernel_type) {
        CUDA_CHECK(cudaMalloc(&d_b1_mtx,  bytes_b1_mtx));
        CUDA_CHECK(cudaMalloc(&d_b2_mtx,  bytes_b2_mtx));
    }


    // 4. host debug buffer for pulling results back
    h_debug_out = (half*)malloc(sizeof(half) * input_elements_number);

    //5. set all the weights on the device and host respectively
    set_data(
        n1b_data,
        n1g_data,
        n2b_data,
        n2g_data,
    
        q_data,
        k_data,
        v_data,
        p_data, 
        qb_data,
        kb_data,
        vb_data,
        pb_data,

        A1_data,
        b1_data,
        A2_data,
        b2_data
    );

    //6. Initialize the attention descriptors
    if(initialize_descriptors)
        init_attn_descriptor(
            h_q , 
            h_k ,
            h_v ,
            h_p ,
            h_qb,
            h_kb,
            h_vb,
            h_pb
        );

}
