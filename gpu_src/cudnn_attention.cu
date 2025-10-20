#include "../gpu_include/cudnn_attention.h"

void cudnn_attention(
    mtx q_host,
    mtx k_host,
    mtx v_host,
    mtx p_host,
    h_tensor x_host,
    vector<__half> qb_data,
    vector<__half> kb_data,
    vector<__half> vb_data,
    vector<__half> pb_data,
    half * host_out
){
    //Variable definition

    const int seq_length = x_host.C; // 7
    const int batch_size = x_host.B; // 2
    const int beam_dim = x_host.H;   // 9
    const int emb_dim = x_host.W;    // 9
    const int input_size = x_host.C * x_host.B * x_host.W * x_host.H;
    // const double scale = NUM_HEADS == 1 ? 1.0 : pow( x_host.W / NUM_HEADS, -0.5);
    const double scale = pow( x_host.W / NUM_HEADS, -0.5); // seems right?

    assert(qb_data.size() > 0);
    assert(kb_data.size() > 0);
    assert(vb_data.size() > 0);
    assert(pb_data.size() > 0);

    cout << __half2float( qb_data.at(0)) << endl;
    cout << __half2float( kb_data.at(0)) << endl;
    cout << __half2float( vb_data.at(0)) << endl;
    cout << __half2float( pb_data.at(0)) << endl;

    cout<< "first elem: "<< __half2float(x_host.data[0]) << endl;
    const int qkv_projSize = x_host.W / NUM_HEADS; //should be the size of projection (so MHA like 3)
    const int o_projSize = qkv_projSize * NUM_HEADS;
    // const int qk_projSize = 9;
    // const int ov_projSize = 9;

    // 0) Handle and tensor descriptors creation
    cudnnHandle_t handle;
    CUDNN_CHECK( cudnnCreate(&handle));
    
    
    //1) Create the attnDropout descriptors (not used)
    // --- Dropout descriptors (set to 0.0) required by Attn descriptor
    cudnnDropoutDescriptor_t attnDrop = nullptr, postDrop = nullptr;
    CUDNN_CHECK(cudnnCreateDropoutDescriptor(&attnDrop));
    CUDNN_CHECK(cudnnCreateDropoutDescriptor(&postDrop));
    size_t statesSize = 0;
    CUDNN_CHECK(cudnnDropoutGetStatesSize(handle, &statesSize));
    void* states = nullptr;
    CUDA_CHECK(cudaMalloc(&states, statesSize));
    CUDNN_CHECK(cudnnSetDropoutDescriptor(attnDrop, handle, /*dropout*/0.0f, states, statesSize, /*seed*/0ULL));
    CUDNN_CHECK(cudnnSetDropoutDescriptor(postDrop, handle, /*dropout*/0.0f, states, statesSize, /*seed*/0ULL));


    //2) Create and set the attention descriptor
    cudnnAttnDescriptor_t attention_descriptor;
    CUDNN_CHECK(
         cudnnCreateAttnDescriptor(&attention_descriptor)
    );
    
    cudnnDataType_t dataType = CUDNN_DATA_HALF;
    
    CUDNN_CHECK(
        cudnnSetAttnDescriptor(
            attention_descriptor,
            /*attnMode flags*/ ATTN_MODE, 
            /*nHeads*/ NUM_HEADS,
            /*smScaler*/ scale,              
            /*dataType*/ dataType,  // inputs/weights/output datatype
            /*computePrec*/ ATTN_COMPUTE_TYPE,
            /*mathType*/ ATTN_MATH_TYPE, 
            /*attnDropoutDesc*/ attnDrop,
            /*postDropoutDesc*/ postDrop,
            /*qSize*/ emb_dim, /*kSize*/ emb_dim, /*vSize*/ emb_dim,
            /*qProjSize*/ qkv_projSize, /*kProjSize*/ qkv_projSize, /*vProjSize*/ qkv_projSize, /*oProjSize*/ o_projSize,
            /*qoMaxSeqLength*/ seq_length,
            /*kvMaxSeqLength*/ seq_length,
            /*maxBatchSize*/ batch_size,
            /*maxBeamSize*/ beam_dim
        )
    );

    //3) Create and set sequence descriptors
    auto makeSeqDesc = [&](cudnnSeqDataDescriptor_t& desc, int vecSize){
        CUDNN_CHECK(cudnnCreateSeqDataDescriptor(&desc));
        // int dims[4]  = {seq_length, batch_size,beam_dim,vecSize};
        int dims[4];
        dims[CUDNN_SEQDATA_TIME_DIM]  = seq_length;
        dims[CUDNN_SEQDATA_BATCH_DIM] = batch_size;
        dims[CUDNN_SEQDATA_BEAM_DIM]  = beam_dim;
        dims[CUDNN_SEQDATA_VECT_DIM]  = vecSize; 
        cudnnSeqDataAxis_t axes[4] = {CUDNN_SEQDATA_BATCH_DIM, CUDNN_SEQDATA_TIME_DIM, CUDNN_SEQDATA_BEAM_DIM, CUDNN_SEQDATA_VECT_DIM};
        
        // Per-sample sequence lengths (Q/O and K/V — here all full length)
        int lens[batch_size];
        for(u_int b = 0; b < batch_size; b++) lens[b] = seq_length;
        
        CUDNN_CHECK(
            cudnnSetSeqDataDescriptor(
            desc, dataType, /*nbDims*/4, dims, axes,
            /*seqLengthArraySize*/ batch_size, lens, /*paddingFill*/nullptr)
        );
    };

    cudnnSeqDataDescriptor_t qDesc, kDesc, vDesc, oDesc;
    makeSeqDesc(qDesc, /*vecSize before proj*/emb_dim);
    makeSeqDesc(kDesc, /*vecSize before proj*/emb_dim);
    makeSeqDesc(vDesc, /*vecSize before proj*/emb_dim);
    makeSeqDesc(oDesc, /*vecSize after  proj*/emb_dim); 

    //4) Input allocation (and output vector buffer)
    half *dQ=nullptr, *dK=nullptr, *dV=nullptr, *dO=nullptr;
    CUDA_CHECK(cudaMalloc(&dQ, input_size*sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dK, input_size*sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dV, input_size*sizeof(half)));
    CUDA_CHECK(cudaMemcpy(dQ, x_host.data, input_size*sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, x_host.data, input_size*sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, x_host.data, input_size*sizeof(half), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMalloc(&dO, input_size*sizeof(half)));

    //5) Dev seq lengths (must be on device for forward)
    std::vector<int> hLen(batch_size*beam_dim, seq_length);
    int *dLenQO=nullptr, *dLenKV=nullptr;
    CUDA_CHECK(cudaMalloc(&dLenQO, hLen.size()*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dLenKV, hLen.size()*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dLenQO, hLen.data(), hLen.size()*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dLenKV, hLen.data(), hLen.size()*sizeof(int), cudaMemcpyHostToDevice));

    std::vector<int> loWin(seq_length, 0), hiWin(seq_length);
    for (int t=0; t<seq_length; ++t) hiWin[t] = seq_length;

    //6) Get buffer sizes
      // --- Query buffer sizes & allocate weights/workspace (reserve=NULL => inference)
    size_t weightBytes=0, workBytes=0;
    CUDNN_CHECK(cudnnGetMultiHeadAttnBuffers(handle, attention_descriptor, &weightBytes, &workBytes, /*reserveSpaceSize=*/nullptr)); // :contentReference[oaicite:4]{index=4}
    void* dWeights = nullptr; // no projections => may be 0 bytes
    if (weightBytes) CUDA_CHECK(cudaMalloc(&dWeights, weightBytes));
    void* dWork = nullptr;
    if (workBytes)  CUDA_CHECK(cudaMalloc(&dWork, workBytes));

    

    //7) Allocate weight array and set the projection descriptors
    auto fill_proj_identity = [&](cudnnMultiHeadAttnWeightKind_t kind, half * host_data, size_t host_size){
        cudnnTensorDescriptor_t wDesc;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&wDesc));
        void* wAddr = nullptr;
        // Get a descriptor and device address for this weight group
        CUDNN_CHECK(
            cudnnGetMultiHeadAttnWeights(handle, attention_descriptor, kind, weightBytes, dWeights,wDesc, &wAddr)
        );

        // Copy to the returned device address 
        CUDA_CHECK(cudaMemcpy(wAddr, host_data, host_size*sizeof(half), cudaMemcpyHostToDevice));

        CUDNN_CHECK(cudnnDestroyTensorDescriptor(wDesc));
    };
    
    if(qkv_projSize != 0){
        cout<< "qk projections" << endl;
        fill_proj_identity(CUDNN_MH_ATTN_Q_WEIGHTS, q_host.data, q_host.col_n * q_host.row_n);
        fill_proj_identity(CUDNN_MH_ATTN_K_WEIGHTS, k_host.data, k_host.col_n * k_host.row_n);
        fill_proj_identity(CUDNN_MH_ATTN_V_WEIGHTS, v_host.data, v_host.col_n * v_host.row_n);
        //biases
        fill_proj_identity(CUDNN_MH_ATTN_Q_BIASES, qb_data.data(), qb_data.size());
        fill_proj_identity(CUDNN_MH_ATTN_K_BIASES, kb_data.data(), kb_data.size());
        fill_proj_identity(CUDNN_MH_ATTN_V_BIASES, vb_data.data(), vb_data.size());
        
    }
    if(o_projSize != 0){
        cout << "ov projections" << endl;
        fill_proj_identity(CUDNN_MH_ATTN_O_WEIGHTS, p_host.data, p_host.col_n * p_host.row_n);
        fill_proj_identity(CUDNN_MH_ATTN_O_BIASES, pb_data.data(), pb_data.size());

    }


    //8) Forward: no residuals, whole sequence (currIdx = -1), reserve=NULL
    CUDNN_CHECK(cudnnMultiHeadAttnForward(
        handle, attention_descriptor,
        /*currIdx*/ -1,
        loWin.data(), hiWin.data(),
        dLenQO, dLenKV,
        qDesc, dQ,
        /*residuals*/ nullptr,// dQ,           
        kDesc, dK,
        vDesc, dV,
        oDesc, dO,
        weightBytes, dWeights,
        workBytes, dWork,
        /*reserveSpaceSizeInBytes*/ 0,   // inference path
        /*reserveSpace*/ nullptr)); 
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_out, dO, input_size * sizeof(half), cudaMemcpyDeviceToHost));
}