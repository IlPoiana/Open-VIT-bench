#include "../gpu_include/gpu_mlp.h"

void cublasLt_matmul_desc::destroy_descriptors(){
    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtMatrixLayoutDestroy(xDesc);
    cublasLtMatrixLayoutDestroy(fcDesc);
    cublasLtMatrixLayoutDestroy(cDesc);
    cublasLtMatrixLayoutDestroy(yDesc);
}

mlp_dimensions::mlp_dimensions(u_int _B, u_int _T,u_int _C,u_int _K,u_int _M){
    B = _B;
    T = _T;
    C = _C;
    K = _K;
    M = _M;
}

/**
 * @brief 
 * 
 * @param h_b 1xrow row-major bias vector 
 * @param h_b_mtx rowxcol col-major bias matrix
 * @param row
 * @param col
 */
void bias_matrix(half * h_b, half * h_b_mtx, u_int row, u_int col){
    for(u_int r = 0; r < row; r++){
        for(u_int c = 0; c < col; c++){
            h_b_mtx[r * col + c] = h_b[r];
        }
    }
    return;
}

void GEMM(
    cublasLtHandle_t &handle, cudaStream_t & stream,
    u_int B, u_int T, u_int C, u_int K,
    void * d_x, void * d_fc, void * d_b, 
    void * y
){

    // -- Descriptor creation --
    //-Matmul creation
    cublasLtMatmulDesc_t matmulDesc; CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmulDesc,MLP_COMPUTE_DATA_TYPE, CUDA_R_16F));

    cublasOperation_t Avalue = CUBLAS_OP_N;
    cublasOperation_t Bvalue = CUBLAS_OP_T;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA,&Avalue, sizeof(cublasOperation_t))); // do not transpose x
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB,&Bvalue, sizeof(cublasOperation_t))); // transpose fc

    cublasLtMatrixLayout_t xDesc; CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&xDesc,MLP_DATA_TYPE,B*T,C,C)); // row-major, not good for cublas it wants that the leading dim is the major one.
    cublasLtMatrixLayout_t fcDesc; CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&fcDesc,MLP_DATA_TYPE,K,C,C)); //row-major to transpose
    cublasLtMatrixLayout_t cDesc; CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&cDesc,MLP_DATA_TYPE,B*T,K,K)); // same as D because we are not using it! ==> beta = 0.0
    cublasLtMatrixLayout_t yDesc; CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&yDesc,MLP_DATA_TYPE,B*T,K,K)); // have to be col-major to enable the epilouge(fused kernel)

    //-Layout setting
    cublasLtOrder_t row = CUBLASLT_ORDER_ROW;

    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(xDesc,CUBLASLT_MATRIX_LAYOUT_ORDER,&row,sizeof(cublasLtOrder_t))); // row-major
    
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(fcDesc,CUBLASLT_MATRIX_LAYOUT_ORDER,&row,sizeof(cublasLtOrder_t)));
    
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(cDesc,CUBLASLT_MATRIX_LAYOUT_ORDER,&row,sizeof(cublasLtOrder_t))); // Setting C & D row-major
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(yDesc,CUBLASLT_MATRIX_LAYOUT_ORDER,&row,sizeof(cublasLtOrder_t)));



    // -- alpha and beta definition --
    float alpha = 1.0f, beta = 1.0f;
    cublasLtMatmul(
        handle, matmulDesc, &alpha,
        d_x, xDesc,
        d_fc, fcDesc,
        &beta,
        y, cDesc,
        y, yDesc,
        nullptr, nullptr, 0, stream
    );
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return;
}

//Check if everything is converted to float, in that case is more convenient keeping all the variables to float
__device__ half GELU(half val){
    half cube = val *val *val; 
     
    return __float2half(0.5f) * val * 
            ((half)1 + h_tanh(
                (SQRT_2_PI_fp16 *
                    (val + (__float2half(0.044715f) * cube))
                )
            )
        );
}

// In place add and GELU
__global__ void bias_GELU(half * d_x, half * d_bias, u_int bias_length, u_int N){

    for(int stride = blockIdx.x * blockDim.x + threadIdx.x;  stride < N;stride += gridDim.x *  blockDim.x){
        half val = d_x[stride];
        //add 
        val += d_bias[stride % bias_length];

        //GELU
        d_x[stride] = GELU(val);
        
    }
    return;
    
}

// In place add and GELU
__global__ void bias(half * d_x, half * d_bias, u_int bias_length, u_int N){
    
    for(int stride = blockIdx.x * blockDim.x + threadIdx.x;  stride < N;stride += gridDim.x *  blockDim.x){
        d_x[stride] += d_bias[stride % bias_length];
    }
    return;
    
}

//Create the descriptors for cublasLt matmul op
void create_cublasLt_linlay_desc(
    u_int B, u_int T, u_int C, u_int K,
    cublasLt_matmul_desc & matmul
){
    // -- Descriptor creation --
    //-Matmul creation
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmul.matmulDesc,MLP_COMPUTE_DATA_TYPE, CUDA_R_32F));

    cublasOperation_t Avalue = CUBLAS_OP_N;
    cublasOperation_t Bvalue = CUBLAS_OP_T;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul.matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA,&Avalue, sizeof(cublasOperation_t))); // do not transpose x
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul.matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB,&Bvalue, sizeof(cublasOperation_t))); // transpose fc

    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&matmul.xDesc,MLP_DATA_TYPE,B*T,C,C)); // row-major, not good for cublas it wants that the leading dim is the major one.
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&matmul.fcDesc,MLP_DATA_TYPE,K,C,C)); //row-major to transpose
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&matmul.cDesc,MLP_DATA_TYPE,B*T,K,K)); // same as D because we are not using it! ==> beta = 0.0
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&matmul.yDesc,MLP_DATA_TYPE,B*T,K,K)); // have to be col-major to enable the epilouge(fused kernel)

    //-Layout setting
    cublasLtOrder_t row = CUBLASLT_ORDER_ROW;

    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(matmul.xDesc,CUBLASLT_MATRIX_LAYOUT_ORDER,&row,sizeof(cublasLtOrder_t))); // row-major
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(matmul.fcDesc,CUBLASLT_MATRIX_LAYOUT_ORDER,&row,sizeof(cublasLtOrder_t)));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(matmul.cDesc,CUBLASLT_MATRIX_LAYOUT_ORDER,&row,sizeof(cublasLtOrder_t))); // Setting C & D row-major
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(matmul.yDesc,CUBLASLT_MATRIX_LAYOUT_ORDER,&row,sizeof(cublasLtOrder_t)));
    matmul.alpha = 1.0f; matmul.beta = 0.0f;
    return;
}

/**
 * @brief Create a cublasLt linlay desc object, overload for the fused_mlp
 * @param gelu 
 * @param memory_order: if true, the x matrix will be in row-major, other wise will be in col-major */
void create_cublasLt_linlay_desc(
    u_int B, u_int T, u_int C, u_int K,
    cublasLt_matmul_desc & matmul, 
    bool gelu, bool memory_order
){
    u_int N = B * T;
    // -- Descriptor creation --
    //-Matmul creation
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmul.matmulDesc,MLP_COMPUTE_DATA_TYPE, CUDA_R_32F)); // Scale type CUDA_R_16F not supported(doc)

    cublasOperation_t Avalue = CUBLAS_OP_N;
    cublasOperation_t Bvalue = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul.matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA,&Avalue, sizeof(cublasOperation_t))); // do not transpose x
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul.matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB,&Bvalue, sizeof(cublasOperation_t))); // transpose fc
    //Transpose for C not supported(doc)
    
    //-Epilogue creation
    if(gelu){
        cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_GELU;
        cublasLtMatmulDescSetAttribute(matmul.matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));
    }

    if(memory_order)
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&matmul.xDesc,MLP_DATA_TYPE,N,C,C)); 
    else
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&matmul.xDesc,MLP_DATA_TYPE,N,C,N)); 
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&matmul.fcDesc,MLP_DATA_TYPE,C,K,C));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&matmul.cDesc,MLP_DATA_TYPE,N,K,N)); // Specifically the leading dimension of C can be 0 to achieve row or column broadcast.
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&matmul.yDesc,MLP_DATA_TYPE,N,K,N)); // have to be col-major to enable the epilouge(fused kernel)

    //-Layout setting
    cublasLtOrder_t row = CUBLASLT_ORDER_ROW, col = CUBLASLT_ORDER_COL;

    if(memory_order)
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(matmul.xDesc,CUBLASLT_MATRIX_LAYOUT_ORDER,&row,sizeof(cublasLtOrder_t)));    
    else
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(matmul.xDesc,CUBLASLT_MATRIX_LAYOUT_ORDER,&col,sizeof(cublasLtOrder_t)));    

    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(matmul.fcDesc,CUBLASLT_MATRIX_LAYOUT_ORDER,&col,sizeof(cublasLtOrder_t)));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(matmul.cDesc,CUBLASLT_MATRIX_LAYOUT_ORDER,&col,sizeof(cublasLtOrder_t))); // Setting C & D col-major
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(matmul.yDesc,CUBLASLT_MATRIX_LAYOUT_ORDER,&col,sizeof(cublasLtOrder_t)));
    matmul.alpha = 1.0f; matmul.beta = 1.0f;

}

cublasLtMatmulAlgo_t fetch_matmul_algos(cublasLtHandle_t &handle,cublasLt_matmul_desc &matmul, void ** d_workspace,  bool initialize_workspace){
    u_int requested_count = 10;
    int count = 0;
    cublasLtMatmulHeuristicResult_t heur_array[requested_count];

    cublasLtMatmulPreference_t preference; CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    
    size_t workspace_size = WORKSPACE_SIZE;
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));

    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
       handle,
       matmul.matmulDesc,
       matmul.xDesc,
       matmul.fcDesc,
       matmul.cDesc,
       matmul.yDesc,
       preference,
       requested_count,
       heur_array,
       &count));
    cublasLtMatmulHeuristicResult_t res_algo; u_int idx = 0;
    cublasLtMatmulAlgo_t algo;
    for(u_int i = 0; i < count; i++){
        res_algo = heur_array[i];
        idx = i;
        if(res_algo.state == CUBLAS_STATUS_SUCCESS){
            algo = res_algo.algo;
            break;
        }
    }
    assert(heur_array[idx].state == CUBLAS_STATUS_SUCCESS); //otherwise no algo have been found(strange)
    if(initialize_workspace)
        CUDA_CHECK(cudaMalloc(d_workspace, workspace_size));

    return algo;
}

/**
 * @brief Create a mlp descriptors object
 * 
 * @param matmul: two elements array, one for each MLP matmul (first -> first layer)
 * @param d_workspace : device workspace (already initialized)
 * @param algo: two elements array, one for each MLP matmul algo (first -> first layer)
 */
void create_mlp_descriptors(
    cublasLtHandle_t &handle,
    cublasLt_matmul_desc * matmul, void * d_workspace, cublasLtMatmulAlgo_t * algo,
    mlp_dimensions dimensions,
    bool fused
){
    u_int B = dimensions.B,T = dimensions.T, C = dimensions.C ,K = dimensions.K, M = dimensions.M;
    //-Layer 1
    if(fused){
        create_cublasLt_linlay_desc(
            B,T,C,K,
            matmul[0],
            true, true
        );
    }
    else{
        create_cublasLt_linlay_desc(
            B,T,C,K,
            matmul[0]
        );
    }

    algo[0] = fetch_matmul_algos(handle, matmul[0], &d_workspace, false);

    //-Layer 2
    if(fused){
        create_cublasLt_linlay_desc(
            B,T,K,M,
            matmul[1],
            false, false
        );
    }
    else{
        create_cublasLt_linlay_desc(
            B,T,K,M,
            matmul[1]
        );
    }

    algo[1] = fetch_matmul_algos(handle, matmul[1], &d_workspace, false);
}


/*
0) We passed the already instantiate matmul descriptor, the algorithm to perform and the workspace.
*/
void linear_layer(
    cublasLtHandle_t & handle, cudaStream_t & stream,
    u_int B, u_int T, u_int K,
    cublasLt_matmul_desc &matmul,cublasLtMatmulAlgo_t &algo,void * d_workspace,
    void * d_x, void * d_fc, void * d_b, 
    void * d_y, bool gelu
){

    
    CUBLAS_CHECK(cublasLtMatmul(
        handle, matmul.matmulDesc, &matmul.alpha,
        d_x, matmul.xDesc,
        d_fc, matmul.fcDesc,
        &matmul.beta,
        d_y, matmul.cDesc,
        d_y, matmul.yDesc,
        &algo, d_workspace, (size_t)MLP_WORKSPACE_SIZE, stream
    ));
    // CUDA_CHECK(cudaStreamSynchronize(stream));

    u_int block_num = ((B*T*K) / MLP_BLOCK_DIM) + 1;
    if(gelu)
        bias_GELU<<<block_num, MLP_BLOCK_DIM,0,stream>>>((half *)d_y, (half *)d_b, K, B*T*K);
    else    
        bias<<<block_num, MLP_BLOCK_DIM,0,stream>>>((half *)d_y, (half *)d_b, K, B*T*K);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return;

}

/*
1) Strided version for the bias of the linear layer. Used in GpuViT
*/
void strided_linear_layer(
    cublasLtHandle_t & handle, cudaStream_t & stream,
    u_int B, u_int T, u_int K, u_int stride_val,
    cublasLt_matmul_desc &matmul,cublasLtMatmulAlgo_t &algo,void * d_workspace,
    void * d_x, void * d_fc, void * d_b, 
    void * d_y, bool gelu
){

    
    CUBLAS_CHECK(cublasLtMatmul(
        handle, matmul.matmulDesc, &matmul.alpha,
        d_x, matmul.xDesc,
        d_fc, matmul.fcDesc,
        &matmul.beta,
        d_y, matmul.cDesc,
        d_y, matmul.yDesc,
        &algo, d_workspace, (size_t)MLP_WORKSPACE_SIZE, stream
    ));
    // CUDA_CHECK(cudaStreamSynchronize(stream));

    u_int block_num = ((B*T*K) / (stride_val * MLP_BLOCK_DIM)) + 1;
    if(gelu)
        bias_GELU<<<block_num, MLP_BLOCK_DIM,0,stream>>>((half *)d_y, (half *)d_b, K, B*T*K);
    else    
        bias<<<block_num, MLP_BLOCK_DIM,0,stream>>>((half *)d_y, (half *)d_b, K, B*T*K);
    // CUDA_CHECK(cudaStreamSynchronize(stream));

    return;

}

/**
 * @brief 
 * The three added params have to be initialized
 * @param matmul: the struct yielding all the descriptors for the matmul
 * @param algo: the algorithm to perform
 * @param d_workspace: the device pointer to the allocated workspace
 */
void fused_linear_layer(
    cublasLtHandle_t &handle, cudaStream_t & stream,
    cublasLt_matmul_desc &matmul,cublasLtMatmulAlgo_t &algo,void * d_workspace,
    void * d_x, void * d_fc, void * d_b, 
    void * y
){
    
    CUBLAS_CHECK(cublasLtMatmul(
        handle, matmul.matmulDesc, &matmul.alpha,
        d_x, matmul.xDesc,
        d_fc, matmul.fcDesc,
        &matmul.beta,
        d_b, matmul.cDesc,
        y, matmul.yDesc,
        &algo, d_workspace, (size_t)MLP_WORKSPACE_SIZE, stream
    ));
    // CUDA_CHECK(cudaStreamSynchronize(stream)); //not necessary cause on the same stream
    return;
}



void gpu_mlp(
    cublasLtHandle_t & handle, cudaStream_t & stream,
    u_int B, u_int T, u_int C, u_int K,u_int M,
    void * d_x, void * d_fc1, void * d_h,void * d_b1, void * d_fc2, void * d_b2, 
    void * d_y
){
    linear_layer(
        handle,stream, 
        B,  T,  C,  K,
        d_x, d_fc1, d_b1, d_h
    );
    linear_layer(
        handle,stream,
        B,  T,  K,  M,
        d_h, d_fc2, d_b2, d_y,
        false
    );
    return;
}

/**
 * @brief 
 * Overload method for supporting cached descriptors, algo and workspace
 */
void gpu_mlp(
    cublasLtHandle_t & handle, cudaStream_t & stream,
    u_int B, u_int T, u_int K,u_int M,
    cublasLt_matmul_desc * matmul,cublasLtMatmulAlgo_t * algo,void * d_workspace,
    void * d_x, void * d_fc1, void * d_h,void * d_b1, void * d_fc2, void * d_b2, 
    void * d_y, int stride_val
){
    strided_linear_layer(
        handle,stream, 
        B,  T, K, stride_val,
        matmul[0], algo[0], d_workspace,
        d_x, d_fc1, d_b1, d_h, 
        true
    );
    strided_linear_layer(
        handle,stream,
        B,  T, M, stride_val,
        matmul[1], algo[1], d_workspace,
        d_h, d_fc2, d_b2, d_y,
        false
    );
    return;
}


/**
 * @brief 
 * Overload method for supporting cached descriptors, algo and workspace
 */
void fused_gpu_mlp(
    cublasLtHandle_t & handle, cudaStream_t & stream,
    cublasLt_matmul_desc * matmul,cublasLtMatmulAlgo_t * algo,void * d_workspace,
    void * d_x, void * d_fc1, void * d_h,void * d_b1, void * d_fc2, void * d_b2, 
    void * d_y
){
    fused_linear_layer(
        handle,stream, 
        matmul[0], algo[0], d_workspace,
        d_x, d_fc1, d_b1, d_h
    );
    fused_linear_layer(
        handle,stream,
        matmul[1], algo[1], d_workspace,
        d_h, d_fc2, d_b2, d_y
    );
    return;
}

/*
-- DEV phase functions --
*/


/*
0) GEMM + BIAS
*/
void linear_layer(
    cublasLtHandle_t & handle, cudaStream_t & stream,
    u_int B, u_int T, u_int C, u_int K,
    void * d_x, void * d_fc, void * d_b, 
    void * d_y, bool gelu
){

    cublasLt_matmul_desc matmul;
    create_cublasLt_linlay_desc(
        B,T,C,K,
        matmul
    );

    void * d_workspace;
    cublasLtMatmulAlgo_t algo = fetch_matmul_algos(handle, matmul, &d_workspace);

    
    CUBLAS_CHECK(cublasLtMatmul(
        handle, matmul.matmulDesc, &matmul.alpha,
        d_x, matmul.xDesc,
        d_fc, matmul.fcDesc,
        &matmul.beta,
        d_y, matmul.cDesc,
        d_y, matmul.yDesc,
        &algo, d_workspace, (size_t)MLP_WORKSPACE_SIZE, stream
    ));
    // CUDA_CHECK(cudaStreamSynchronize(stream));

    u_int block_dim = 256;
    u_int block_num = ((B*T*K) / block_dim) + 1;
    if(gelu)
        bias_GELU<<<block_num,block_dim,0,stream>>>((half *)d_y, (half *)d_b, K, B*T*K);
    else    
        bias<<<block_num,block_dim,0,stream>>>((half *)d_y, (half *)d_b, K, B*T*K);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return;

}


/**
 * @brief 
 * d_b has to be a matrix in col-major! (required by cuBLAS for having a fused bias + epilogue working)
 * @param gelu 
 * @param memory_order: if true, the x matrix will be in row-major, other wise will be in col-major
 */
void fused_linear_layer(
    cublasLtHandle_t &handle, cudaStream_t & stream,
    u_int B, u_int T, u_int C, u_int K,
    void * d_x, void * d_fc, void * d_b, 
    void * y, bool gelu, bool memory_order
){

    cublasLt_matmul_desc matmul;

    create_cublasLt_linlay_desc(
        B, T, C, K,
        matmul, 
        gelu, memory_order
    );

    void * d_workspace;
    cublasLtMatmulAlgo_t algo = fetch_matmul_algos(handle, matmul, &d_workspace);

    CUBLAS_CHECK(cublasLtMatmul(
        handle, matmul.matmulDesc, &matmul.alpha,
        d_x, matmul.xDesc,
        d_fc, matmul.fcDesc,
        &matmul.beta,
        d_b, matmul.cDesc,
        y, matmul.yDesc,
        &algo, d_workspace, (size_t)MLP_WORKSPACE_SIZE, stream
    ));
    // CUDA_CHECK(cudaStreamSynchronize(stream)); //not necessary cause on the same stream
    return;
}

/*
Fused MLP
*/
void fused_gpu_mlp(
    cublasLtHandle_t & handle, cudaStream_t & stream,
    u_int B, u_int T, u_int C, u_int K,u_int M,
    void * d_x, void * d_fc1, void * d_h,void * d_b1, void * d_fc2, void * d_b2, 
    void * d_y
){
    fused_linear_layer(
        handle,stream, 
        B,  T,  C,  K,
        d_x, d_fc1, d_b1, d_h,
        true,true
    );
    fused_linear_layer(
        handle,stream,
        B,  T,  K,  M,
        d_h, d_fc2, d_b2, d_y,
        false, false
    );
    return;
}

void cuBLAS_test(cublasLtHandle_t & handle, cudaStream_t & stream){
    u_int M = 2, N = 3, K = 4;
    
    float a[M*K] = {1.0,1.0,1.0,1.0};//{1.0}; // {1.0,1.0}; 
    float b[N*K] = {1.0,1.0,1.0,1.0};//{1.0}; // {1.0,1.0,1.0,1.0,1.0};
    float c[M*N] = { 1.0,0.0,1.0,0.0,1.0,0.0};//{0.0}; // {1.0,1.0};
    float d[M*N] = {0.0}; // all zeros

    mtx A(a,M,K) ,B(b,K,N), C(c,M,N), D(d,M,N);

    void * d_A, * d_B,* d_C ,* d_D;
    CUDA_CHECK(cudaMalloc(&d_A,sizeof(half) * M * K)); cudaMemcpy(d_A, A.data, sizeof(half) * M * K, cudaMemcpyHostToDevice);
    CUDA_CHECK(cudaMalloc(&d_B,sizeof(half) * N * K)); cudaMemcpy(d_B, B.data, sizeof(half) * N * K, cudaMemcpyHostToDevice);
    CUDA_CHECK(cudaMalloc(&d_C,sizeof(half) * M * N)); cudaMemcpy(d_C, C.data, sizeof(half) * M * N, cudaMemcpyHostToDevice);
    CUDA_CHECK(cudaMalloc(&d_D,sizeof(half) * M * N)); cudaMemcpy(d_D, D.data, sizeof(half) * M * N, cudaMemcpyHostToDevice);

    
    // -- Descriptor creation --
    //-Matmul creation
    cublasLtMatmulDesc_t matmulDesc; CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmulDesc,MLP_COMPUTE_DATA_TYPE, CUDA_R_32F)); // Scale type CUDA_R_16F not supported(doc)

    cublasOperation_t Avalue = CUBLAS_OP_N;
    cublasOperation_t Bvalue = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA,&Avalue, sizeof(cublasOperation_t))); // do not transpose x
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB,&Bvalue, sizeof(cublasOperation_t))); // transpose fc
    //Transpose for C not supported(doc)
    
    //-Epilogue creation
    // cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_GELU;

    // cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));

    cublasLtMatrixLayout_t xDesc; CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&xDesc,MLP_DATA_TYPE,M,K,K)); 
    // cublasLtMatrixLayout_t xDesc; CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&xDesc,MLP_DATA_TYPE,B*T,C,B*T)); // col-major
    cublasLtMatrixLayout_t fcDesc; CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&fcDesc,MLP_DATA_TYPE,K,N,K));
    // cublasLtMatrixLayout_t cDesc; CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&cDesc,MLP_DATA_TYPE,B*T,K,K)); 
    // cublasLtMatrixLayout_t yDesc; CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&yDesc,MLP_DATA_TYPE,B*T,K,K));
    cublasLtMatrixLayout_t cDesc; CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&cDesc,MLP_DATA_TYPE,M,N,M)); // Specifically the leading dimension of C can be 0 to achieve row or column broadcast.
    cublasLtMatrixLayout_t yDesc; CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&yDesc,MLP_DATA_TYPE,M,N,M)); // have to be col-major to enable the epilouge(fused kernel)

    //-Layout setting
    cublasLtOrder_t row = CUBLASLT_ORDER_ROW, col = CUBLASLT_ORDER_COL;

    // CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(xDesc,CUBLASLT_MATRIX_LAYOUT_ORDER,&row,sizeof(cublasLtOrder_t))); // row-major
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(xDesc,CUBLASLT_MATRIX_LAYOUT_ORDER,&row,sizeof(cublasLtOrder_t)));
    
    // CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(fcDesc,CUBLASLT_MATRIX_LAYOUT_ORDER,&row,sizeof(cublasLtOrder_t)));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(fcDesc,CUBLASLT_MATRIX_LAYOUT_ORDER,&col,sizeof(cublasLtOrder_t)));
    
    // CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(cDesc,CUBLASLT_MATRIX_LAYOUT_ORDER,&row,sizeof(cublasLtOrder_t))); // Setting C & D row-major
    // CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(yDesc,CUBLASLT_MATRIX_LAYOUT_ORDER,&row,sizeof(cublasLtOrder_t)));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(cDesc,CUBLASLT_MATRIX_LAYOUT_ORDER,&col,sizeof(cublasLtOrder_t))); // Setting C & D col-major
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(yDesc,CUBLASLT_MATRIX_LAYOUT_ORDER,&col,sizeof(cublasLtOrder_t)));
    // cout << "Finished Layout" << endl;

    // -- alpha and beta definition --
    // half alpha = __float2half(1.0), beta = __float2half(0.0);
    float alpha = 1.0f, beta = 1.0f;
    CUBLAS_CHECK(cublasLtMatmul(
        handle, matmulDesc, &alpha,
        d_A, xDesc,
        d_B, fcDesc,
        &beta,
        d_C, cDesc,
        d_D, yDesc,
        nullptr, nullptr, 0, stream
    ));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    half * y = (half *)malloc(sizeof(half) * M * N);
    
    CUDA_CHECK(cudaMemcpy(y, d_D, sizeof(half) * M * N, cudaMemcpyDeviceToHost));
    for(u_int i = 0; i < M * N; i++){
        if(i % N == 0) cout << endl;
        cout << " " << __half2float(y[i]) << " ";
    }
    return;
}

