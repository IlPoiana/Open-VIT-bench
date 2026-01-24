#include "../include/mlp.h"
#include "../gpu_include/gpu_mlp.h"

#include <iostream>

using namespace std;
// ----

// Simple error-check helpers
#define CHECK_CUDA(call) do {                                  \
    cudaError_t e = (call);                                    \
    if (e != cudaSuccess) {                                    \
        fprintf(stderr, "CUDA error %s:%d: %s\n",              \
                __FILE__, __LINE__, cudaGetErrorString(e));    \
    }                                                          \
} while(0)

#define CHECK_CUBLASLT(call) do {                              \
    cublasStatus_t s = (call);                                 \
    if (s != CUBLAS_STATUS_SUCCESS) {                          \
        fprintf(stderr, "cuBLASLt error %s:%d: %d\n",          \
                __FILE__, __LINE__, (int)s);                   \
    }                                                          \
} while(0)

// Helper to print cudaDataType_t
const char* cudaDataTypeToStr(cudaDataType_t t) {
    switch (t) {
    case CUDA_R_32F: return "CUDA_R_32F";
    case CUDA_R_64F: return "CUDA_R_64F";
    case CUDA_R_16F: return "CUDA_R_16F";
    case CUDA_R_8I:  return "CUDA_R_8I";
    case CUDA_R_8U:  return "CUDA_R_8U";
    case CUDA_C_32F: return "CUDA_C_32F";
    case CUDA_C_64F: return "CUDA_C_64F";
    case CUDA_C_16F: return "CUDA_C_16F";
    default: return "UNKNOWN";
    }
}

void print_layout_attributes(cublasLtMatrixLayout_t layout) {
    cublasStatus_t status;
    size_t got = 0;

    // Data type
    cudaDataType_t dtype;
    status = cublasLtMatrixLayoutGetAttribute(layout,
                                              CUBLASLT_MATRIX_LAYOUT_TYPE,
                                              &dtype, sizeof(dtype),
                                              &got);
    if (status == CUBLAS_STATUS_SUCCESS) {
        printf("TYPE: %s\n", cudaDataTypeToStr(dtype));
    } else {
        printf("TYPE: <not available> (status=%d)\n", (int)status);
    }

    // Rows
    int rows = 0;
    status = cublasLtMatrixLayoutGetAttribute(layout,
                                              CUBLASLT_MATRIX_LAYOUT_ROWS,
                                              &rows, sizeof(rows),
                                              &got);
    if (status == CUBLAS_STATUS_SUCCESS) {
        printf("ROWS: %d\n", rows);
    } else {
        printf("ROWS: <not available> (status=%d)\n", (int)status);
    }

    // Cols
    int cols = 0;
    status = cublasLtMatrixLayoutGetAttribute(layout,
                                              CUBLASLT_MATRIX_LAYOUT_COLS,
                                              &cols, sizeof(cols),
                                              &got);
    if (status == CUBLAS_STATUS_SUCCESS) {
        printf("COLS: %d\n", cols);
    } else {
        printf("COLS: <not available> (status=%d)\n", (int)status);
    }

    // Batch count
    int batchCount = 0;
    status = cublasLtMatrixLayoutGetAttribute(layout,
                                              CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                              &batchCount, sizeof(batchCount),
                                              &got);
    if (status == CUBLAS_STATUS_SUCCESS) {
        printf("BATCH_COUNT: %d\n", batchCount);
    } else {
        printf("BATCH_COUNT: <not available> (status=%d)\n", (int)status);
    }

    // Strided batch offset (may be 64-bit)
    int64_t stride = 0;
    status = cublasLtMatrixLayoutGetAttribute(layout,
                                              CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                              &stride, sizeof(stride),
                                              &got);
    if (status == CUBLAS_STATUS_SUCCESS) {
        printf("STRIDED_BATCH_OFFSET: %" PRId64 "\n", stride);
    } else {
        printf("STRIDED_BATCH_OFFSET: <not available> (status=%d)\n", (int)status);
    }

    // Order: depending on cuBLASLt version this might be an enum; try to fetch as int
    int order = 0;
    status = cublasLtMatrixLayoutGetAttribute(layout,
                                              CUBLASLT_MATRIX_LAYOUT_ORDER,
                                              &order, sizeof(order),
                                              &got);
    if (status == CUBLAS_STATUS_SUCCESS) {
        printf("ORDER (numeric): %d\n", order);
    } else {
        printf("ORDER: <not available> (status=%d)\n", (int)status);
    }

    // TILE or other implementation-specific attributes can be queried similarly:
    // e.g. CUBLASLT_MATRIX_LAYOUT_TILE, CUBLASLT_MATRIX_LAYOUT_ALIGNMENT, etc.
}

//----

// from A: MxN (row-major) to B: NxM (row-major)
template <class T>
void transpose_out_of_place(const T* A, T* B, std::size_t M, std::size_t N) {
    for (std::size_t i = 0; i < M; ++i) {
        const T* Ai = A + i * N;
        for (std::size_t j = 0; j < N; ++j) {
            B[j * M + i] = Ai[j];
        }
    }
}

// Returns the MRE of the cpu `y` Tensor and `gpu_y`. Attention! There is a tolerance instroduced to avoid division by zero
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

// Two implementation comparison
void cpu_gpu_comparison(bool debug = false){

    u_int B = 32,T = 196,C = 768,K = 3072,M = 768;
    if(debug){
        B = 4,T = 16,C = 64,K = 16,M = 8;
    }
    cout << "Tensor: [" << B << ","<< T << "," << C << "]" << endl;
    cout << "fc1: [" << C << ","<< K << "]" << endl;
    cout << "fc2: [" << K << ","<< M << "]" << endl;

    // -Host allocation
    float * x_data, * b1_data,* b2_data,* A1_data,* A2_data;
    u_int input_elements_number = B * T * C;
    u_int hidden_elements_number = B * T * K;
    u_int output_elements_number = B * T * M;
    
    x_data = (float*)malloc(sizeof(float) * input_elements_number);
    b1_data = (float*)malloc(sizeof(float) * K);
    b2_data = (float*)malloc(sizeof(float) * M);
    A1_data = (float*)malloc(sizeof(float) * C * K);
    A2_data = (float*)malloc(sizeof(float) * M * K);

    // -Random initialization

    random_device rd;          
    mt19937 gen(rd());         
    uniform_real_distribution<float> dist(-0.1f, 0.1f);

    size_t loop_range = max({input_elements_number, K * C, K * M});
    for(size_t i = 0; i < loop_range; i++){
        if(i < input_elements_number){
            x_data[i] = dist(gen);
        }
        if(i < C *K){
            A1_data[i] = dist(gen);
        }
        if(i < K){
            b1_data[i] = dist(gen);
        }
        if(i < K * M){
            A2_data[i] = dist(gen);
        }
        if(i < M){
            b2_data[i] = dist(gen);
        }

    }

    void * d_x,
    * d_b1_data, * d_b1_mtx,* d_b2_data, * d_b2_mtx,
    * d_fc1, * d_fc2, * d_h,
    * d_y;
    
    h_tensor x_gpu(x_data,B,C,1,T);
    mtx b1_gpu(b1_data,1,K);
    mtx b2_gpu(b2_data,1,M);
    vector<half> b1_gpu_mtx(K * B * T); bias_matrix(b1_gpu.data, b1_gpu_mtx.data(), K, B*T);
    vector<half> b2_gpu_mtx(M * B * T); bias_matrix(b2_gpu.data, b2_gpu_mtx.data(), M, B*T);
    mtx fc1_gpu(A1_data,K,C);
    mtx fc2_gpu(A2_data,M,K);
    half * h_gpu = (half *)malloc(sizeof(half) * B * T * K);
    half * y_gpu = (half *)malloc(sizeof(half) * B * T * M); 
    half * y_gpu_fused = (half *)malloc(sizeof(half) * B * T * M); 

    // -First layer
    CUDA_CHECK(cudaMalloc(&d_x, sizeof(half) * input_elements_number));
    CUDA_CHECK(cudaMalloc(&d_fc1, sizeof(half) * K * C));
    CUDA_CHECK(cudaMalloc(&d_b1_data, sizeof(half) * K));
    CUDA_CHECK(cudaMalloc(&d_b1_mtx, sizeof(half) * hidden_elements_number));
    CUDA_CHECK(cudaMalloc(&d_h, sizeof(half) * hidden_elements_number)); // for now, then will have different shape
    
    CUDA_CHECK(cudaMemcpy(d_x, x_gpu.data, sizeof(half) *  input_elements_number, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fc1, fc1_gpu.data, sizeof(half) * K *C, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1_data, b1_gpu.data, sizeof(half) * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1_mtx, b1_gpu_mtx.data(), sizeof(half) * hidden_elements_number, cudaMemcpyHostToDevice));

    // -Second layer
    CUDA_CHECK(cudaMalloc(&d_fc2, sizeof(half) * M * K));
    CUDA_CHECK(cudaMalloc(&d_b2_data, sizeof(half) * M));
    CUDA_CHECK(cudaMalloc(&d_b2_mtx, sizeof(half) * output_elements_number));
    CUDA_CHECK(cudaMalloc(&d_y, sizeof(half) * output_elements_number)); // for now, then will have different shape
    
    CUDA_CHECK(cudaMemcpy(d_fc2, fc2_gpu.data, sizeof(half) * M * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2_data, b2_gpu.data, sizeof(half) * M, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2_mtx, b2_gpu_mtx.data(), sizeof(half) * output_elements_number, cudaMemcpyHostToDevice));
    
    // -Handle creation
    cublasLtHandle_t handle;CUBLAS_CHECK(cublasLtCreate(&handle));
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    // -- CPU reference --
    Matrix A1(A1_data, K*C, K, C);
    RowVector b1(b1_data, K);

    Matrix A2(A2_data, M*K, M, K);
    RowVector b2(b2_data, M);

    Linear fc1(C, K, true);
    fc1.move_A(A1);
    fc1.move_b(b1);
    
    Linear fc2(K, M, true);
    fc2.move_A(A2);
    fc2.move_b(b2);
    
    Mlp mlp(C, K, M, GELU, true, false);
    mlp.move_fc1(fc1);
    mlp.move_fc2(fc2);

    Tensor x(x_data, B*T*C, B, T, C);
    Tensor y;
    
    mlp.forward(x, y);
    cout << "CPU MLP REFERENCE: " << endl;
    if(debug) y.print();
    
    // -- GPU -- 
    // -GPU non-fused
    gpu_mlp(
        handle, stream1,
        B,T,C,K,M,
        d_x, d_fc1, d_h,d_b1_data, d_fc2,d_b2_data,d_y
    );

    CUDA_CHECK(cudaMemcpy(y_gpu, d_y, sizeof(half) * output_elements_number, cudaMemcpyDeviceToHost));

    // -GPU fused
    
    // -Reset d_h and d_y buffers
    CUDA_CHECK(cudaMemsetAsync(d_h, 0, sizeof(half) * hidden_elements_number, stream1));
    CUDA_CHECK(cudaMemsetAsync(d_y, 0, sizeof(half) * output_elements_number, stream1));

    // -Create descriptors for fused MLP
    cublasLt_matmul_desc matmul[2];
    cublasLtMatmulAlgo_t algo[2];
    mlp_dimensions dim(B, T, C, K, M);

    void *d_workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&d_workspace, WORKSPACE_SIZE));

    create_mlp_descriptors(
        handle,
        matmul,
        d_workspace,
        algo,
        dim,
        true
    );
    fused_gpu_mlp(
        handle,
        stream1,
        matmul, algo, 
        d_workspace,
        d_x, d_fc1, d_h, d_b1_mtx, d_fc2, d_b2_mtx, d_y
    );

    half * temp_gpu = (half *)malloc(sizeof(half) * output_elements_number);
    CUDA_CHECK(cudaMemcpy(temp_gpu,d_y, sizeof(half) * output_elements_number, cudaMemcpyDeviceToHost));
    transpose_out_of_place(temp_gpu, y_gpu_fused, M,B*T);

    if(debug){
        vector<float> gpu_host(output_elements_number);
        f16_to_f32(y_gpu, gpu_host.data(),output_elements_number);
        Tensor out(gpu_host.data(), output_elements_number, B, T, M);
        f16_to_f32(y_gpu_fused, gpu_host.data(),output_elements_number);
        Tensor out_fused(gpu_host.data(), output_elements_number, B, T, M);
        cout << "GPU MLP" << endl; out.print();
        cout << "GPU-FUSED MLP" << endl; out_fused.print();
    }

    // -- Comparison --
    // -CPU/GPU
    cout << "avg difference for CPU/GPU MLP: " << compare_results(y,y_gpu) * 100 << "%" << endl;
    cout << "avg difference for CPU/GPU-FUSED MLP: " << compare_results(y, y_gpu_fused) * 100 << "%" << endl;

    // -Cleanup
    free(x_data); free(b1_data); free(b2_data); free(A1_data); free(A2_data);
    free(h_gpu); free(y_gpu); free(y_gpu_fused); free(temp_gpu);
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_b1_data)); CUDA_CHECK(cudaFree(d_b1_mtx));
    CUDA_CHECK(cudaFree(d_b2_data)); CUDA_CHECK(cudaFree(d_b2_mtx));
    CUDA_CHECK(cudaFree(d_fc1)); CUDA_CHECK(cudaFree(d_fc2));
    CUDA_CHECK(cudaFree(d_h));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_workspace));

    matmul[0].destroy_descriptors();
    matmul[1].destroy_descriptors();
    CUBLAS_CHECK(cublasLtDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream1));
}

int main() {
    bool debug = false;
    cpu_gpu_comparison(debug);

    return 0;
}