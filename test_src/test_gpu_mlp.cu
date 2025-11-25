#include "../include/mlp.h"
#include "../gpu_include/gpu_mlp.h"

#include <iostream>

using namespace std;
// ----
#include <cstddef>

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

double compare_results(Tensor &y, half * gpu_y){
    double avg = 0;
    for(u_int b = 0; b < y.get_B(); b++){
        for(u_int t = 0; t < y.get_N(); t++){
            for(u_int c = 0; c < y.get_C(); c++){
                assert(!isnanf( y.at(b,t,c)));
                assert(!isnanf( __half2float(gpu_y[c + y.get_C() * t + y.get_C() * y.get_N() * b])));
                avg += (double)abs(y.at(b,t,c) - __half2float(gpu_y[c + y.get_C() * t + y.get_C() * y.get_N() * b]));
                
            }
        }
    }
    return avg / (double(y.get_B()) * y.get_N() * y.get_C());
}

void cpu_gpu_comparison(){
    cout << "Test Mlp" << endl;

    u_int B = 2, N = 8, C=6;
    u_int K = 10, M = 8;

    vit_float A1_data[K*C] = {
        -0.456297, 0.451657, 0.790088, -0.792936, -0.640623, 0.283185,
        -0.686055, 0.276620, 0.659866, -0.011135, 0.430428, -0.378445,
        -0.620894, 0.601418, -0.575021, 0.246048, 0.333867, 0.860185,
        -0.463988, -0.217379, 0.652214, -0.578656, -0.905936, 0.707247,
        -0.708495, 0.267307, -0.129294, 0.521801, 0.373023, -0.193839,
        0.342945, 0.872465, -0.463590, 0.324824, 0.389333, 0.217408,
        -0.490482, -0.828896, 0.133649, -0.049476, -0.769367, 0.159828,        
        0.599725, -0.938836, 0.197874, -0.021849, -0.227208, -0.912308,
        -0.285737, -0.105809, 0.647727, -0.029205, 0.209804, 0.876799,
        -0.006798, -0.411250, -0.551676, 0.055781, -0.335824, -0.228423
    };
    Matrix A1(A1_data, K*C, K, C);
    cout << "### A1" << endl;
    A1.print();

    vit_float b1_data[K] = {
        -0.019805, -0.530365, -0.815562, -0.535694, -0.774685, -0.847759, -0.198200, -0.784896, -0.147666, 0.603477  };
    for (size_t i = 0; i < K; i++)
    {
        b1_data[i] = 0.0;
    }
    b1_data[0] = 1.0; b1_data[4] = 1.0; b1_data[9] = 1.0; 
    
    RowVector b1(b1_data, K);
    cout << "### b1" << endl;
    b1.print();

    vit_float A2_data[M*K] = {
         -2.005, -14.575,  17.934, -29.395,  -5.142,  28.463,  32.815, -74.448,  76.309,   0.199,
        -32.319, -50.704,  79.610, -53.554, -59.941,  -4.564,   7.415,  50.209, -28.249,  67.815,
         75.023,  99.586,  25.779,  -9.515, -87.194, -93.804, -68.875, -33.723,  78.107,  76.215,
         62.282,  10.427, -45.408,  16.962,  48.725,  -8.866, -68.867,  78.927, -58.144, -99.136,
         28.442,  19.411, -83.094, -53.910,  63.968,  13.114,  12.642, -64.282,  79.999,  95.254,
        -86.990, -49.479, -26.197,  21.675,  96.535, -37.169,  56.591, -90.600, -62.138,  39.213,
         48.827, -44.107, -42.021, -58.035,  40.707,  39.406,  34.763,  25.091, -65.111, -81.942,
        -51.952, -84.394,  51.219, -83.677,  -8.525,  43.929, -61.486, -13.540, -54.732, -62.259
    };
    Matrix A2(A2_data, M*K, M, K);
    cout << "### A2" << endl;
    A2.print();

    vit_float b2_data[M] = {-55.690, 61.838, -25.379, 95.026, 2.756, 12.244, 85.241, 7.426};
    RowVector b2(b2_data, M);
    cout << "### b2" << endl;
    b2.print();

    Linear fc1(C, K, true);
    fc1.move_A(A1);
    fc1.move_b(b1);
    Linear fc2(K, M, true);
    fc2.move_A(A2);
    fc2.move_b(b2);

    // WATCH OUT FOR LAYER NORM!!
    // Mlp mlp(5, 10, 8, GELU, true, true);

    Mlp mlp(C, K, M, GELU, true, false);

    mlp.move_fc1(fc1);
    mlp.move_fc2(fc2);

    
    vit_float x_data[2*8*6] = {
        -0.598662, 0.123939, -0.735337, 0.773240, 0.341039, 0.255914,
        0.377830, 0.697902, -0.924890, -0.460011, -0.481212, -0.175776,
        0.847194, -0.752708, -0.968637, -0.850046, 0.103257, 0.557820,
        -0.210567, 0.733842, 0.299217, 0.915666, 0.908730, -0.499567,
        -0.431871, 0.845170, 0.151241, 0.887892, 0.702863, 0.331856,
        -0.292996, 0.328416, 0.983022, 0.165157, -0.614076, -0.450108,
        0.993807, 0.117445, 0.354478, 0.561162, -0.811542, -0.138070,
        -0.409118, -0.105013, -0.581583, 0.752852, 0.035026, 0.046210,

        -0.867359, -0.518816, 0.730830, -0.101440, 0.084215, -0.751911,
        -0.420023, -0.573539, -0.692645, 0.113312, 0.459171, -0.263308,
        -0.255053, -0.417768, 0.120394, -0.031679, 0.956135, -0.219269,
        -0.528727, -0.326731, 0.118325, 0.593465, 0.945926, -0.651224,
        0.142245, -0.855496, 0.559943, -0.023597, 0.202991, 0.357500,
        -0.066550, 0.529705, -0.143649, 0.811908, 0.955749, 0.985978,
        0.699356, -0.080135, -0.007364, 0.197332, -0.565641, 0.689737,
        0.678729, -0.467943, -0.269886, -0.945391, 0.689606, -0.224848,
    };

    Tensor x(x_data, B*N*C, B, N, C);
    cout << "### x" << endl;
    x.print();
    
    // B, N, C
    Tensor x_in(x_data, B*N*C, B, N, C);
    // B, N, K
    Tensor input_layer(B,N,K);
    Linear test_fc1(C, K, true);
    
    Matrix t_A1(A1_data, K*C, K, C);
    
    RowVector t_B1(b1_data, K);

    test_fc1.move_A(t_A1);
    test_fc1.move_b(t_B1);
    cout << "CPU INPUT LAYER REFERENCE: " << endl;
    test_fc1(x_in, input_layer);

    input_layer.print();

    bool gelu = true;
    float cpu_gelu[B*N*K];
    if(gelu)
    {    
        cout << "CPU GELU REFERENCE: " << endl;
        for(u_int b = 0; b < B; b++){
            cout << "[" << endl;
            for(u_int t = 0; t < N; t++){
                cout << "[";
                for(u_int c = 0; c < K; c++){
                    cpu_gelu[b * N * K + t*K + c] = GELU(input_layer.at(b,t,c));
                    cout << " " << cpu_gelu[b * N * K + t*K + c] << " ";
                }
                cout << "]"<<endl;
            }
            cout << "]" << endl;
        }
    }
    
    


    Tensor y;
    mlp.forward(x, y);
    cout << "CPU MLP REFERENCE: " << endl;
    y.print();


    // ---
    // GPU ONLY VARIABLES
    // --


    h_tensor x_gpu(x_data,B,C,1,N);

    mtx b1_gpu(b1_data,1,K); mtx b1_gpu_mtx(K, B*N); bias_matrix(b1_gpu.data, b1_gpu_mtx.data, K, B*N);
    mtx b2_gpu(b2_data,1,M); mtx b2_gpu_mtx(M, B*N); bias_matrix(b2_gpu.data, b2_gpu_mtx.data, M, B*N);
    mtx fc1_gpu(A1_data,K,C);
    mtx fc2_gpu(A2_data,M,K);
    half * h_gpu = (half *)malloc(sizeof(half) * B * N * K);
    half * y_gpu = (half *)malloc(sizeof(half) * B * N * M); // TO MALLOC!!

    //-Device allocation
    void * d_x,
    * d_b1_data, * d_b1_mtx,* d_b2_data, * d_b2_mtx,
    * d_fc1, * d_fc2, * d_h,
    * d_y;
    //First layer
    CUDA_CHECK(cudaMalloc(&d_x, sizeof(half) * B * N * C));CUDA_CHECK(cudaMemcpy(d_x, x_gpu.data, sizeof(half) * B * N *C, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_fc1, sizeof(half) * K * C));CUDA_CHECK(cudaMemcpy(d_fc1, fc1_gpu.data, sizeof(half) * K *C, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_b1_data, sizeof(half) * K));CUDA_CHECK(cudaMemcpy(d_b1_data, b1_gpu.data, sizeof(half) * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_b1_mtx, sizeof(half) * B * N * K));CUDA_CHECK(cudaMemcpy(d_b1_mtx, b1_gpu_mtx.data, sizeof(half) * B * N * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_h, sizeof(half) * B * N * K)); // for now, then will have different shape
    //Second layer
    CUDA_CHECK(cudaMalloc(&d_fc2, sizeof(half) * M * K));CUDA_CHECK(cudaMemcpy(d_fc2, fc2_gpu.data, sizeof(half) * M * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_b2_data, sizeof(half) * M));CUDA_CHECK(cudaMemcpy(d_b2_data, b2_gpu.data, sizeof(half) * M, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_b2_mtx, sizeof(half) * B * N * M));CUDA_CHECK(cudaMemcpy(d_b2_mtx, b2_gpu_mtx.data, sizeof(half) * B * N * M, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_y, sizeof(half) * B * N * M)); // for now, then will have different shape


    //-Handle creation
    cublasLtHandle_t handle;CUBLAS_CHECK(cublasLtCreate(&handle));
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    
    bool test = false;
    if(test){
        void cuBLAS_test(cublasLtHandle_t & handle, cudaStream_t & stream);
        return;
    }

    gpu_mlp(
        handle, stream1,
        B,N,C,K,M,
        d_x, d_fc1, d_h,d_b1_data, d_fc2,d_b2_data,d_y
    );
    CUDA_CHECK(cudaMemcpy(y_gpu,d_y, sizeof(half) * B * N * M, cudaMemcpyDeviceToHost));

    vector<float> host_gpu(B * N * M);
    f16_to_f32(y_gpu, host_gpu.data(), B * N * M);
    Tensor out(host_gpu.data(),B * N * M, B, N, M);
    cout << "GPU MLP" << endl; out.print();
    cout << "avg difference for CPU/GPU MLP: "  << compare_results(y, y_gpu) << endl;
    
    // -- FUSED GPU MLP --
    CUDA_CHECK(cudaMemset(d_h, 0, sizeof(half) * B * N * K));
    CUDA_CHECK(cudaMemset(d_y, 0, sizeof(half) * B * N * M));

    fused_gpu_mlp(
        handle, stream1,
        B,N,C,K,M,
        d_x, d_fc1, d_h,d_b1_mtx, d_fc2,d_b2_mtx,d_y
    );
    half temp_gpu[B*N*M];
    CUDA_CHECK(cudaMemcpy(temp_gpu,d_y, sizeof(half) * B * N * M, cudaMemcpyDeviceToHost));
    transpose_out_of_place(temp_gpu, y_gpu, M,B*N);

    f16_to_f32(y_gpu, host_gpu.data(), B * N * M);
    Tensor out_fused(host_gpu.data(),B * N * M, B, N, M);
    cout << "GPU MLP" << endl; out_fused.print();
    cout << "avg difference for CPU/GPU-FUSED MLP: " << compare_results(y, y_gpu) << endl;

    return;
}

// Two implementation comparison
void gpu_comparison(){
    bool debug = true;

    u_int B = 32,T = 196,C = 768,K = 3072,M = 768;
    if(debug){
        B = 4,T = 16,C = 64,K = 16,M = 8;
    }
    cout << "Tensor: [" << B << ","<< T << "," << C << "]" << endl;
    cout << "fc1: [" << C << ","<< K << "]" << endl;
    cout << "fc2: [" << K << ","<< M << "]" << endl;

    //-Host allocation
    float * x_data, * b1_data,* b2_data,* A1_data,* A2_data;
    u_int input_elements_number = B * T * C;
    u_int hidden_elements_number = B * T * K;
    u_int output_elements_number = B * T * M;
    
    x_data = (float*)malloc(sizeof(float) * input_elements_number);
    b1_data = (float*)malloc(sizeof(float) * K);
    b2_data = (float*)malloc(sizeof(float) * M);
    A1_data = (float*)malloc(sizeof(float) * C * K);
    A2_data = (float*)malloc(sizeof(float) * M * K);

    //-Device allocation
    float * d_ref, * d_a1,* d_a2;
    CUDA_CHECK(cudaMalloc(&d_ref, sizeof(float) * input_elements_number));
    CUDA_CHECK(cudaMalloc(&d_a1, sizeof(float) * C * K));
    CUDA_CHECK(cudaMalloc(&d_a2, sizeof(float) * K * M));

    //Reference generation
    //-x
    // cout << "x" << endl;
    u_int block_size = 256;
    u_int block_num = (input_elements_number/ block_size) + 1; 
    generate_reference<<<block_num, block_size>>>(d_ref, input_elements_number);
    CUDA_CHECK(cudaMemcpy(x_data, d_ref, sizeof(float) * input_elements_number, cudaMemcpyDeviceToHost));
    if(debug){
        Tensor tmp(x_data, input_elements_number, B, T, C);
        cout << "x" << endl; tmp.print();
    }

    //-bias fc1
    // cout << "bias" << endl;
    block_num = (K/ block_size) + 1; 
    generate_reference<<<block_num, block_size>>>(d_ref,K);
    CUDA_CHECK(cudaMemcpy(b1_data, d_ref, sizeof(float) * K, cudaMemcpyDeviceToHost));
    if(debug){
        RowVector tmp(b1_data, K);
        cout << "b1" << endl; tmp.print();
    }

    //-fc1
    // cout << "fc1" << endl;
    block_num = ((C * K )/ block_size) + 1; 
    generate_reference<<<block_num, block_size>>>(d_a1, C * K);
    CUDA_CHECK(cudaMemcpy(A1_data,d_a1 , sizeof(float) * C * K, cudaMemcpyDeviceToHost));
    if(debug){
        Matrix tmp(A1_data, C*K, C, K);
        cout << "fc1" << endl; tmp.print();
    }

    //-bias fc2
    block_num = (M/ block_size) + 1; 
    generate_reference<<<block_num, block_size>>>(d_ref,M);
    CUDA_CHECK(cudaMemcpy(b2_data, d_ref, sizeof(float) * M, cudaMemcpyDeviceToHost));
    if(debug){
        RowVector tmp(b2_data, M);
        cout << "b2" << endl; tmp.print();
    }

    //-fc2
    // cout << "fc2" << endl;
    block_num = ((K *M)/ block_size) + 1; 
    generate_reference<<<block_num, block_size>>>(d_a2, K * M);
    CUDA_CHECK(cudaMemcpy(A2_data, d_a2, sizeof(float) * K * M, cudaMemcpyDeviceToHost));
    if(debug){
        Matrix tmp(A2_data, K*M, K, M);
        cout << "fc2" << endl; tmp.print();
    }

    void * d_x,
    * d_b1_data, * d_b1_mtx,* d_b2_data, * d_b2_mtx,
    * d_fc1, * d_fc2, * d_h,
    * d_y;
    
    // cout << "gpu host variables" << endl;
    h_tensor x_gpu(x_data,B,C,1,T);
    mtx b1_gpu(b1_data,1,K); mtx b1_gpu_mtx(K, B*T); bias_matrix(b1_gpu.data, b1_gpu_mtx.data, K, B*T);
    mtx b2_gpu(b2_data,1,M); mtx b2_gpu_mtx(M, B*T); bias_matrix(b2_gpu.data, b2_gpu_mtx.data, M, B*T);
    mtx fc1_gpu(A1_data,K,C);
    mtx fc2_gpu(A2_data,M,K);
    half * h_gpu = (half *)malloc(sizeof(half) * B * T * K);
    half * y_gpu = (half *)malloc(sizeof(half) * B * T * M); 
    half * y_gpu_fused = (half *)malloc(sizeof(half) * B * T * M); 
    // cout << "starting gpu malloc" << endl;

    //First layer
    CUDA_CHECK(cudaMalloc(&d_x, sizeof(half) * input_elements_number));CUDA_CHECK(cudaMemcpy(d_x, x_gpu.data, sizeof(half) *  input_elements_number,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_fc1, sizeof(half) * K * C));CUDA_CHECK(cudaMemcpy(d_fc1, fc1_gpu.data, sizeof(half) * K *C, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_b1_data, sizeof(half) * K));CUDA_CHECK(cudaMemcpy(d_b1_data, b1_gpu.data, sizeof(half) * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_b1_mtx, sizeof(half) * hidden_elements_number));CUDA_CHECK(cudaMemcpy(d_b1_mtx, b1_gpu_mtx.data, sizeof(half) * hidden_elements_number, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_h, sizeof(half) * hidden_elements_number)); // for now, then will have different shape
    //Second layer
    CUDA_CHECK(cudaMalloc(&d_fc2, sizeof(half) * M * K));CUDA_CHECK(cudaMemcpy(d_fc2, fc2_gpu.data, sizeof(half) * M * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_b2_data, sizeof(half) * M));CUDA_CHECK(cudaMemcpy(d_b2_data, b2_gpu.data, sizeof(half) * M, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_b2_mtx, sizeof(half) * output_elements_number));CUDA_CHECK(cudaMemcpy(d_b2_mtx, b2_gpu_mtx.data, sizeof(half) * output_elements_number, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_y, sizeof(half) * output_elements_number)); // for now, then will have different shape

    //-Handle creation
    cublasLtHandle_t handle;CUBLAS_CHECK(cublasLtCreate(&handle));
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    //-CPU reference
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
    //-GPU 
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start); cudaEventCreate(&stop);

    // cudaEventRecord(start, stream1);
    gpu_mlp(
        handle, stream1,
        B,T,C,K,M,
        d_x, d_fc1, d_h,d_b1_data, d_fc2,d_b2_data,d_y
    );
    // cudaEventRecord(stop, stream1);

    CUDA_CHECK(cudaMemcpy(y_gpu, d_y, sizeof(half) * output_elements_number, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaStreamSynchronize(stream1)); // This should ensure that all the work on my stream has finished before memsetting it
    //-GPU Fused
    // cout << "Fused" << endl;
    CUDA_CHECK(cudaMemsetAsync(d_h, 0, sizeof(half) * hidden_elements_number, stream1));
    CUDA_CHECK(cudaMemsetAsync(d_y, 0, sizeof(half) * output_elements_number, stream1));

    //TESTING
    cublasLt_matmul_desc matmul[2];
    cublasLtMatmulAlgo_t algo[2];
    void * d_workspace; cudaMalloc(&d_workspace, (size_t) MLP_WORKSPACE_SIZE);
    mlp_dimensions dim(B,T,C,K,M);
    create_mlp_descriptors(handle, matmul, d_workspace, algo, dim);
    // cout << "descriptors fine" << endl;
    fused_gpu_mlp(
        handle, stream1,
        matmul, algo, d_workspace,
        d_x, d_fc1, d_h,d_b1_mtx, d_fc2,d_b2_mtx,d_y
    );
    // cout << "fused op fine" << endl;
    //----
    half * temp_gpu = (half *)malloc(sizeof(half) * output_elements_number);
    CUDA_CHECK(cudaMemcpy(temp_gpu,d_y, sizeof(half) * output_elements_number, cudaMemcpyDeviceToHost));
    transpose_out_of_place(temp_gpu, y_gpu_fused, M,B*T);
    // cout << "transpose ok" << endl;
    if(debug){
        vector<float> gpu_host(output_elements_number);
        f16_to_f32(y_gpu, gpu_host.data(),output_elements_number);
        Tensor out(gpu_host.data(), output_elements_number, B, T, M);
        f16_to_f32(y_gpu_fused, gpu_host.data(),output_elements_number);
        Tensor out_fused(gpu_host.data(), output_elements_number, B, T, M);
        cout << "GPU MLP" << endl; out.print();
        cout << "GPU-FUSED MLP" << endl; out_fused.print();
    }
    //-- Comparison --
    //-CPU/GPU
    cout << "avg difference for CPU/GPU MLP: " << compare_results(y,y_gpu) << endl;
    cout << "avg difference for CPU/GPU-FUSED MLP: " << compare_results(y, y_gpu_fused) << endl;
}

int main() {

    test_type test = GPU_COMPARISON;
    if(test == CPU_COMPARISON){
        cpu_gpu_comparison();
    }
    else{
        gpu_comparison();
    }

    return 0;
}