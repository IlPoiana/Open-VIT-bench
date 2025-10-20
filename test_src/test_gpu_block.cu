#define TEST_BLOCK
#include "../include/block.h"
#include "../gpu_include/gpu_block.h"


#define B 2
#define T 7
#define C 9
#define K 10
#define M 9

#define LAYER_SCALE 0.00004
/*
1) Test the cpu implementation
2) Finish the GPU porting:
    2.1) Create an appropriate Attention function

*/

Tensor cpu_baseline(
    vit_float * x_data,
    blocks_data &block_data
){
    //Attention
    cout << "Test Block" << endl;

    // Attention Initialization
    attn_data attention_data = block_data.attention;
    vit_float * q_data = attention_data.q_gen.A;
    vit_float * k_data = attention_data.k_gen.A;
    vit_float * v_data = attention_data.v_gen.A;
    vit_float * qb_data = attention_data.q_gen.b;
    vit_float * kb_data = attention_data.k_gen.b;
    vit_float * vb_data = attention_data.v_gen.b;
    vit_float * p_data = attention_data.proj.A;
    vit_float * pb_data = attention_data.proj.b;
    cout << "Test Attention" << endl;
    u_int input_elements_number = B*T*C;
    u_int hidden_elements_number = B * T * K;
    u_int output_elements_number = B * T * M;
    u_int qkv_dimensions = C * C;
    u_int proj_dimensions = C * C;

    // Linear weights
    
    Matrix q(q_data, C*C, C, C);
    cout << "### q" << endl;
    q.print();

    Matrix k(k_data, C*C, C, C);
    cout << "### k" << endl;
    k.print();

    
    Matrix v(v_data, C*C, C, C);

    cout << "### v" << endl;
    v.print();

    // Biases
    RowVector qb(qb_data, C);
    cout << "### qb" << endl;
    qb.print();
    RowVector kb(kb_data, C);
    cout << "### kb" << endl;
    kb.print();
    
    RowVector vb(vb_data, C);
    cout << "### vb" << endl;
    vb.print();

    //final linear projection, to set back the right dimension
    
    Matrix p(p_data, C*C, C, C);
    cout << "### p" << endl;
    p.print();

    // bias for proj
    RowVector pb(pb_data, C);
    cout << "### pb" << endl;
    pb.print();

    //input test data batch 2 sequence 7 embeddings 9

    Tensor x(x_data, input_elements_number, B, T, C);
    cout << "### x" << endl;
    x.print();

    Attention attn(C, TEST_NUM_HEADS, false, false); //No layer norm

    Linear q_gen(C, C, true);
    q_gen.move_A(q);
    q_gen.move_b(qb);
    Linear k_gen(C, C, true);
    k_gen.move_A(k);
    k_gen.move_b(kb);
    Linear v_gen(C, C, true);
    v_gen.move_A(v);
    v_gen.move_b(vb);
    Linear proj(C, C, true);
    proj.move_A(p);
    proj.move_b(pb);

    attn.move_qkv_gen(q_gen, k_gen, v_gen);
    attn.move_proj(proj);

    //TO REMOVE
    // Tensor y;
    // attn.forward(x, y);
    // cout << "### y = attn(x)" << endl;
    // y.print();

    // cout << "avg. difference CPU/cuDNN GPU: "  << compare_results(y, host_out) << endl;
    //----

    // Mlp Initialization
    mlp_data mlp_cpu_data = block_data.mlp;
    vit_float * A1_data = mlp_cpu_data.fc1.A;
    vit_float * b1_data = mlp_cpu_data.fc1.b;
    vit_float * A2_data = mlp_cpu_data.fc2.A;
    vit_float * b2_data = mlp_cpu_data.fc2.b;
    Matrix A1(A1_data, K*C, K, C);
    cout << "### A1" << endl;
    A1.print();
    
    RowVector b1(b1_data, K);
    cout << "### b1" << endl;
    b1.print();
    
    Matrix A2(A2_data, M*K, M, K);
    cout << "### A2" << endl;
    A2.print();

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

    // Block Initialization
    vit_float * n1g_data = block_data.norm1.g;
    vit_float * n1b_data = block_data.norm1.bias;
    vit_float * n2g_data = block_data.norm2.g;
    vit_float * n2b_data = block_data.norm2.bias;
    
    RowVector n1g(n1g_data, C);
    cout << "### n1g" << endl;
    n1g.print();
    
    RowVector n1b(n1b_data, C);
    cout << "### n1b" << endl;
    n1b.print();

    
    RowVector n2g(n2g_data, C);
    cout << "### n2g" << endl;
    n2g.print();
    
    RowVector n2b(n2b_data, C);
    cout << "### n2b" << endl;
    n2b.print();

    LayerNorm block_n1(C, 0.00001, true);
    block_n1.move_g(n1g);
    block_n1.move_b(n1b);
    LayerNorm block_n2(C, 0.00001, true);
    block_n2.move_g(n2g);
    block_n2.move_b(n2b);

    cout << "### LayerScale is 0.00004" << endl << endl;
    //mlp_ratio is a utility variable only, it tells the K size of the mlp related to the C size
    Block blk(C, TEST_NUM_HEADS, 1.4, true, false, LAYER_SCALE, GELU);
    blk.move_attn(attn);
    blk.move_mlp(mlp);
    blk.move_norm1(block_n1);
    blk.move_norm2(block_n2);

    // Actual Test

    Tensor y;
    blk.forward(x, y);
    cout << "### y = encoder_block(x)" << endl;
    y.print();
    return y;
}

void cpu_gpu_comparison(){
    u_int input_elements_number = B*T*C;
    u_int hidden_elements_number = B * T * K;
    u_int output_elements_number = B * T * M;
    u_int qkv_dimensions = C * C;
    u_int proj_dimensions = C * C;
    double epsilon = 0.00001;
    //INput data
    vit_float x_data[input_elements_number] = {
        -0.703,  -0.155,   0.869,  -0.876,  -0.116,   0.148,  -0.865,  -0.431,  -0.442,
         0.335,   0.172,   0.187,  -0.907,   0.904,  -0.837,  -0.622,   0.454,  -0.883,
        -0.464,   0.737,  -0.623,   0.004,  -0.188,   0.945,  -0.351,  -0.552,   0.301,
         0.760,   0.513,   0.895,  -0.060,   0.869,  -0.955,  -0.402,  -0.071,  -0.962,
        -0.832,  -0.235,   0.727,  -0.438,   0.710,  -0.460,  -0.787,   0.725,  -0.743,
        -0.889,   0.982,   0.762,  -0.097,   0.207,  -0.988,  -0.610,   0.722,   0.416,
         0.038,  -0.314,   0.475,  -0.502,   0.638,  -0.355,  -0.609,  -0.231,  -0.272,

         0.533,  -0.786,  -0.958,   0.928,  -0.281,   0.966,   0.095,   0.865,   0.446,
        -0.912,   0.476,   0.139,  -0.204,  -0.546,   0.614,   0.496,   0.985,   0.227,
         0.564,  -0.304,  -0.318,   0.477,   0.369,   0.052,   0.042,   0.076,  -0.579,
         0.265,  -0.826,  -0.396,  -0.484,   0.481,  -0.182,   0.770,  -0.362,  -0.601,
         0.806,   0.902,  -0.803,   0.431,  -0.398,  -0.146,  -0.262,   0.269,  -0.887,
        -0.026,  -0.604,  -0.381,   0.490,   0.022,   0.256,  -0.408,  -0.321,   0.330,
        -0.307,  -0.789,  -0.262,   0.662,  -0.323,  -0.478,  -0.487,  -0.490,   0.682
    };
    //Attention data
    vit_float q_data[C*C] = {
        0.758294, -0.727619, -0.330454, -0.717287, 0.822706, 0.610953, 0.408179, -0.605157, 0.830348,
        0.725958, -0.930735, -0.986684, 0.783906, 0.338275, -0.561106, -0.696360, 0.845882, 0.276506,
        -0.160658, 0.383336, -0.710762, -0.617399, -0.453931, -0.943827, -0.591806, 0.115698, -0.888167,
        0.801469, 0.899521, 0.855709, -0.737493, -0.068429, 0.967107, -0.308915, -0.817668, 0.433885,
        -0.308675, 0.170276, -0.575701, 0.262051, -0.410352, -0.080728, -0.484885, 0.410175, -0.378071,
        0.314474, -0.118119, 0.414797, -0.462171, -0.016918, 0.933693, 0.147110, -0.290505, 0.882674,
        -0.199357, -0.571346, 0.476810, -0.203064, 0.831118, 0.407663, 0.628251, 0.219807, 0.679447,
        -0.065085, 0.621072, 0.756602, 0.452867, -0.496811, 0.373570, 0.940523, 0.667954, 0.114777,
        -0.227100, -0.061652, -0.829483, -0.519480, -0.965037, -0.877678, 0.884411, 0.592530, 0.336187
    };
    vit_float k_data[C*C] = {
        -0.792000, -0.479908, -0.742727, 0.377966, -0.344953, -0.129817, -0.147616, -0.880930, 0.847170,
        -0.696408, -0.702998, 0.543170, 0.600282, 0.489631, -0.951209, -0.576907, 0.217496, 0.328485,
        0.298473, 0.114261, -0.946716, 0.235580, -0.252250, -0.111759, -0.937504, 0.046617, 0.985900,
        0.725671, -0.527940, 0.731302, -0.606755, 0.391039, -0.037762, -0.675364, -0.995391, 0.807426,
        0.639046, -0.073534, 0.737945, -0.067340, -0.683074, -0.609855, -0.904569, -0.936620, 0.698799,
        -0.270607, 0.185978, -0.714229, 0.319039, -0.107037, 0.551522, -0.554689, -0.966164, 0.955272,
        -0.776169, 0.467594, -0.962677, -0.005187, -0.466437, 0.052940, -0.040200, 0.207045, -0.869164,
        -0.951610, -0.329654, 0.517901, 0.211466, 0.801962, -0.309224, -0.396201, 0.744334, 0.310957,
        -0.563039, -0.261428, 0.184488, 0.796891, -0.526099, 0.242236, -0.717917, 0.769103, -0.066658
    };
    vit_float v_data[C*C] = {
         0.338028, -0.293926, 0.000460, -0.655297, 0.360831, -0.032444, -0.152898, -0.825067, 0.057772,
        0.914789, 0.238221, 0.245677, -0.268523, 0.761661, -0.361399, -0.849006, -0.896083, 0.177261,
        0.317324, -0.640804, -0.302885, -0.263762, -0.932062, 0.859797, 0.674119, -0.529998, 0.081886,
        -0.310355, 0.015906, -0.621151, -0.122542, -0.071480, -0.218666, -0.447618, 0.622280, 0.105224,
        -0.947940, -0.441825, 0.904577, -0.823825, 0.958628, -0.199077, 0.438047, -0.575452, -0.144570,
        0.920906, -0.482354, 0.482662, -0.039985, -0.868218, -0.642136, -0.461837, 0.249278, -0.526823,
        0.447369, -0.111263, -0.080426, -0.515979, 0.362862, 0.712538, 0.787162, 0.356369, -0.451191,
        0.752151, 0.658250, -0.280474, -0.450187, 0.295469, 0.235497, 0.553665, 0.600022, 0.543702,
        -0.239996, 0.069296, 0.955199, -0.863160, 0.084838, -0.250309, -0.135878, 0.832456, 0.496013
    };
    vit_float p_data[C*C] = {
        0.190721, 0.463033, 0.892001, -0.384570, 0.873542, 0.932012, -0.691990, -0.658554, -0.650910,
        0.099912, 0.059305, 0.727300, 0.608268, 0.774360, 0.424879, 0.096697, -0.483499, -0.254434,
        0.048386, -0.151807, 0.052029, 0.523549, -0.451472, 0.754321, -0.901313, 0.092159, 0.911630,
        -0.473757, 0.865227, 0.620753, 0.780934, 0.354285, 0.950550, -0.623814, 0.136006, -0.552911,
        0.024759, 0.276824, 0.901874, 0.638582, -0.008210, -0.754831, -0.982054, 0.854790, -0.067657,
        -0.108737, -0.523107, 0.107152, -0.050898, -0.212699, -0.531593, 0.662633, 0.914513, 0.192820,
        -0.042026, -0.002532, -0.931864, 0.623818, -0.897300, -0.734560, 0.848118, -0.922245, 0.586201,
        -0.560232, -0.474477, 0.867411, 0.273795, -0.139959, -0.447935, 0.231609, 0.875990, -0.501251,
        -0.715188, 0.723610, -0.255640, 0.837134, 0.360450, 0.852773, -0.908113, 0.942807, 0.550265
    };
    
    vit_float qb_data[C] = {-0.067304, 0.196617, -0.791649, 0.552098, 0.686811, 0.359159, 0.395233, 0.665119, 0.273050};     
    vit_float kb_data[C] = {-0.067304, 0.196617, -0.791649, 0.552098, 0.686811, 0.359159, 0.395233, 0.665119, 0.273050};
    vit_float vb_data[C] = {-0.213135, 0.884225, -0.646646, -0.524352, 0.570676, -0.602515, -0.492012, -0.658386, -0.906315};
    vit_float pb_data[C] = {0.439476, -0.321163, -0.100588, -0.699733, 0.149049, -0.465826, -0.940250, 0.509871, -0.375616};
    //Mlp data
    vit_float A1_data[K*C] = {
            -0.456297, 0.451657, 0.790088, -0.792936, -0.640623, 0.283185,-0.456297, 0.451657, 0.790088, 
            -0.686055, 0.276620, 0.659866, -0.011135, 0.430428, -0.378445,-0.686055, 0.276620, 0.659866, 
            -0.620894, 0.601418, -0.575021, 0.246048, 0.333867, 0.860185, -0.620894, 0.601418, -0.575021,
            -0.463988, -0.217379, 0.652214, -0.578656, -0.905936, 0.707247,-0.463988, -0.217379, 0.652214,
            -0.708495, 0.267307, -0.129294, 0.521801, 0.373023, -0.193839,-0.708495, 0.267307, -0.129294,
            0.342945, 0.872465, -0.463590, 0.324824, 0.389333, 0.217408, 0.342945, 0.872465, -0.463590,
            -0.490482, -0.828896, 0.133649, -0.049476, -0.769367, 0.159828, -0.490482, -0.828896, 0.133649,     
            0.599725, -0.938836, 0.197874, -0.021849, -0.227208, -0.912308, 0.599725, -0.938836, 0.197874, 
            -0.285737, -0.105809, 0.647727, -0.029205, 0.209804, 0.876799, -0.285737, -0.105809, 0.647727,
            -0.006798, -0.411250, -0.551676, 0.055781, -0.335824, -0.228423, -0.006798, -0.411250, -0.551676
    };

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

    vit_float b1_data[K] = {-0.019805, -0.530365, -0.815562, -0.535694, -0.774685, -0.847759, -0.198200, -0.784896, -0.147666, 0.603477};
    vit_float b2_data[M] = {-55.690, 61.838, -25.379, 95.026, 2.756, 12.244, 85.241, 7.426, 1.000};
    //Layer norm
    vit_float n1g_data[C] = {0.439476, -0.321163, -0.100588, -0.699733, 0.149049, -0.465826, -0.940250, 0.509871, -0.375616};
    vit_float n1b_data[C] = {0.439476, -0.321163, -0.100588, -0.699733, 0.149049, -0.465826, -0.940250, 0.509871, -0.375616};
    vit_float n2g_data[C] = {0.439476, -0.321163, -0.100588, -0.699733, 0.149049, -0.465826, -0.940250, 0.509871, -0.375616};
    vit_float n2b_data[C] = {0.439476, -0.321163, -0.100588, -0.699733, 0.149049, -0.465826, -0.940250, 0.509871, -0.375616};
    

    layer_data ln1(n1g_data,n1b_data,epsilon, true);

    layer_data _t_; linear_data q(q_data,qb_data,C,C,true), k(k_data,kb_data,C,C,true), v(v_data,vb_data,C,C,true), p(p_data,pb_data,C,C,true);
    attn_data attn(
        q,k,v,
        _t_,_t_,
        p, 
        C, TEST_NUM_HEADS, C / TEST_NUM_HEADS, 0.0f,
        false
    );

    scale_data scale1(C,LAYER_SCALE);

    layer_data ln2(n2g_data,n2b_data,epsilon, true);

    linear_data fc1(A1_data,b1_data,C,K,true), fc2(A2_data,b2_data,K,M,true);
    mlp_data mlp(C,K,M,false,fc1,_t_,GELU,fc2);

    scale_data scale2(C,LAYER_SCALE);

    blocks_data block_data(
        C, TEST_NUM_HEADS, 1.4,
        ln1,
        attn,
        scale1,
        ln2,
        mlp,
        scale2
    );
    
    
    cpu_baseline(x_data, block_data);
    return;

    //GPU Part
    h_tensor x_gpu(x_data,B,C,1,T);

    
    half * y_gpu = (half *)malloc(sizeof(half) * B * T * M); // TO MALLOC!!

    //-Device allocation
    void * d_x,
    * d_b1_data, * d_b1_mtx,* d_b2_data, * d_b2_mtx,
    * d_fc1, * d_fc2, * d_h,
    * d_y;
    CUDA_CHECK(cudaMalloc(&d_x, sizeof(half) * B * T * C));CUDA_CHECK(cudaMemcpy(d_x, x_gpu.data, sizeof(half) * B * T *C, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_y, sizeof(half) * B * T * M)); // for now, then will have different shape
    
    //-Handle creation
    cublasLtHandle_t handle;CUBLAS_CHECK(cublasLtCreate(&handle));
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    //-MLP
    mtx b1_gpu(b1_data,1,K); mtx b1_gpu_mtx(K, B*T); bias_matrix(b1_gpu.data, b1_gpu_mtx.data, K, B*T);
    mtx b2_gpu(b2_data,1,M); mtx b2_gpu_mtx(M, B*T); bias_matrix(b2_gpu.data, b2_gpu_mtx.data, M, B*T);
    mtx fc1_gpu(A1_data,K,C);
    mtx fc2_gpu(A2_data,M,K);
    half * h_gpu = (half *)malloc(sizeof(half) * B * T * K);
    //First layer
    CUDA_CHECK(cudaMalloc(&d_fc1, sizeof(half) * K * C));CUDA_CHECK(cudaMemcpy(d_fc1, fc1_gpu.data, sizeof(half) * K *C, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_b1_data, sizeof(half) * K));CUDA_CHECK(cudaMemcpy(d_b1_data, b1_gpu.data, sizeof(half) * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_b1_mtx, sizeof(half) * B * T * K));CUDA_CHECK(cudaMemcpy(d_b1_mtx, b1_gpu_mtx.data, sizeof(half) * B * T * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_h, sizeof(half) * B * T * K)); // for now, then will have different shape
    //Second layer
    CUDA_CHECK(cudaMalloc(&d_fc2, sizeof(half) * M * K));CUDA_CHECK(cudaMemcpy(d_fc2, fc2_gpu.data, sizeof(half) * M * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_b2_data, sizeof(half) * M));CUDA_CHECK(cudaMemcpy(d_b2_data, b2_gpu.data, sizeof(half) * M, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_b2_mtx, sizeof(half) * B * T * M));CUDA_CHECK(cudaMemcpy(d_b2_mtx, b2_gpu_mtx.data, sizeof(half) * B * T * M, cudaMemcpyHostToDevice));
    //cuBLASLt
    cublasLt_matmul_desc matmul[2];
    cublasLtMatmulAlgo_t algo[2];
    void * d_workspace; cudaMalloc(&d_workspace, (size_t) MLP_WORKSPACE_SIZE);
    mlp_dimensions dim(B,T,C,K,M);
    create_mlp_descriptors(handle, matmul, d_workspace, algo, dim);

    fused_gpu_mlp(
        handle,stream1,
        matmul, algo, d_workspace,
        d_x, d_fc1, d_h,d_b1_mtx, d_fc2,d_b2_mtx,d_y
    );
    //Transpose
    cublasLtMatrixTransformDesc_t transposeDesc; CUBLAS_CHECK(cublasLtMatrixTransformDescCreate(&transposeDesc, MLP_COMPUTE_DATA_TYPE));
    cublasLtMatrixLayout_t mlp_out_desc; CUBLAS_CHECK(cublasLt);
    cublasLtMatrixLayout_t ln_in_desc;

    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&mlp_out_, CUDA_R_32F, N, M, N)); // A as transposed
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_32F, M, N, M));
    
    float alpha = 1.0f, beta = 0.0f;
    cublasLtMatrixTransform(
        handle, transposeDesc,
        &alpha, d_y, mlp_out_desc,
        &beta, nullptr, nullptr,
        d_x, ln_in_desc, stream1
    );

    //-Layer Norm
    half * d_n1_bias, * d_n1_scale;
    half gpu_epsilon = __double2half(epsilon);
    unrolled_multi_elem_cub_ln((half *)d_y, d_n1_scale, d_n1_bias, gpu_epsilon);
    //-Residual
    
    
    //-Attention
    // function that compute the mha and keeps the result on the device

    //-Layer Norm
    unrolled_multi_elem_cub_ln();
    //-Residual
    //...
    return;
}

int main() {

    cpu_gpu_comparison();
    return 0;
}