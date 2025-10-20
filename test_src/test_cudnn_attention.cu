#include <limits.h>
#include <random>
#include "../gpu_include/cudnn_attention.h"
#include "../include/attention.h"

#include <cstdio>
#include <cassert>

static random_device rd;
static mt19937 gen(rd());

float random_num(){
    uniform_real_distribution<float> distr(-0.1f, 0.1f);
    return distr(gen);
}

double compare_results(Tensor &y, half * gpu_y){
    u_int B = y.get_B(),T = y.get_N(),C = y.get_C();
    double avg = 0;
    for(u_int b = 0; b < B; b++){
        for(u_int t = 0; t < T; t++){
            for(u_int c = 0; c < C; c++){
                assert(!isnanf( y.at(b,t,c)));
                assert(!isnanf( __half2float(gpu_y[c + C * t + C * T * b])));
                avg += (double)abs(y.at(b,t,c) - __half2float(gpu_y[c + C * t + C * T * b]));
                
            }
        }
    }
    return avg / (double(B) * T * C);
}

void vector_f32_to_f16(float* in,vector<__half> &out, size_t dim){

    for (size_t i = 0; i < dim; ++i) {
        out.push_back(__float2half_rn(in[i]));   // host-available intrinsic
    }
}

template <class T>
void transpose(const T* src, T* dst, size_t rows, size_t cols){
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      dst[j*rows + i] = src[i*cols + j];
    }
  }
}

void cpu_gpu_comparison(){
    cout << "Test Attention" << endl;
    u_int B = 2, T = 7,C = 9, K = 9;
    u_int input_elements_number = B*T*C;
    u_int qkv_dimensions = C * K;
    u_int proj_dimensions = K * C;

    // Linear weights
    vit_float q_data[C*K] = {
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
    Matrix q(q_data, C*K, C, K);
    cout << "### q" << endl;
    q.print();

    vit_float k_data[C*K] = {
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

    Matrix k(k_data, C*K, C, K);
    cout << "### k" << endl;
    k.print();

    vit_float v_data[C*K] = {
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
    Matrix v(v_data, C*K, C, K);

    cout << "### v" << endl;
    v.print();

    // Biases
    vit_float qb_data[K] = {-0.067304, 0.196617, -0.791649, 0.552098, 0.686811, 0.359159, 0.395233, 0.665119, 0.273050};
    RowVector qb(qb_data, K);
    cout << "### qb" << endl;
    qb.print();
    vit_float kb_data[K] = {-0.067304, 0.196617, -0.791649, 0.552098, 0.686811, 0.359159, 0.395233, 0.665119, 0.273050};
    RowVector kb(kb_data, K);
    cout << "### kb" << endl;
    kb.print();
    vit_float vb_data[K] = {-0.213135, 0.884225, -0.646646, -0.524352, 0.570676, -0.602515, -0.492012, -0.658386, -0.906315};
    RowVector vb(vb_data, K);
    cout << "### vb" << endl;
    vb.print();

    //final linear projection, to set back the right dimension
    vit_float p_data[K*C] = {
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
    Matrix p(p_data, K*C, K, C);
    cout << "### p" << endl;
    p.print();

    // bias for proj
    vit_float pb_data[C] = {0.439476, -0.321163, -0.100588, -0.699733, 0.149049, -0.465826, -0.940250, 0.509871, -0.375616};
    RowVector pb(pb_data, C);
    cout << "### pb" << endl;
    pb.print();

    //input test data batch 2 sequence 7 embeddings 9
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

    Tensor x(x_data, input_elements_number, B, T, C);
    cout << "### x" << endl;
    x.print();


    //GPU PART
    cout << "creating the matrix in half precision" << endl;
    //ATTENTION!
    vit_float q_data_t[qkv_dimensions], k_data_t[qkv_dimensions], v_data_t[qkv_dimensions], p_data_t[proj_dimensions];
    transpose(q_data,q_data_t,C,K); /// TRANSPOSE EVERYTHING
    transpose(k_data,k_data_t,C,K); 
    transpose(v_data,v_data_t,C,K); 
    transpose(p_data,p_data_t,K,C); 
    
    mtx q_host(q_data_t, C,K);
    mtx k_host(k_data_t, C,K);
    mtx v_host(v_data_t, C,K);
    mtx p_host(p_data_t, K,C);
    cout << "creating the input tensor (array) in half precision" << endl;
    h_tensor x_host(x_data, B,T,1,C); //B, SEQ, BEAM, VECTOR
    cout << "creating the bias vectors in half precision" << endl;
    vector<__half> qb_host;
    vector<__half> kb_host;
    vector<__half> vb_host;
    vector<__half> pb_host;
    vector_f32_to_f16(qb_data,qb_host,K);
    vector_f32_to_f16(kb_data,kb_host,K);
    vector_f32_to_f16(vb_data,vb_host,K);
    vector_f32_to_f16(pb_data,pb_host,K);
    cout<< "initialized to half all the input data" << endl;
    half * host_out;
    host_out = (half *)malloc(sizeof(half) * x_host.B * x_host.C * x_host.H * x_host.W);
    
    cudnn_attention(
        q_host,k_host,v_host,p_host,x_host,
        qb_host,kb_host, vb_host, pb_host, //not used rn
        host_out
    );
    
    cout << "Tensor[" << x_host.B << "x" << x_host.C << "x" << x_host.W << "]:" << endl;
    for(int b=0;b<x_host.B;++b) {
        cout << "   B[" << b << "]" << endl;
        for (int n=0;n<x_host.C;++n) {
            cout << "   ";
            for (int c=0;c<x_host.W;++c) {
                printf("%7.3f ", __half2float( host_out[c + (n* x_host.W)+( b*x_host.W*x_host.C)]));
            }
            cout << endl;
        }
    }
    cout << endl;


    //CPU PART

    Attention attn(C, TEST_NUM_HEADS, false, false); //No layer norm

    Linear q_gen(C, K, true);
    // Linear q_gen(9, 9, false);
    q_gen.move_A(q);
    q_gen.move_b(qb);
    Linear k_gen(C, K, true);
    // Linear k_gen(9, 9, false);
    k_gen.move_A(k);
    k_gen.move_b(kb);
    Linear v_gen(C, K, true);
    // Linear v_gen(9, 9, false);
    v_gen.move_A(v);
    v_gen.move_b(vb);
    Linear proj(K, C, true);
    // Linear proj(9, 9, false);
    proj.move_A(p);
    proj.move_b(pb);

    attn.move_qkv_gen(q_gen, k_gen, v_gen);
    attn.move_proj(proj);

    Tensor y;
    attn.forward(x, y);
    cout << "### y = attn(x)" << endl;
    y.print();

    cout << "avg. difference CPU/cuDNN GPU: "  << compare_results(y, host_out) << endl;
    return;
}

/*
To see how cuDNN attention scale on a real problem
*/
void gpu_comparison(){

    u_int B = 16, T = 196,C = 768, K = 768;
    u_int input_elements_number = B*T*C;
    u_int qkv_dimensions = C * K;
    u_int proj_dimensions = K * C;
    float * q_data, * k_data, * v_data, * p_data;
    float * x_data;
    float * qb_data, * kb_data, * vb_data, * pb_data;
    x_data = (float *)malloc(sizeof(float) * input_elements_number);
    q_data = (float *)malloc(sizeof(float) * qkv_dimensions); 
    k_data= (float *)malloc(sizeof(float) * qkv_dimensions);
    v_data= (float *)malloc(sizeof(float) * qkv_dimensions);
    p_data= (float *)malloc(sizeof(float) * proj_dimensions);
    qb_data= (float *)malloc(sizeof(float) * C);
    kb_data= (float *)malloc(sizeof(float) * C); 
    vb_data= (float *)malloc(sizeof(float) * C); 
    pb_data= (float *)malloc(sizeof(float) * K);

    cout << "Tensor: [" << B << ","<< T << "," << C << "]" << endl;
    cout << "initialize" << endl;
    for (size_t i = 0; i < input_elements_number; i++){
        x_data[i] = random_num();
    }
    for (size_t i = 0; i < qkv_dimensions; i++){
        q_data[i] = random_num();
        k_data[i] = random_num();
        v_data[i] = random_num();
        p_data[i] = random_num();
    }
    for (size_t i = 0; i < C; i++){
        qb_data[i] = random_num();
        kb_data[i] = random_num();
        vb_data[i] = random_num();
    }
    for (size_t i = 0; i < K; i++){
        pb_data[i] = random_num();
    }
    
    
    
    // GPU
    cout << "GPU" << endl;
    vit_float * q_data_t, * k_data_t, * v_data_t, * p_data_t;
    q_data_t= (float *)malloc(sizeof(float) * qkv_dimensions); 
    k_data_t= (float *)malloc(sizeof(float) * qkv_dimensions); 
    v_data_t= (float *)malloc(sizeof(float) * qkv_dimensions); 
    p_data_t = (float *)malloc(sizeof(float) * proj_dimensions);
    transpose(q_data,q_data_t,C,K); /// TRANSPOSE EVERYTHING, this probably because
    transpose(k_data,k_data_t,C,K); 
    transpose(v_data,v_data_t,C,K); 
    transpose(p_data,p_data_t,K,C); 
    
    mtx q_host(q_data_t, C,K);
    mtx k_host(k_data_t, C,K);
    mtx v_host(v_data_t, C,K);
    mtx p_host(p_data_t, K,C);
    cout << "creating the input tensor (array) in half precision" << endl;
    h_tensor x_host(x_data, B,T,1,C); //B, SEQ, BEAM, VECTOR
    cout << "creating the bias vectors in half precision" << endl;
    vector<__half> qb_host;
    vector<__half> kb_host;
    vector<__half> vb_host;
    vector<__half> pb_host;
    vector_f32_to_f16(qb_data,qb_host,C);
    vector_f32_to_f16(kb_data,kb_host,C);
    vector_f32_to_f16(vb_data,vb_host,C);
    vector_f32_to_f16(pb_data,pb_host,K);
    cout<< "initialized to half all the input data" << endl;
    half * host_out;
    host_out = (half *)malloc(sizeof(half) * x_host.B * x_host.C * x_host.H * x_host.W);
    
    cudnn_attention(
        q_host,k_host,v_host,p_host,x_host,
        qb_host,kb_host, vb_host, pb_host, //not used rn
        host_out
    );


    //CPU PART
    cout << "CPU" << endl;
    Matrix q(q_data, C*K, C, K);
    Matrix k(k_data, C*K, C, K);
    Matrix v(v_data, C*K, C, K);
    Matrix p(p_data, K*C, K, C);

    RowVector qb(qb_data, K);
    RowVector kb(kb_data, K);
    RowVector vb(vb_data, K);
    RowVector pb(pb_data, C);

    Attention attn(C, NUM_HEADS, true, false); //No layer norm

    Linear q_gen(C, K, true);
    // Linear q_gen(9, 9, false);
    q_gen.move_A(q);
    q_gen.move_b(qb);
    Linear k_gen(C, K, true);
    // Linear k_gen(9, 9, false);
    k_gen.move_A(k);
    k_gen.move_b(kb);
    Linear v_gen(C, K, true);
    // Linear v_gen(9, 9, false);
    v_gen.move_A(v);
    v_gen.move_b(vb);
    Linear proj(K, C, true);
    // Linear proj(9, 9, false);
    proj.move_A(p);
    proj.move_b(pb);

    attn.move_qkv_gen(q_gen, k_gen, v_gen);
    attn.move_proj(proj);

    Tensor x(x_data,input_elements_number,B,T,C);
    Tensor y(B,T,C);
    attn.forward(x, y);
    cout << "computing the differences" << endl;
    cout << "avg. difference CPU/cuDNN GPU: "  << compare_results(y, host_out) << endl;
    return;

}

int main(){
    test_type test = GPU_COMPARISON;
    if(test == CPU_COMPARISON){
        cpu_gpu_comparison();
    }
    else{
        gpu_comparison();
    }

    return 0;
}


