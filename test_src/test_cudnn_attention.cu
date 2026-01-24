#include <limits.h>
#include <random>
#include "../gpu_include/cudnn_attention.h"
#include "../include/attention.h"

#include <cstdio>
#include <cassert>

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

/*
To see how cuDNN attention scale on a real problem
*/
void cpu_gpu_comparison(){

    u_int B = 4, T = 196,C = 768, K = 768;
    cout << "Tensor: [" << B << ","<< T << "," << C << "]" << endl;    
    cout << "Output Tensor: [" << B << ","<< T << "," << K << "]" << endl;

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

    
    // -- GPU --
    cout << "GPU" << endl;
    vit_float * q_data_t, * k_data_t, * v_data_t, * p_data_t;
    q_data_t= (float *)malloc(sizeof(float) * qkv_dimensions); 
    k_data_t= (float *)malloc(sizeof(float) * qkv_dimensions); 
    v_data_t= (float *)malloc(sizeof(float) * qkv_dimensions); 
    p_data_t = (float *)malloc(sizeof(float) * proj_dimensions);

    random_device rd;          
    mt19937 gen(rd());         
    uniform_real_distribution<float> dist(-0.1f, 0.1f);

    size_t loop_range = input_elements_number;
    for(size_t i = 0; i < loop_range; i++){
        if(i < C){
            qb_data[i] = dist(gen);
            kb_data[i] = dist(gen);
            vb_data[i] = dist(gen);
        }
        if(i < C * K){
            q_data[i] = dist(gen);
            k_data[i] = dist(gen);
            v_data[i] = dist(gen);
            p_data[i] = dist(gen);
        }
        if(i < K){
            pb_data[i] = dist(gen);
        }
        x_data[i] = dist(gen);
        
    }

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
    
    cout<< "initialized to half all the input data" << endl;
    vector_f32_to_f16(qb_data,qb_host,C);
    vector_f32_to_f16(kb_data,kb_host,C);
    vector_f32_to_f16(vb_data,vb_host,C);
    vector_f32_to_f16(pb_data,pb_host,K);
    
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
    q_gen.move_A(q);
    q_gen.move_b(qb);
    Linear k_gen(C, K, true);
    k_gen.move_A(k);
    k_gen.move_b(kb);
    Linear v_gen(C, K, true);
    v_gen.move_A(v);
    v_gen.move_b(vb);
    Linear proj(K, C, true);
    proj.move_A(p);
    proj.move_b(pb);

    attn.move_qkv_gen(q_gen, k_gen, v_gen);
    attn.move_proj(proj);

    Tensor x(x_data, input_elements_number, B, T, C);
    Tensor y;
    attn.forward(x, y);
    cout << "computing the differences" << endl;
    cout << "avg. difference CPU/cuDNN GPU: "  << compare_results(y, host_out) * 100<< "%" << endl;

    // -Cleanup
    free(q_data); free(k_data); free(v_data); free(p_data);
    free(qb_data); free(kb_data); free(vb_data); free(pb_data);
    free(x_data); free(q_data_t); free(k_data_t); free(v_data_t); free(p_data_t);
    free(host_out);

}

int main(){
    cpu_gpu_comparison();

    return 0;
}


