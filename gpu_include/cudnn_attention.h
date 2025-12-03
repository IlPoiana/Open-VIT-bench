#include <cstdlib>

#include "../gpu_include/cudnn_utils.h"
// #include "../gpu_include/gpu_datatypes.h"

#define TEST_NUM_HEADS 3
#define NUM_HEADS 12

#define ATTN_MODE CUDNN_ATTN_ENABLE_PROJ_BIASES
#define ATTN_COMPUTE_TYPE CUDNN_DATA_FLOAT// CUDNN_DATA_HALF works or CUDNN_DATA_FLOAT // (To match MLP GEMM compute type)
#define ATTN_MATH_TYPE CUDNN_DEFAULT_MATH

using namespace std;

template<typename T>
struct attn_data_gpu{
    T * d_q;
    T * d_k;
    T * d_v;
    T * d_o;
    T * d_qb;
    T * d_kb;
    T * d_vb;
    T * d_ob;

    attn_data_gpu(
        T * in_d_q,
        T * in_d_k,
        T * in_d_v,
        T * in_d_o,
        T * in_d_qb,
        T * in_d_kb,
        T * in_d_vb,
        T * in_d_ob
    );

    attn_data_gpu();
};

//Implementing here to avoid errors
template<typename T>
attn_data_gpu<T>::attn_data_gpu(
    T * in_d_q,
    T * in_d_k,
    T * in_d_v,
    T * in_d_o,
    T * in_d_qb,
    T * in_d_kb,
    T * in_d_vb,
    T * in_d_ob
):
d_q (in_d_q), d_k (in_d_k), d_v (in_d_v), d_o (in_d_o),
d_qb( in_d_qb), d_kb( in_d_kb), d_vb( in_d_vb), d_ob( in_d_ob) 
{}

template<typename T>
attn_data_gpu<T>::attn_data_gpu():
d_q (nullptr), d_k (nullptr), d_v (nullptr), d_o (nullptr),
d_qb(nullptr), d_kb( nullptr), d_vb( nullptr), d_ob( nullptr) 
{}


struct attn_dimensions_gpu{
    u_int B;
    u_int T;
    u_int C;
    u_int proj_dim;

    attn_dimensions_gpu(u_int _B,u_int _T,u_int _C,u_int _proj);
};

struct attn_cuDNN_descriptors{
    cudnnDropoutDescriptor_t attnDrop = nullptr, postDrop = nullptr;
    cudnnAttnDescriptor_t attn;
    cudnnDataType_t dataType = CUDNN_DATA_HALF;
    cudnnSeqDataDescriptor_t qDesc, kDesc, vDesc, oDesc;
    int *dLenQO=nullptr, *dLenKV=nullptr;
    std::vector<int> loWin, hiWin;
    size_t weightBytes=0, workBytes=0;
    void* dWeights = nullptr, * dWork = nullptr;

    void destroy_descriptors();
};

void initialize_attn_descriptors(
    cudnnHandle_t &handle,
    attn_data_gpu<half> weights,
    attn_dimensions_gpu dim,
    attn_cuDNN_descriptors &descriptors,
    int num_heads = TEST_NUM_HEADS
);

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
);

void attention_device(
    void * d_input, void * d_output,
    attn_data_gpu<void> weights,
    attn_dimensions_gpu dim
);

void attention_device(
    cudnnHandle_t &handle,
    void * d_input, void * d_output,
    attn_cuDNN_descriptors &descriptors,
    bool residual = false, void * d_residuals = nullptr
);