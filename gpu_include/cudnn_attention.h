#include <cstdlib>

#include "../gpu_include/cudnn_utils.h"
// #include "../gpu_include/gpu_datatypes.h"

#define TEST_NUM_HEADS 3
#define NUM_HEADS 12

#define ATTN_MODE CUDNN_ATTN_ENABLE_PROJ_BIASES
#define ATTN_COMPUTE_TYPE CUDNN_DATA_FLOAT// CUDNN_DATA_HALF works or CUDNN_DATA_FLOAT // (To match MLP GEMM compute type)
#define ATTN_MATH_TYPE CUDNN_DEFAULT_MATH

using namespace std;

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