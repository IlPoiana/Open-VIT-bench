#include "./cudnn_attention.h"
#include "./gpu_mlp.h"
#include "./gpu_layer.h"
#include "../include/block.h"

#define RESIDUAL_BLOCK_DIM 256
#define RESIDUAL_ELEM_PER_THREAD 4

#define LAYER_SCALE 0.00004


//One thread for each element
__global__ void residual_test(half * d_x, half * d_y, u_int N);

__global__ void residual_strided(half * d_x, half * d_y, u_int , float scale = 1.0f);

__global__ void gpu_scale(half * d_x, half * d_y, u_int N, float scale);

// -------

class GpuBlock {
public:
    // Dimensions / config
    u_int batch, tokens, channels, k_channels;
    bool kernel_type;          //true means fused_mlp
    float scale = LAYER_SCALE; //scale applied before the residual
    float epsilon = 0.00001;   //epsilon value for the layer norm
    int num_heads = 1;         //multi-head attention heads number

    // CUDA stream
    cudaStream_t stream;

    // cuBLASLt / cuDNN handles
    cublasLtHandle_t ltHandle;
    cudnnHandle_t    cudnnHandle;

    // cuBLASLt matmul descriptors for the MLP path
    cublasLt_matmul_desc matmul[2];
    cublasLtMatmulAlgo_t algo[2];
    // void* d_workspace = nullptr;

    // cuDNN attention descriptors
    attn_cuDNN_descriptors fused_desc;

    // Device buffers (main activations and temps)
    void* d_x      = nullptr;  // [B*T*C] input / running residual
    void* d_t      = nullptr;  // temp/transpose
    void* d_y      = nullptr;  // output scratch [B*T*M] etc.
    void* d_h      = nullptr;  // mlp hidden [B*T*K]
    void* d_workspace_mlp = nullptr; 

    // LayerNorm params
    void* d_n1_bias  = nullptr;
    void* d_n1_scale = nullptr;
    void* d_n2_bias  = nullptr;
    void* d_n2_scale = nullptr;

    // Attention weights/biases 
    // - host side (useful for debugging)
    float* h_q  = nullptr;
    float* h_k  = nullptr;
    float* h_v  = nullptr;
    float* h_p  = nullptr;
    float* h_qb = nullptr;
    float* h_kb = nullptr;
    float* h_vb = nullptr;
    float* h_pb = nullptr;

    // MLP weights/biases
    void* d_fc1      = nullptr;
    void* d_b1_data  = nullptr;
    void* d_b1_mtx   = nullptr;
    void* d_fc2      = nullptr;
    void* d_b2_data  = nullptr;
    void* d_b2_mtx   = nullptr;

    // transpose descriptors etc (optional path)
    float mlp_alpha = 1.0f, mlp_beta = 0.0f;
    cublasLtMatrixTransformDesc_t transposeDesc = nullptr;
    cublasLtMatrixLayout_t        mlp_out_desc  = nullptr;
    cublasLtMatrixLayout_t        res_in_desc    = nullptr;

    // Host scratch for debug output
    half* h_debug_out = nullptr;

public:

    //Pass all the descriptors(except the attention one) and the shapes and initialize all the memory buffers
    GpuBlock(GpuBlock &precedent_block);

    GpuBlock(
        u_int B_, u_int T_, u_int C_, u_int K_,
        bool kernel_type_,
        double epsilon_, float scale_, int num_heads_ = NUM_HEADS,
        float rand_scale_ = 0.1f
    );

    /*
    `initialize_descriptors`: if true, it will create and initialize the stream, handles and mlp descriptors
    */
    GpuBlock(
        u_int B_, u_int T_, u_int C_, u_int K_,
        void * d_x_, void * d_h_, void * d_t_, void * d_y_,
        bool kernel_type_,
        double epsilon_, float scale_, int num_heads_ = NUM_HEADS,
        float rand_scale_ = 0.1f, bool initialize_descriptors = false
    );

    /*
    Initialize the block with the weights passed as input:
    `initialize_descriptors`: if true, it create from scratch the descriptors, 
    otherwise, descriptors have to be passed successively.

    Ex:
    ```
    ...descriptors 

    GpuBlock first_block(variables, initialize_descriptors = true);
    first_block.get_descriptors(...descriptors);

    GpuBlock second_block(variables, initialize_descriptors = false);
    second_block.set_descriptors(...descriptors);
    second_block.init_attn_descriptor();
    second_block.d_x = first_block.d_x;
    ```
    */
    GpuBlock(
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

        int num_heads_ = NUM_HEADS,
        float rand_scale_ = 0.1f,
        bool initialize_descriptors = false
    );

    // ---- dtor ----
    ~GpuBlock();


    void init_attn_descriptor();

    //Initialize the attention descriptor given the weight matrices
    void init_attn_descriptor(
        float * h_q_, float * h_k_, float * h_v_, float * h_p_,
        float * h_qb_, float * h_kb_, float * h_vb_, float * h_pb_
    ); 


    /*Generate random weights for the block
    - `attn_init`: true if the attn descriptor is already initialized, otherwise initialize it
    */
    void random_data(bool attn_init, bool input = true);

    //random input generation, the result is stored in dx
    // - `debug`: if true, puts the result in the h_debug_out
    void forward(bool debug = false, u_int tokens_per_block = 2);

    void forward(float * h_x, bool debug = false, u_int tokens_per_block = 2);

    void forward(half * h_x, bool debug = false, u_int tokens_per_block = 2);

    // ---- Setters ----

    /*
    Copy on the host device the block weights, converting to half
    */
    void set_data(
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
    );

    // Replace matmul descriptors (e.g. autotuned algos)
    void set_matmul_descriptors(
        const cublasLt_matmul_desc newMatmul[2],
        const cublasLtMatmulAlgo_t newAlgo[2],
        void* newWorkspace
    );

    void set_rand_scale(float _scale);

    //Call this method to destroy the shared device pointer and descriptors between block, should be called before the destructor call.
    void mark_shared_buffers();

    void set_last_block();

    //Initialize the block descriptors 
    void set_descriptors(
        cudaStream_t stream_,
        cublasLtHandle_t ltHandle_,
        cudnnHandle_t cudnnHandle_,
        attn_cuDNN_descriptors fused_desc_, /*MHA cuDNN descriptors*/
        cublasLt_matmul_desc matmul_[2], cublasLtMatmulAlgo_t algo_[2], /*MLP descriptors*/
        cublasLtMatrixTransformDesc_t transposeDesc_, /*MLP Transpose descriptors*/
        void * d_mlp_workspace_, /*MLP workspace device pointer*/
        cublasLtMatrixLayout_t mlp_out_desc_, cublasLtMatrixLayout_t res_in_desc_     
    );

    //Copy of the descriptors
    void get_descriptors(
        cudaStream_t &_stream,
        cublasLtHandle_t &_ltHandle,
        cudnnHandle_t &_cudnnHandle,
        attn_cuDNN_descriptors &_fused_desc, /*MHA cuDNN descriptors*/
        cublasLt_matmul_desc (&_matmul)[2], cublasLtMatmulAlgo_t (&_algo)[2], /*MLP descriptors*/
        cublasLtMatrixTransformDesc_t &_transposeDesc, /*MLP Transpose descriptors*/
        cublasLtMatrixLayout_t &_mlp_out_desc, cublasLtMatrixLayout_t &_res_in_desc 
    );

    float get_rand_scale();

    void to_CPU(Block &cpu_block, bool debug = false);

    void print_h_out();

    // helper: pull d_x (B*T*C) back to host_half buffer for debug
    void download_x(float * h_x);

private:
    float rand_scale = 0.1f;
    u_int input_elements_number;
    u_int hidden_elements_number;
    bool destroy_shared_buffers = false;
    bool destroy_shared_weights = false;

    //print d_x, d_t, d_h, d_y
    void print_debug();

    void destroyCudnnDescriptors();

    //Transpose a square channels X channels mtx from float to half 
    void transposeHostF32toHalf(float* src, vector<half>& dst);

    //Initialize randomly the passed DEVICE vector with `dim` size (half type), scaled by `rand_scale`
    void populate_rand(void * d_var, u_int dim);

    //Initialize randomly the passed HOST vector with `dim` size (float type), , scaled by `rand_scale`
    void populate_rand(float * h_var, u_int dim);
};