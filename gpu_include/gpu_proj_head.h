#include "./gpu_layer.h"
#include "./gpu_mlp.h"

struct softmax_desc {
    cudnnTensorDescriptor_t x_desc;
    cudnnSoftmaxAlgorithm_t algo = CUDNN_SOFTMAX_FAST;
    cudnnSoftmaxMode_t mode = CUDNN_SOFTMAX_MODE_INSTANCE;

    void destroy_descriptors();
};

int argmax(vector<float>& vec, int begin, int end);

// `dimensions` will be [B,1,E,CLS_NUM,0]
void create_ph_desc(
    cublasLtHandle_t &cublas_handle,
    mlp_dimensions dimensions,
    cublasLt_matmul_desc &matmul, cublasLtMatmulAlgo_t &cublas_algo,
    softmax_desc &softmax,
    void * d_workspace
);

class GpuPredictionHead {
    public:
        cudnnHandle_t cudnn_handle;
        cublasLtHandle_t cublas_handle;
        cudaStream_t stream;

        softmax_desc softmax;
        cublasLtMatmulAlgo_t algo;
        cublasLt_matmul_desc matmul;

        // Shared pointers
        void * d_x;         
        void * d_t;         
        void * d_y;         
        void * d_pred;      
        void * d_workspace;

        // Unique pointers
        void * d_ln_scale;      
        void * d_ln_bias;       
        void * d_head_weights;  
        void * d_head_bias;     

        u_int batch;
        u_int tokens;
        u_int embeddings;
        u_int class_num;

        u_int stride_val = 2;
        u_int blocks_num;
        u_int block_dim;
        u_int tokens_per_block = 1;

        vector<float> probabilities_array;
        vector<int> class_prediction;

        // Debug
        half * gpu_x;
        float * h_x; 
        
        //Initialize the object istance and descriptors, allocate unique pointers 
        GpuPredictionHead(
            u_int batch_,
            u_int tokens_,
            u_int embeddings_,
            u_int class_num_,
            cudnnHandle_t &cudnn_handle_,
            cublasLtHandle_t &cublas_handle_,
            cudaStream_t &stream_,
            void * d_workspace_ = nullptr,
            bool allocate = true
        );

        ~GpuPredictionHead();

        void mark_shared_buffers();

        void mark_shared_weights();

        void destroy_descriptors();

        void allocate_ptrs();

        void load_weights(
            half * h_ln_scale_,   
            half * h_ln_bias_,    
            half * h_head_weights_,
            half * h_head_bias_  
        );

        void set_shared_weights(
            void * d_ln_scale_,   
            void * d_ln_bias_,    
            void * d_head_weights_,
            void * d_head_bias_  
        );

        void set_shared_buffers(
            void * d_x_,        
            void * d_t_,        
            void * d_y_,        
            void * d_pred_    
        );

        void compute_predictions();

        void forward(bool debug = false);

    private:
        u_int input_elements_number;
        float alpha = 1.0f;
        float beta = 0.0f;
        double epsilon = 1e-4;
        bool destroy_shared_weights = false;
        bool destroy_shared_buffers = false;

};