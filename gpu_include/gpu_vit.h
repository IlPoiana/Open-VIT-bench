#include "../include/vision_transformer.h"
#include "../include/utils.h"

#include "../gpu_include/gpu_patch_embedder.h"
#include "../gpu_include/gpu_block.h"
#include "../gpu_include/gpu_pred_head.h"

void transpose_out_of_place(const float * in, half* out, size_t rows, size_t cols);

patch_emb_weights convert_patch_emb(VisionTransformer &cpu_vit);

void convert_blocks( VisionTransformer &cpu_vit, vector<block_weights> &blk_w);

pred_head_weights convert_pred_head(VisionTransformer &cpu_vit);

// `blk_w`: is a pointer(non-initialized), it will point an array of depth size (number of encoder blocks)
void convert_vit_weights(
    VisionTransformer &vit,
    patch_emb_weights &pe_w,
    vector<block_weights> &blk_w,
    pred_head_weights &ph_w
);

/**
 * @brief Inference ViT implementation, supporting single or multi stream execution
 * 
 */
class GpuVit {
    private:
        convolution_dim conv_dim;
        int tokens;
        u_int input_pic_elements_num;
        u_int embedded_elements_num; // [B,T,E]
        u_int hidden_elements_number;

        double block_epsilon = 1e-6;
        double pred_head_epsilon = 1e-6;

        float block_scale = 1.0f;

        bool own_shared_buffers = false;
        bool own_weights = false;
        bool descriptors_are_initialized = false;

        void set_class_buffers();
        void reset_buffers();

    public:
        cudaStream_t     stream;
        cudnnHandle_t    cudnn_handle;
        cublasLtHandle_t cublas_handle;

        vit_size batch = 1;
        vit_size img_h = 224;
        vit_size img_w = 224;
        vit_size patch_h  = 16;
        vit_size patch_w  = 16;
        vit_size channels = 3;
        vit_size embeddings = 768;
        
        vit_size  depth;        
        vit_size  num_heads;
        vit_float scale_val;
        vit_size num_classes;
        int tokens_per_block = 4;

        void * d_pic    ; //[B,C,H,W]
        void * d_x      ; //[B,T,E]
        void * d_t      ; //[B,T,E]
        void * d_y      ; //[B,T,E]
        void * d_h      ; //[B,T,K]
        void * d_workspace = nullptr;

        GpuPatchEmbedder  pe;
        GpuPredictionHead ph;
        vector<GpuBlock>  blocks;          

        GpuVit(const GpuVit&) = delete;
        GpuVit& operator=(const GpuVit&) = delete;

        GpuVit(GpuVit&& other) noexcept;
        GpuVit& operator=(GpuVit&& vit) noexcept;

        GpuVit(
            cudaStream_t     &_stream,
            cudnnHandle_t    &_cudnn_handle,
            cublasLtHandle_t &_cublas_handle,

            convolution_dim _conv_dim,

            vit_size _tokens = 196,
            vit_size _num_classes = 1000,
            vit_size _depth = 12,
            vit_size  _num_heads = 12,
            vit_float _scale_val = 1.0,
            double    _block_epsilon = 1e-6,
            double    _ph_epsilon = 1e-6,
            vit_size mlp_hidden = 3072,

            vit_bool init_pe_descriptors = true,
            vit_bool allocate_pe_shared_ptrs = true, //initialize the weights shared pointers

            vit_bool block_mlp_kernel_type = true,
            vit_bool allocate_blocks_shared_ptrs = true //initialize the weights shared pointers
        );

        ~GpuVit();

        // 0) Allocate on device the buffers used in all the ops
        void allocate_shared_buffers();

        // 1) Create all the descriptors for all the library functions used(cuBLAS and cuDNN)
        void create_descriptors();

        // 2) Allocate on device the buffers for the weights, also the workspace used for this block
        void allocate_weights();

        // 3) Load all the weights for each component to the device
        void load_weights(
            patch_emb_weights &pe_w,
            vector<block_weights> &blk_w,
            pred_head_weights &ph_w
        );

        // 4) Load the input data to the model
        void load_pics(half * pics);
        
        /* 5) Forward of the model, starts from d_pic result in d_x!
        */
        void forward();
        
        void compute_predictions();

        void print_dimensions();

        void print_predictions(bool debug = false);
        
        void free_weights();
        
        void free_buffers();
        
        void destroy_descriptors();

        
};

class GpuVitBasePatch16_224 : public GpuVit{
    public:
        const int _img_w = 224, _img_h = 224; 
        const int _Ho    = 16 , _Wo    = 16 ;
        const int _channels   = 3;
        const int _embeddings = 768;
        const int _tokens     = 196;
        const int _depth      = 12;
        const int _num_heads  = 12;
        const int _mlp_hidden = 3072;
        const double _epsilon = 1e-5;
        const bool _mlp_kernel_type = true; //TO CHECK what is better

        GpuVitBasePatch16_224(
            cudaStream_t     &_stream,
            cudnnHandle_t    &_cudnn_handle,
            cublasLtHandle_t &_cublas_handle,
            int _batch,
            int _num_classes,
            vit_float _scale_val = 1.0
        ):
        GpuVit(
            _stream,
            _cudnn_handle,
            _cublas_handle,
            convolution_dim(
                _batch,
                _channels,
                _img_h,
                _img_w,
                _embeddings,
                _Ho,
                _Wo
            ),
            _tokens,
            _num_classes,
            _depth,     
            _num_heads, 
            _scale_val,
            _mlp_hidden,
            false,
            false,
            _mlp_kernel_type,
            false
        ){}

        void forward(){} // TO DO
};