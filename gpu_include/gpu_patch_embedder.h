#include "./cudnn_conv2d.h"

//Could improve this version with one where I stride inside the block and have not to use the `%` operator
__global__ void add_pos_embeddings(half * d_x, half * d_pos_emb, u_int n, u_int single_sample_size);

class GpuPatchEmbedder {
    public:
        cudaStream_t stream;
        int batch, channels, height, width;
        int embeddings, Ho, Wo;

        u_int transpose_blocks_n, pos_emb_blocks_n, block_dim;

        //Unique pointers
        void * d_pic;
        void * d_out_pic;
        void * d_x;
        void * d_t;

        //Shared pointers
        void * d_w;
        void * d_bias;
        void * d_pos_emb;

        convolution_desc conv_desc;
               
        GpuPatchEmbedder(const GpuPatchEmbedder&) = delete;
        GpuPatchEmbedder& operator=(const GpuPatchEmbedder&) = delete;

        GpuPatchEmbedder& operator=(GpuPatchEmbedder&& pe) noexcept;

        GpuPatchEmbedder(){}
        
        //Initialize the descriptors and allocate the device pointers
        GpuPatchEmbedder(
            cudaStream_t &stream_,
            cudnnHandle_t &handle,
            convolution_dim &conv_dim_,
            bool init_shared_ptrs = true
        );

        ~GpuPatchEmbedder();

        void free_weights();

        //The flatten op results in d_t
        void add_cls_token();

        //Transform the input pictures stored in d_pic into tokens stored in d_x
        void forward(bool debug = false);

        //Copy the result in d_y
        void forward(half * out, bool on_device, bool debug = false);

        //Is intended for passing some already 
        void set_weights_data(void * d_w_, void * d_bias_, void * d_pos_emb_);

        void load_weights_data(half * conv_w, half * bias, half * pos_emb, bool on_device);

        void load_pics(half * h_pic);

    private:
        int tokens;

        bool own_device_ptrs = true; /*if false, i don't have to free the memory area associated*/
        u_int input_pic_elements_num;
        u_int output_pic_elements_num;
        u_int flatten_elements_num;
        u_int embedded_elements_num;
        u_int conv_kernel_elements_num;
        
        convolution_dim conv_dim;

};