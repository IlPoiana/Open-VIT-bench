#include "../gpu_include/gpu_patch_embedder.h"
patch_emb_weights::patch_emb_weights():
    conv_w(),
    bias(),
    pos_emb()
{}

//Just a pointers struct
patch_emb_weights::patch_emb_weights(half * _conv_w, half * _bias, half * _pos_emb){
    conv_w = _conv_w;
    bias = _bias;
    pos_emb = _pos_emb;
}

//Could improve this version with one where I stride inside the block and have not to use the `%` operator
__global__ void add_pos_embeddings(half * d_x, half * d_pos_emb, u_int n, u_int single_sample_size){
    u_int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int tok_idx;
    for (int i = idx; i < n; i += stride) {
        tok_idx = i % single_sample_size;
        d_x[i] += d_pos_emb[tok_idx];
    }
}

GpuPatchEmbedder &GpuPatchEmbedder::operator=(GpuPatchEmbedder &&pe) noexcept{
    stream = pe.stream;
    conv_dim = pe.conv_dim;
    batch = pe.conv_dim.batch; 
    channels = pe.conv_dim.channels; 
    height = pe.conv_dim.height;
    width = pe.conv_dim.width;
    embeddings = pe.conv_dim.embeddings;
    Ho = pe.conv_dim.Ho; Wo = pe.conv_dim.Wo;
    tokens = pe.conv_dim.y_height * pe.conv_dim.y_width;

    transpose_blocks_n = pe.transpose_blocks_n;
    pos_emb_blocks_n = pe.pos_emb_blocks_n;
    block_dim = pe.block_dim;   
    
    d_pic = pe.d_pic;
    d_out_pic = pe.d_out_pic;
    d_x=pe.d_x;
    d_t=pe.d_t;
    d_w=pe.d_w;
    d_bias = pe.d_bias;
    d_pos_emb = pe.d_pos_emb;

    pe.own_device_ptrs = false;

    conv_desc = pe.conv_desc;
    input_pic_elements_num = pe.input_pic_elements_num;

    output_pic_elements_num = pe.output_pic_elements_num;
    flatten_elements_num = pe.flatten_elements_num;   
    embedded_elements_num = pe.embedded_elements_num;  
    conv_kernel_elements_num = pe.conv_kernel_elements_num;

    return *this;
}
        
//Initialize the descriptors and allocate the device pointers
GpuPatchEmbedder::GpuPatchEmbedder(
    cudaStream_t &stream_,
    cudnnHandle_t &handle,
    convolution_dim &conv_dim_,
    bool init_shared_ptrs,
    bool init_descriptors,
    bool allocate_shared_buffers
):
    stream(stream_),
    conv_dim(conv_dim_),
    batch(conv_dim_.batch), 
    channels(conv_dim_.channels), 
    height(conv_dim_.height),
    width(conv_dim_.width),
    embeddings(conv_dim_.embeddings),
    Ho(conv_dim_.Ho), Wo(conv_dim_.Wo),
    tokens(conv_dim_.y_height * conv_dim_.y_width)
{
    // /*TO REMOVE*/ cout << conv_dim_.batch << endl << batch<< "," << channels<< "," << height<< "," << width << endl;
    // cout << embeddings<< "," << Ho<< "," << Wo << endl;
    // cout << "tokens: " << tokens << endl;
    // 0. initialize all the utility variables
    input_pic_elements_num = batch * channels * height * width;
    output_pic_elements_num = batch * embeddings * conv_dim.y_height * conv_dim.y_width;
    flatten_elements_num = batch * tokens * embeddings;
    embedded_elements_num = batch * (tokens + 1) * embeddings;
    conv_kernel_elements_num = embeddings * channels * Ho * Wo;
    
    // 1. Allocate all the device pointers 
    if(allocate_shared_buffers){
        CUDA_CHECK(cudaMallocAsync(&d_pic, sizeof(half) * input_pic_elements_num, stream));
        CUDA_CHECK(cudaMallocAsync(&d_out_pic, sizeof(half) * output_pic_elements_num, stream));
        CUDA_CHECK(cudaMallocAsync(&d_t, sizeof(half) * flatten_elements_num, stream));
        CUDA_CHECK(cudaMallocAsync(&d_x, sizeof(half) * embedded_elements_num, stream));
        own_device_ptrs = true;
    }
    if(init_shared_ptrs){
        CUDA_CHECK(cudaMallocAsync(&d_w, sizeof(half) * conv_kernel_elements_num, stream));
        CUDA_CHECK(cudaMallocAsync(&d_bias, sizeof(half) * embeddings, stream));
        CUDA_CHECK(cudaMallocAsync(&d_pos_emb, sizeof(half) * (tokens + 1) *embeddings, stream));
    }

    // 2. Initialize the descriptors (cuDNN)
    conv_desc.handle = handle;
    if(init_descriptors)
        init_conv2d_descriptors(conv_desc, conv_dim, true);

    // 3. Initialize the kernel launch variables
    block_dim = 256;
    transpose_blocks_n = ((flatten_elements_num) / (block_dim * 4)) + 1; //To tune this value
    pos_emb_blocks_n = ((embedded_elements_num) / (block_dim * 4)) + 1;  //To tune this value

}

GpuPatchEmbedder::~GpuPatchEmbedder(){
    if(own_device_ptrs){
        CUDA_CHECK(cudaFree(d_pic));
        CUDA_CHECK(cudaFree(d_out_pic));
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_t));
        //Free the cudnn descriptors only if I'm owning them
        CUDA_CHECK(cudaFree(conv_desc.d_workspace));        
    }
}

void GpuPatchEmbedder::allocate_weights(){
    CUDA_CHECK(cudaMallocAsync(&d_w, sizeof(half) * conv_kernel_elements_num, stream));
    CUDA_CHECK(cudaMallocAsync(&d_bias, sizeof(half) * embeddings, stream));
    CUDA_CHECK(cudaMallocAsync(&d_pos_emb, sizeof(half) * (tokens + 1) *embeddings, stream));
}

void GpuPatchEmbedder::free_weights(){
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_pos_emb));
}

//The flatten op results in d_t
void GpuPatchEmbedder::add_cls_token(){
    half * d_tok = (half*)d_x;
    half * d_flat = (half*)d_t;
    for(int b = 0; b < batch; b++){
        d_tok += embeddings; //Skip the first token(CLS token)
        CUDA_CHECK(
            cudaMemcpyAsync(d_tok,d_flat,sizeof(half) * tokens * embeddings,cudaMemcpyDeviceToDevice,stream)
        );
        d_tok += (tokens * embeddings); //next batch
        d_flat += (tokens * embeddings);
    }
}


void GpuPatchEmbedder::init_descriptors(){
    init_conv2d_descriptors(conv_desc, conv_dim, true);
}

void GpuPatchEmbedder::destroy_descriptors(){
    conv_desc.destroy_descriptors();
}


//Transform the input pictures stored in d_pic into tokens stored in d_x
void GpuPatchEmbedder::forward(bool debug){
    //conv2d 
    execute_cudnn_conv2d_bias(d_pic, d_w, d_out_pic, d_bias, conv_desc);
    
    if(debug){
        vector<half>host_t(output_pic_elements_num);
        vector<float> t_float(output_pic_elements_num); 
        CUDA_CHECK(cudaMemcpyAsync(host_t.data(),d_out_pic,sizeof(half) * output_pic_elements_num,cudaMemcpyDeviceToHost,stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        f16_to_f32(host_t.data(), t_float.data(), output_pic_elements_num);
        PictureBatch h_out_pic(t_float.data(), output_pic_elements_num, batch, embeddings, conv_dim.y_height, conv_dim.y_width);
        cout << "h_out_pic: " << endl;h_out_pic.print();
    }
    //transpose 
    transpose_strided_tensor3d<<<transpose_blocks_n,block_dim,0,stream>>>((half *)d_out_pic,(half *)d_t,batch,embeddings, tokens);
    
    if(debug){
        vector<half>host_t(flatten_elements_num);
        vector<float> t_float(flatten_elements_num); 
        CUDA_CHECK(cudaMemcpyAsync(host_t.data(),d_t,sizeof(half) * flatten_elements_num,cudaMemcpyDeviceToHost,stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        f16_to_f32(host_t.data(), t_float.data(), flatten_elements_num);
        Tensor h_out(t_float.data(), flatten_elements_num, batch, tokens,embeddings);
        cout << "flatten tensor: " << endl;h_out.print();
    }
    
    //add cls token             
    // For each sample of the batch, launch an async memcpy
    add_cls_token();
    // cudaStreamSynchronize(stream);  /*Should not be necessary, because on the same stream */

    if(debug){
        vector<half>host_t(embedded_elements_num);
        vector<float> t_float(embedded_elements_num); 
        CUDA_CHECK(cudaMemcpyAsync(host_t.data(),d_x,sizeof(half) * embedded_elements_num,cudaMemcpyDeviceToHost,stream));
        cudaStreamSynchronize(stream);
        f16_to_f32(host_t.data(), t_float.data(), embedded_elements_num);
        Tensor h_out(t_float.data(), embedded_elements_num, batch, tokens + 1,embeddings);
        cout << "flatten tensor + cls token: " << endl;h_out.print();
    }
    
    //add pos embeddings
    add_pos_embeddings<<<pos_emb_blocks_n, block_dim, 0, stream>>>((half *)d_x, (half *)d_pos_emb, embedded_elements_num, (embeddings * (tokens + 1)) );
    
    if(debug){
        vector<half>host_t(embedded_elements_num);
        vector<float> t_float(embedded_elements_num); 
        CUDA_CHECK(cudaMemcpyAsync(host_t.data(),d_x,sizeof(half) * embedded_elements_num,cudaMemcpyDeviceToHost,stream));
        cudaStreamSynchronize(stream);
        f16_to_f32(host_t.data(), t_float.data(), embedded_elements_num);
        Tensor h_out(t_float.data(), embedded_elements_num, batch, tokens + 1,embeddings);
        cout << "position embeddings: " << endl;h_out.print();
    }
}

//Copy the result in d_y
void GpuPatchEmbedder::forward(half * out, bool on_device, bool debug){
    forward(debug);
    cudaMemcpyKind memcpy_kind = cudaMemcpyDeviceToHost;
    if(on_device) memcpy_kind = cudaMemcpyDeviceToDevice;
    CUDA_CHECK(cudaMemcpyAsync(out ,d_x, sizeof(half) * embedded_elements_num, memcpy_kind, stream));
}

void GpuPatchEmbedder::set_buffers(void * _d_pic, void * _d_out_pic, void * _d_x,void * _d_t, void * _d_workspace){
    if(own_device_ptrs)
        throw runtime_error("patch emb: trying to set external buffers on already allocated ones");
    
    d_pic     = _d_pic;
    d_out_pic = _d_out_pic;
    d_x       = _d_x;
    d_t       = _d_t;
    conv_desc.d_workspace = _d_workspace;
    conv_desc.workspace_size = WORKSPACE_SIZE; //4MB
}

//Is intended for passing some already 
void GpuPatchEmbedder::set_weights_data(void * d_w_, void * d_bias_, void * d_pos_emb_){
    d_w = d_w_;
    d_bias = d_bias_;
    d_pos_emb = d_pos_emb_;
}

void GpuPatchEmbedder::load_weights_data(half * conv_w, half * bias, half * pos_emb, bool on_device){
    cudaMemcpyKind memcpy_kind = cudaMemcpyHostToDevice;
    if(on_device) memcpy_kind = cudaMemcpyDeviceToDevice;
    CUDA_CHECK(cudaMemcpyAsync(d_w, conv_w, sizeof(half) * conv_kernel_elements_num ,memcpy_kind, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_bias, bias, sizeof(half) * embeddings , memcpy_kind, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_pos_emb , pos_emb , sizeof(half) * embeddings * (tokens + 1), memcpy_kind, stream));
}

void GpuPatchEmbedder::load_pics(half * h_pic){
    CUDA_CHECK(
        cudaMemcpyAsync(d_pic, h_pic, sizeof(half) * input_pic_elements_num, cudaMemcpyHostToDevice, stream)
    );
}

u_int GpuPatchEmbedder::get_input_pic_elem_n(){
    return input_pic_elements_num;
}

u_int GpuPatchEmbedder::get_flatten_elem_n(){
    return flatten_elements_num;
}
u_int GpuPatchEmbedder::get_embedded_elem_n(){
    return embedded_elements_num;
}