// #include "../gpu_include/cuda_utils.h"
#include "../gpu_include/gpu_vit.h"

void transpose_out_of_place(const float * in, half* out, std::size_t rows, std::size_t cols) {
    for (std::size_t i = 0; i < rows; ++i) {
        const float* Ai = in + i * cols;
        for (std::size_t j = 0; j < cols; ++j) {
            out[j * rows + i] = Ai[j];
        }
    }
}

patch_emb_weights convert_patch_emb(VisionTransformer &cpu_vit){
    
    vit_size img_height, img_width;
    cpu_vit.get_img_size(img_height, img_width);
    int conv_kernel_shape[6];
    cpu_vit.get_kernel_shape(conv_kernel_shape);
    int channels = conv_kernel_shape[0], 
    embeddings = conv_kernel_shape[1],
    Ho = conv_kernel_shape[2],
    Wo = conv_kernel_shape[3];
    assert((img_height % Ho == 0) && (img_width % Wo == 0));
    int tokens = (img_height / Ho) * (img_width / Wo);

    half * half_conv_w =(half*)malloc(sizeof(half) * channels * embeddings * Ho * Wo);
    half * half_bias =(half*)malloc(sizeof(half) * embeddings);
    half * half_pos_emb =(half*)malloc(sizeof(half) * embeddings * (tokens + 1));

    f32_to_f16(
        cpu_vit.get_conv2d_kernel(),
        half_conv_w,
        channels * embeddings * Ho * Wo
    );
    f32_to_f16(
        cpu_vit.get_conv2d_bias(),
        half_bias,
        embeddings
    );
    f32_to_f16(
        cpu_vit.get_pos_embed(),
        half_pos_emb,
        embeddings * (tokens + 1)
    );
    
    return patch_emb_weights(half_conv_w, half_bias, half_pos_emb);

}

void convert_blocks( VisionTransformer &cpu_vit, vector<block_weights> &out){
    vector<blocks_data> blocks = cpu_vit.get_blocks();
    vector<blocks_shape> b_shape = cpu_vit.get_blocks_shape();
    vit_size depth = cpu_vit.get_depth();
    layer_shape norm_dim = b_shape[0].norm1_shape;
    linear_shape fc1_dim = b_shape[0].mlperc_shape.fc1_shape;
    attn_shape attn_dim = b_shape[0].attention_shape;
    int embeddings = norm_dim.bias_size; 
    int k_embeddings = fc1_dim.a_row; //should be in col major the weight mtx, so the rows of it are the output dimensions
    
    vector<block_weights> blk_w; blk_w.reserve(depth);
    
    for(int i = 0; i< depth; i++){
        // - init half weights vectors
        half * n1_bias  = (half *)malloc(sizeof(half) * embeddings);
        half * n1_scale = (half *)malloc(sizeof(half) * embeddings);
        half * n2_bias  = (half *)malloc(sizeof(half) * embeddings);
        half * n2_scale = (half *)malloc(sizeof(half) * embeddings);
    
        half * q =(half *)malloc(sizeof(half) * embeddings * embeddings); 
        half * k =(half *)malloc(sizeof(half) * embeddings * embeddings);
        half * v =(half *)malloc(sizeof(half) * embeddings * embeddings);
        half * p =(half *)malloc(sizeof(half) * embeddings * embeddings);
        half * qb =(half *)malloc(sizeof(half) * embeddings);
        half * kb =(half *)malloc(sizeof(half) * embeddings);
        half * vb =(half *)malloc(sizeof(half) * embeddings);
        half * pb =(half *)malloc(sizeof(half) * embeddings);

        half * fc1     = (half *)malloc(sizeof(half) * embeddings * k_embeddings);
        half * b1_data = (half *)malloc(sizeof(half) * k_embeddings);
        half * fc2     = (half *)malloc(sizeof(half) * embeddings * k_embeddings);
        half * b2_data = (half *)malloc(sizeof(half) * embeddings);

        // - copy from cpu
        f32_to_f16(blocks[i].norm1.bias, n1_bias , embeddings);
        f32_to_f16(blocks[i].norm1.g, n1_scale, embeddings);
        f32_to_f16(blocks[i].norm2.bias, n2_bias , embeddings);
        f32_to_f16(blocks[i].norm2.g, n2_scale, embeddings);

        f32_to_f16(blocks[i].attention.q_gen.A, q , embeddings * embeddings);
        f32_to_f16(blocks[i].attention.k_gen.A, k , embeddings * embeddings);
        f32_to_f16(blocks[i].attention.v_gen.A, v , embeddings * embeddings);
        f32_to_f16(blocks[i].attention.proj.A, p , embeddings * embeddings);
        f32_to_f16(blocks[i].attention.q_gen.b, qb, embeddings);
        f32_to_f16(blocks[i].attention.k_gen.b, kb, embeddings);
        f32_to_f16(blocks[i].attention.v_gen.b, vb, embeddings);
        f32_to_f16(blocks[i].attention.proj.b, pb, embeddings);
        transpose_out_of_place(blocks[i].attention.q_gen.A, q, embeddings, embeddings);
        transpose_out_of_place(blocks[i].attention.k_gen.A, k, embeddings, embeddings);
        transpose_out_of_place(blocks[i].attention.v_gen.A, v, embeddings, embeddings);
        transpose_out_of_place(blocks[i].attention.proj.A,  p, embeddings, embeddings);

        f32_to_f16(blocks[i].mlp.fc1.A, fc1    , embeddings * k_embeddings);
        f32_to_f16(blocks[i].mlp.fc1.b, b1_data, k_embeddings);
        f32_to_f16(blocks[i].mlp.fc2.A, fc2    , embeddings * k_embeddings);
        f32_to_f16(blocks[i].mlp.fc2.b, b2_data, embeddings );

        blk_w.emplace_back(
            n1_bias, n1_scale,
            n2_bias, n2_scale,
            q ,
            k ,
            v ,
            p ,
            qb,
            kb,
            vb,
            pb,
            fc1    ,
            b1_data,
            fc2    ,
            b2_data
        );
    }

    out = std::move(blk_w);
}

pred_head_weights convert_pred_head(VisionTransformer &cpu_vit){
    linear_data lin_head = cpu_vit.get_head();
    layer_data ln = cpu_vit.get_norm();
    int embeddings = cpu_vit.get_embed_dim(); 
    int class_num = cpu_vit.get_num_classes();

    half * ln_scale    = (half *)malloc(sizeof(half) * embeddings);
    half * ln_bias     = (half *)malloc(sizeof(half) * embeddings);
    half * head_weights= (half *)malloc(sizeof(half) * embeddings * class_num);
    half * head_bias   = (half *)malloc(sizeof(half) * class_num);


    f32_to_f16(ln.g, ln_scale    , embeddings);
    f32_to_f16(ln.bias, ln_bias     , embeddings);
    f32_to_f16(lin_head.A, head_weights, embeddings * class_num);
    f32_to_f16(lin_head.b, head_bias   , class_num);

    return pred_head_weights(
        ln_scale,
        ln_bias,
        head_weights,
        head_bias
    );
}

void convert_vit_weights(
    VisionTransformer &vit,
    patch_emb_weights &pe_w,
    vector<block_weights> &blk_w,
    pred_head_weights &ph_w
){
    // - patch embedder
    pe_w = convert_patch_emb(vit);

    // - encoder blocks
    convert_blocks(vit, blk_w);

    
    // - prediction head
    ph_w = convert_pred_head(vit);
}

void GpuVit::set_class_buffers(){
    pe.set_buffers(
        d_pic,
        d_y,
        d_x,
        d_t,
        d_workspace
    );
    for(int i = 0; i < depth; i++){
        blocks[i].set_buffers(
            d_x,
            d_t,
            d_y,
            d_h,
            d_workspace
        );
    }
    ph.set_shared_buffers(
        d_x,
        d_t,
        d_y,
        d_x,
        d_workspace
    );

}

/**
 * @brief 
 * 
 */
GpuVit::GpuVit(
    cudaStream_t     &_stream,
    cudnnHandle_t    &_cudnn_handle,
    cublasLtHandle_t &_cublas_handle,

    convolution_dim _conv_dim,

    vit_size _tokens ,
    vit_size _num_classes ,
    vit_size _depth ,
    vit_size _num_heads ,
    vit_float _scale_val ,
    vit_size mlp_hidden ,

    vit_bool init_pe_descriptors,
    vit_bool allocate_pe_shared_ptrs, //initialize the weights shared pointers

    vit_bool block_mlp_kernel_type,
    vit_bool init_block_descriptors,
    vit_bool allocate_blocks_shared_ptrs //initialize the weights shared pointers
):
    stream       (_stream),
    cudnn_handle (_cudnn_handle),
    cublas_handle(_cublas_handle),
    conv_dim(_conv_dim),
    batch     (_conv_dim.batch ),
    img_h     (_conv_dim.height ),
    img_w     (_conv_dim.width ),
    patch_h   (_conv_dim.Ho ),
    patch_w   (_conv_dim.Wo ),
    channels  (_conv_dim.channels ),
    embeddings(_conv_dim.embeddings),

    num_classes(_num_classes),
    depth      (_depth),
    num_heads(_num_heads),
    scale_val(_scale_val),

    
    pe(
        _stream, _cudnn_handle, _conv_dim,
        allocate_pe_shared_ptrs, 
        init_pe_descriptors,
        allocate_pe_shared_ptrs
    ),
    blocks(),
    ph(
        _conv_dim.batch, _tokens + 1, _conv_dim.embeddings, _num_classes,
        _cudnn_handle, _cublas_handle, _stream
    )


{
    assert((img_h % patch_h == 0) && (img_w % patch_w == 0));
    tokens = (img_h / patch_h) * (img_w / patch_w);
    assert(tokens == _tokens);
    // blocks
    blocks.reserve(depth);
    for(int i = 0; i < depth; ++i){
        if(i == 0) //To have only 
            blocks.emplace_back(
                stream, cudnn_handle, cublas_handle,
                batch, tokens + 1, embeddings, mlp_hidden,
                block_mlp_kernel_type,
                block_epsilon, block_scale, num_heads,
                init_block_descriptors,
                allocate_blocks_shared_ptrs
            );   // blocks constructed here
        else
            blocks.emplace_back(
                stream, cudnn_handle, cublas_handle,
                batch, tokens + 1, embeddings, mlp_hidden,
                block_mlp_kernel_type,
                block_epsilon, block_scale, num_heads,
                false,
                allocate_blocks_shared_ptrs
            );
    }

    input_pic_elements_num = pe.get_input_pic_elem_n();
    embedded_elements_num = pe.get_embedded_elem_n();
    hidden_elements_number = blocks[0].get_hidden_elements_number();
    ph.epsilon = pred_head_epsilon;
}



// 0) Allocate on device the buffers used in all the ops
void GpuVit::allocate_shared_buffers(){
    assert((batch * num_classes) < embedded_elements_num);
    CUDA_CHECK(cudaMallocAsync(&d_pic    ,sizeof(half) * input_pic_elements_num, stream)); //[B,H,W,C]
    CUDA_CHECK(cudaMallocAsync(&d_x      ,sizeof(half) * embedded_elements_num, stream));
    CUDA_CHECK(cudaMallocAsync(&d_t      ,sizeof(half) * embedded_elements_num, stream));
    CUDA_CHECK(cudaMallocAsync(&d_y      ,sizeof(half) * embedded_elements_num, stream));
    CUDA_CHECK(cudaMallocAsync(&d_h      ,sizeof(half) * hidden_elements_number, stream));
    CUDA_CHECK(cudaMallocAsync(&d_workspace, WORKSPACE_SIZE, stream));            
    set_class_buffers();
}

// 1) Create all the descriptors for all the library functions used(cuBLAS and cuDNN)
void GpuVit::create_descriptors(){
    pe.init_descriptors();
    for (size_t i = 0; i < depth; i++)
    {
        blocks[i].init_descriptors();
    }

    ph.init_descriptors();
}

// 2) Allocate on device the buffers for the weights, also the workspace used for this block
void GpuVit::allocate_weights(){
    pe.allocate_weights();

    for (size_t i = 0; i < depth; i++)
    {
        blocks[i].allocate_weights();
    }

    ph.allocate_weights();
    
};

// 3) Load all the weights for each component to the device
void GpuVit::load_weights(
    patch_emb_weights &pe_w,
    vector<block_weights> &blk_w,
    pred_head_weights &ph_w
){
    CUDA_CHECK(cudaStreamSynchronize(stream)); //needed for the buffer allocation to finish before loading the weights

    pe.load_weights_data(
        pe_w.conv_w,
        pe_w.bias,
        pe_w.pos_emb,
        false
    );

    assert(blk_w.size() == depth);
    for (size_t i = 0; i < depth; i++)
    {

        blocks[i].load_weights(
            blk_w[i].n1_bias ,
            blk_w[i].n1_scale ,
            blk_w[i].n2_bias ,
            blk_w[i].n2_scale ,
            blk_w[i].fc1 ,
            blk_w[i].b1_data ,
            blk_w[i].fc2 ,
            blk_w[i].b2_data ,
            blk_w[i].attn_w
        );
    }

    ph.load_weights(
        ph_w.ln_scale,
        ph_w.ln_bias,
        ph_w.head_weights,
        ph_w.head_bias
    );
}


// 4) Load the input data to the model
void GpuVit::load_pics(half * pics){
    //put asyncronously data in d_pic, have to be `batch` pictures
    pe.load_pics(pics);
}


/* 5) Forward of the model, starts from d_pic result in d_x!
*/
void GpuVit::forward(){
    //Should all be happening in the same stream so no race conditions on the data
    
    pe.forward();
    
    for(int i = 0; i < depth; i++){
        blocks[i].forward();
    }
    
    ph.forward();

}


void GpuVit::print_dimensions(){
    cout << "   picture dimensions: " << 
    "["<< batch << ","<<channels  <<","<< img_h <<","<< img_w <<"]"<<endl;
    cout << "   patch emb. kernel dimensions: " << 
    "["<< embeddings << ","<<channels  <<","<< patch_h <<","<< patch_w <<"]"<<endl;
    cout << "   embedded dimensions: "<<
    "["<< batch << ","<< tokens  <<","<< embeddings <<"]"<<endl;
    cout << "   mlp dimensions: "<<
    "["<< embeddings << ","<< blocks[0].k_channels  << ","<< embeddings <<"]"<<endl;
    cout << "   num classes: " << num_classes << endl;
    cout << "   depth: " << depth << endl;
}


void GpuVit::free_weights(){
    pe.free_weights();
    
    for(int i = 0; i < depth; i++){
        blocks[i].free_weights();
    }

    ph.free_weights();
}

void GpuVit::free_buffers(){
    CUDA_CHECK(cudaFree(d_pic));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_t));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_h));
    CUDA_CHECK(cudaFree(d_workspace));
}     

void GpuVit::destroy_descriptors(){
    pe.destroy_descriptors();
    for(int i = 0; i < depth; i++){
        blocks[i].destroy_descriptors();
    }
    ph.destroy_descriptors();
}


void GpuVit::print_predictions(bool debug){
    cout << "   predicted classes: " << endl;
    for(int i = 0; i < batch; i++)
        cout << i << ": " << ph.class_prediction[i] << endl;
    if(debug){
        cout << "probabilities array\n[";
        for(int b = 0; b < batch; b++){   
            for(int i = 0; i< num_classes; i++)
                cout << ph.probabilities_array[b * num_classes + i] << " ";
            cout << "]\n";
        }
    }
}
