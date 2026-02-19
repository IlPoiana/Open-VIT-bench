#include <cuda_fp16.h>
#include "../gpu_include/gpu_vit.h"

void cpu_gpu_comparison(int argc, char * argv[]){
    assert(argc > 2);
    
    //loading the batch
    //path to the cpic directory
    const string cpic_path = argv[1]; //data/pic_$i.cpic
    PictureBatch pic;
    load_cpic(cpic_path, pic);
    
    //Choose if pinned mem. or not
    int pinned = atoi(argv[2]); // 0 = no pinned, 1 = pinned

    //loading the kernel
    const string cvit_path = "models/vit_1.cvit";
    VisionTransformer vit;
    cout << "loading vit" << endl;
    load_cvit(cvit_path, vit);
    vit.print();

    PredictionBatch pb;
    vit.forward(pic, pb);
    cout << "CPU reference: " << endl;
    pb.print();

    // -- GPU --
    int input_pics_elements_number = pic.get_B() * pic.get_C() * pic.get_H() *pic.get_W() ;
    half * gpu_pics_minibatch;
    if(pinned == 1){
        CUDA_CHECK(cudaHostAlloc(
            (void**)&gpu_pics_minibatch, sizeof(half) * input_pics_elements_number, cudaHostAllocPortable
        ));
    } else {
        gpu_pics_minibatch = (half*)malloc(sizeof(half) * input_pics_elements_number);
    }
    f32_to_f16(pic.get_data(), gpu_pics_minibatch, input_pics_elements_number);



    int pe_size[6];
    vit.get_kernel_shape(pe_size);
    int batch = pic.get_B();
    int channels = pe_size[0];
    int embeddings = pe_size[1];
    int patch_h  = pe_size[2];
    int patch_w  = pe_size[3];

    vit_size img_h;
    vit_size img_w;
    vit.get_img_size(img_h,img_w);
    
    int num_classes = vit.get_num_classes();
    int depth       = vit.get_depth();
    int num_heads   = (vit.get_blocks())[0].attention.num_heads;
    float scale_val   = 1.0f; //VisionTransformer(C reference) default and only option for now
    float epsilon     = 1e-6; 
    int mlp_hidden  = vit.get_blocks_shape()[0].mlperc_shape.fc1_shape.a_row;
    /*Bool flags for the descriptors and buffers allocation*/
    bool init_pe_descriptors            = false; 
    bool allocate_pe_shared_ptrs        = false; 
    bool block_mlp_kernel_type          = true; //Using fused kernel
    bool allocate_blocks_shared_ptrs    = false; 

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cublasLtHandle_t cublas_handle;
    CUBLAS_CHECK(cublasLtCreate(&cublas_handle));
    cudnnHandle_t cudnn_handle;
    CUDNN_CHECK(cudnnCreate(&cudnn_handle));
    CUDNN_CHECK(cudnnSetStream(cudnn_handle,stream));

    convolution_dim conv_dim(
        batch, channels, img_h, img_w,
        embeddings,
        patch_h, patch_w
    );
    int tokens = conv_dim.y_height * conv_dim.y_width;

    cout << "-- DIMENSIONS --" << endl;
    cout << "channels  : "<< channels    << endl;
    cout << "embeddings: "<< embeddings  << endl;
    cout << "patch_h   : "<< patch_h     << endl;
    cout << "patch_w   : "<< patch_w     << endl;
    cout << "tokens    : "<< tokens     << endl;
    cout << "num_classes: " << num_classes << endl; 
    cout << "depth      : " << depth       << endl; 
    cout << "num_heads  : " << num_heads   << endl; 
    cout << "scale_val  : " << scale_val   << endl; 
    cout << "mlp_hidden  : " << mlp_hidden   << endl; 
    cout << " ---- " << endl;

    GpuVit gpu_vit(
        stream, cudnn_handle, cublas_handle,
        conv_dim,
        tokens, num_classes,
        depth, num_heads, scale_val,
        epsilon, epsilon,
        mlp_hidden,
        init_pe_descriptors, 
        allocate_pe_shared_ptrs, 
        block_mlp_kernel_type, 
        allocate_blocks_shared_ptrs
    );

    gpu_vit.print_dimensions();

    gpu_vit.allocate_shared_buffers(); //d_x, d_y, d_pred, d_t....

    gpu_vit.create_descriptors(); // all the descriptors
    
    gpu_vit.allocate_weights(); //d_ln_bias and scale, cudnn_weights ....

    patch_emb_weights pe_w;
    vector<block_weights> blk_w;
    pred_head_weights ph_w;
    convert_vit_weights(vit, pe_w, blk_w, ph_w);

    gpu_vit.load_weights(pe_w, blk_w, ph_w); // every weight used in the model
    
    //Needs to be in NCHW format, where N == batch
    gpu_vit.load_pics(gpu_pics_minibatch);
    gpu_vit.forward();
    gpu_vit.compute_predictions();
    gpu_vit.print_predictions(true);
   
    gpu_vit.free_buffers();       
    gpu_vit.free_weights(); //shared weights!
    gpu_vit.destroy_descriptors();

    // -Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUBLAS_CHECK(cublasLtDestroy(cublas_handle));
    CUDNN_CHECK(cudnnDestroy(cudnn_handle));

    if(pinned == 1){
        CUDA_CHECK(cudaFreeHost(gpu_pics_minibatch));
    }

    return;
}

int main(int argc, char * argv[]) {
    cpu_gpu_comparison(argc, argv);

    return 0;
}