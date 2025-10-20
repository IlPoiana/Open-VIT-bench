// #include "../gpu_include/cuda_utils.h"
#include "../gpu_include/gpu_vit.h"


// Prepare the ViT in GPU memory and the cudaGraph (missing only the input image)
GpuViT::GpuViT(VisionTransformer & vit)
{
    //Could be, call a function that initialize the tensors and creates the cudaGraph?
    // vit_float * get_conv2d_kernel();
    // vit_float * get_conv2d_bias();
    // void get_kernel_shape(int kernel_shape[6]);
    // layer_data get_patch_layer_norm();
    // layer_shape get_patch_layer_shape();
    
    // vit_float *  get_cls_token();
    // vit_size get_cls_token_shape();
    
    // vit_float * get_reg_token();
    // void get_reg_token_shape(int reg_token_shape[2]);
    // vit_float * get_pos_embed();
    // void get_pos_embed_shape(int pos_embed_shape[2]);


    // layer_data get_pre_norm(); // TO DO
    // layer_shape get_pre_norm_shape();
    // layer_data get_norm(); // TO DO
    // layer_shape get_norm_shape();
    // layer_data get_fc_norm();// TO DO
    // layer_shape get_fc_norm_shape();



    // void get_blocks(blocks_data data[]);
    // void get_blocks_shape(blocks_shape shapes[]);
    
    // linear_data get_head();
    // linear_shape get_head_shape();

    auto yesno = [](vit_bool b){ return b ? "true" : "false"; };

    cout << "=== VisionTransformer attributes ===" << endl;

    // Basic config
    cout << "num_classes        : " << vit.get_num_classes() << endl;
    cout << "global_pool        : " << static_cast<int>(vit.get_global_pool()) << "  (0=token,1=avg,2=avgmax,3=max)" << endl;
    cout << "embed_dim          : " << vit.get_embed_dim() << endl;
    cout << "depth (#blocks)    : " << vit.get_depth() << endl;

    // Tokens / prefix
    cout << "has_class_token    : " << yesno(vit.get_has_class_token()) << endl;
    cout << "num_reg_tokens     : " << vit.get_num_reg_tokens() << endl;
    cout << "num_prefix_tokens  : " << vit.get_num_prefix_tokens() << endl;
    cout << "no_embed_class     : " << yesno(vit.get_no_embed_class()) << endl;

    // Booleans / modes
    cout << "use_pos_embed      : " << yesno(vit.get_use_pos_embed()) << endl;
    cout << "use_pre_norm       : " << yesno(vit.get_use_pre_norm()) << endl;
    cout << "use_fc_norm        : " << yesno(vit.get_use_fc_norm()) << endl;
    cout << "dynamic_img_size   : " << yesno(vit.get_dynamic_img_size()) << endl;
    //Patch Embedder
    int kshape[6] = {0,0,0,0,0,0};
    vit.get_kernel_shape(kshape);
    // Convention here is whatever the header/provider defined; we just echo the 6-tuple.
    cout << "\n-- PatchEmbed / Conv2D --" << endl;
    cout << "kernel_shape       : [" << kshape[0] << "," << kshape[1] << "," << kshape[2]
            << "," << kshape[3] << "," << kshape[4] << "," << kshape[5] << "]" << endl;

    
    vit_float* kptr = vit.get_conv2d_kernel();
    vit_float* bptr = vit.get_conv2d_bias();
    cout << "kernel sample      : " << (kptr ? std::to_string(kptr[0]) : "null") << endl;
    cout << "bias sample        : " << (bptr ? std::to_string(bptr[0]) : "null") << endl;

    layer_shape pln_s = vit.get_patch_layer_shape();
    cout << "patch LN g_size    : " << pln_s.g_size
            << "  bias_size: " << pln_s.bias_size << endl;
    cout << "layer_use_norm   : " << yesno(vit.get_layer_use_norm()) << endl;
    if(vit.get_layer_use_norm()){
        layer_data  pln   = vit.get_patch_layer_norm();    
        cout << "  eps: " << pln.eps
            << "  use_bias: " << yesno(pln.use_bias) << endl;
    }

    cout << "\n-- Blocks --" << endl;
    u_int depth = vit.get_blocks_number();
    vector<blocks_shape> block_s;
    block_s = vit.get_blocks_shape();
    vector<blocks_data> blocks;
    blocks = vit.get_blocks();
	cout << " block attention dim: " << blocks[0].attention.dim << endl;
	cout << " block attention head_dim: " << blocks[0].attention.head_dim << endl;
	cout << " block attention num heads: " << blocks[0].attention.num_heads << endl;


    for(u_int idx = 0; idx < block_s.size(); idx++){
        cout << idx <<" block k_gen_shape: col " << block_s[idx].attention_shape.k_gen_shape.a_col <<
        " row: " <<  block_s[idx].attention_shape.k_gen_shape.a_row << endl;
        cout << idx <<" block attention dim: " << blocks[idx].attention.dim << endl;
        cout << idx <<" block layer norm shape: " << block_s[idx].norm1_shape.g_size << "- bias -" <<block_s[idx].norm1_shape.bias_size <<endl;
    }

    if(vit.get_use_pre_norm()){
        layer_shape pre_norm_s = vit.get_pre_norm_shape();
        cout << "patch LN g_size    : " << pre_norm_s.g_size
                << "  bias_size: " << pre_norm_s.bias_size << endl;
        layer_data pre_norm = vit.get_pre_norm();
        cout << "  eps: " << pre_norm.eps
            << "  use_bias: " << yesno(pre_norm.use_bias) << endl;
    }
    if(vit.get_use_fc_norm()){
        layer_shape n_s = vit.get_fc_norm_shape();
        cout << "patch LN g_size    : " << n_s.g_size
                << "  bias_size: " << n_s.bias_size << endl;
        layer_data n = vit.get_fc_norm();
        cout << "  eps: " << n.eps
            << "  use_bias: " << yesno(n.use_bias) << endl;
    }else{
        layer_shape n_s = vit.get_norm_shape();
        cout << "patch LN g_size    : " << n_s.g_size
                << "  bias_size: " << n_s.bias_size << endl;
        layer_data n = vit.get_norm();
        cout << "  eps: " << n.eps
            << "  use_bias: " << yesno(n.use_bias) << endl;
    }

}