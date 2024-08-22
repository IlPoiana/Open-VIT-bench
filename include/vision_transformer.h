#ifndef __VISION_TRANSFORMER_H__
#define __VISION_TRANSFORMER_H__

#include <vector>

#include "datatypes.h"
#include "modules.h"
#include "block.h"
#include "patch_embed.h"



class VisionTransformer {
private:
    vit_size num_classes;
    pool_type global_pool;
    vit_size embed_dim;
    vit_size depth;

    vit_bool has_class_token;
    vit_size num_reg_tokens;
    vit_size num_prefix_tokens;
    vit_bool no_embed_class;

    vit_bool use_pos_embed;
    vit_bool use_pre_norm;
    vit_bool use_fc_norm;
    vit_bool dynamic_img_size;

    RowVector cls_token;
    Matrix reg_token;
    Matrix pos_embed;

    PatchEmbed patch_embed;
    LayerNorm pre_norm; // subjected to use_pre_norm
    std::vector<Block> blocks;
    LayerNorm norm; // subjected to not use_fc_norm
    LayerNorm fc_norm; // subjected to use_fc_norm
    Linear head;
public:
    VisionTransformer(
        vit_size img_size_h = 224,
        vit_size img_size_w = 224,
        vit_size patch_size_h = 16,
        vit_size patch_size_w = 16,
        vit_size in_chans = 3,

        vit_size _num_classes = 1000,
        pool_type _global_pool = pool_token,
        vit_size _embed_dim = 768,
        vit_size _depth = 12,

        vit_size num_heads = 12,
        vit_float mlp_ratio = 4,
        vit_bool use_qkv_bias = true,
        vit_bool use_qk_norm = false,
        vit_float scale_val = 1.0,

        vit_bool _has_class_token = true,
        vit_size _num_reg_tokens = 0,
        vit_bool _no_embed_class = false,

        vit_bool _use_pos_embed = true,
        vit_bool _use_pre_norm = false,
        vit_bool _use_fc_norm = false,

        vit_bool _dynamic_img_size = false,
        vit_bool dynamic_img_pad = false,
        vit_float (*act)(vit_float val) = GELU
    );
    VisionTransformer(const VisionTransformer& vit) = delete;
    VisionTransformer(VisionTransformer&& vit);
    ~VisionTransformer();

    VisionTransformer& operator= (const VisionTransformer& vit) = delete;
    VisionTransformer& operator= (VisionTransformer&& vit);

    vit_size get_num_classes() const;
    pool_type get_global_pool() const;
    vit_size get_embed_dim() const;
    vit_size get_depth() const;

    vit_bool get_has_class_token() const;
    vit_size get_num_reg_tokens() const;
    vit_size get_num_prefix_tokens() const;
    vit_bool get_no_embed_class() const;

    vit_bool get_use_pos_embed() const;
    vit_bool get_use_pre_norm() const;
    vit_bool get_use_fc_norm() const;
    vit_bool get_dynamic_img_size() const;

    void move_cls_token(RowVector _cls_token);
    void move_reg_token(Matrix _reg_token);
    void move_pos_embed(Matrix _pos_embed);

    void move_patch_embed(PatchEmbed _patch_embed);
    void move_pre_norm(LayerNorm _pre_norm);
    void move_blocks(std::vector<Block> _blocks);
    void move_norm(LayerNorm _norm);
    void move_fc_norm(LayerNorm _fc_norm);
    void move_head(Linear _head);

    void reset_classifier(Linear _head, vit_size _num_classes, pool_type _global_pool);

    void to_ofstream(std::ofstream& os) const;
    void from_ifstream(std::ifstream& is);

    void position_embed(const Tensor& x_in, Tensor& x_out) const;
    void forward_features(const PictureBatch& p_in, Tensor& x_out) const;
    void pool(const Tensor& x_in, Tensor& x_out) const;
    void forward_head(const Tensor& x_in, Tensor& x_out) const;

    void forward(const PictureBatch& pic, PredictionBatch& pred) const;

    void timed_forward(const PictureBatch& pic, PredictionBatch& pred, RowVector& times) const;
};



#endif // __VISION_TRANSFORMER_H__
