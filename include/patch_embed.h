#ifndef __PATCH_EMBED_H__
#define __PATCH_EMBED_H__

#include "datatypes.h"
#include "modules.h"
#include "conv2d.h"



class PatchEmbed {
private:
    vit_size img_size_h;
    vit_size img_size_w;
    vit_size patch_size_h;
    vit_size patch_size_w;

    vit_size grid_size_h;
    vit_size grid_size_w;
    vit_size num_patches;

    vit_size in_chans;
    vit_size embed_dim;

    vit_bool strict_img_size;
    vit_bool dynamic_img_pad;
    vit_bool use_norm;

    Conv2d c2d;
    LayerNorm norm; // subject to use_norm
public:
    PatchEmbed(
        vit_size _img_size_h = 224,
        vit_size _img_size_w = 224,
        vit_size _patch_size_h = 16,
        vit_size _patch_size_w = 16,
        vit_size _in_chans = 3,
        vit_size _embed_dim = 768,
        vit_bool c2d_bias = true,
        vit_bool _strict_img_size = true,
        vit_bool _dynamic_img_pad = false,
        vit_bool _use_norm = false
    );
    PatchEmbed(const PatchEmbed& pe) = delete;
    PatchEmbed(PatchEmbed&& pe);
    ~PatchEmbed();

    PatchEmbed& operator= (const PatchEmbed& pe) = delete;
    PatchEmbed& operator= (PatchEmbed&& pe);

    void get_image_size(vit_size& _img_size_h, vit_size& _img_size_w) const;
    void get_patch_size(vit_size& _patch_size_h, vit_size& _patch_size_w) const;
    void get_grid_size(vit_size& _grid_size_h, vit_size& _grid_size_w) const;

    vit_size get_in_chans() const;
    vit_size get_embed_dim() const;
    vit_size get_num_patches() const;

    vit_bool get_strict_img_size() const;
    vit_bool get_dynamic_img_pad() const;
    vit_bool get_use_norm() const;

    vit_size get_feat_ratio() const;
    void get_dynamic_feat_size(
        vit_size _img_size_h, vit_size _img_size_w,
        vit_size& _grid_size_h, vit_size& _grid_size_w
    ) const;

    void move_c2d(Conv2d& _c2d);
    void move_norm(LayerNorm& _norm);

    void to_ofstream(std::ofstream& os) const;
    void from_ifstream(std::ifstream& is);

    void forward(const PictureBatch& p_in, Tensor& x_out) const;
};



#endif // __PATCH_EMBED_H__
