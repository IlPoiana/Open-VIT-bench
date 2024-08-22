#include "../include/patch_embed.h"

#include "../include/datatypes.h"
#include "../include/modules.h"
#include "../include/conv2d.h"

#include <utility>
#include <assert.h>



PatchEmbed::PatchEmbed(
    vit_size _img_size_h,
    vit_size _img_size_w,
    vit_size _patch_size_h,
    vit_size _patch_size_w,
    vit_size _in_chans,
    vit_size _embed_dim,
    vit_bool c2d_bias,
    vit_bool _strict_img_size,
    vit_bool _dynamic_img_pad,
    vit_bool _use_norm
) :
    c2d(
        _in_chans, _embed_dim, _patch_size_h, _patch_size_w,
        _patch_size_h, _patch_size_w, c2d_bias
    ),
    norm(_embed_dim, 0.00001, true)
{
    img_size_h = _img_size_h;
    img_size_w = _img_size_w;
    patch_size_h = _patch_size_h;
    patch_size_w = _patch_size_w;

    assert(_img_size_h % patch_size_h == 0);
    grid_size_h = _img_size_h / patch_size_h;
    assert(_img_size_w % patch_size_w == 0);
    grid_size_w = _img_size_w / patch_size_w;
    num_patches = grid_size_h * grid_size_w;

    in_chans = _in_chans;
    embed_dim = _embed_dim;
    strict_img_size = _strict_img_size;
    dynamic_img_pad = _dynamic_img_pad;
    use_norm = _use_norm;
}

PatchEmbed::PatchEmbed(PatchEmbed&& pe) :
    c2d(std::move(pe.c2d)),
    norm(std::move(pe.norm))
{
    img_size_h = pe.img_size_h;
    img_size_w = pe.img_size_w;
    patch_size_h = pe.patch_size_h;
    patch_size_w = pe.patch_size_w;
    grid_size_h = pe.grid_size_h;
    grid_size_w = pe.grid_size_w;
    num_patches = pe.num_patches;

    in_chans = pe.in_chans;
    embed_dim = pe.embed_dim;
    strict_img_size = pe.strict_img_size;
    dynamic_img_pad = pe.dynamic_img_pad;
    use_norm = pe.use_norm;
}

PatchEmbed::~PatchEmbed() {}

PatchEmbed& PatchEmbed::operator= (PatchEmbed&& pe) {
    img_size_h = pe.img_size_h;
    img_size_w = pe.img_size_w;
    patch_size_h = pe.patch_size_h;
    patch_size_w = pe.patch_size_w;
    grid_size_h = pe.grid_size_h;
    grid_size_w = pe.grid_size_w;
    num_patches = pe.num_patches;

    in_chans = pe.in_chans;
    embed_dim = pe.embed_dim;
    strict_img_size = pe.strict_img_size;
    dynamic_img_pad = pe.dynamic_img_pad;
    use_norm = pe.use_norm;

    c2d = std::move(pe.c2d);
    norm = std::move(pe.norm);

    return *this;
}

void PatchEmbed::get_image_size(vit_size& _img_size_h, vit_size& _img_size_w) const {
    _img_size_h = img_size_h;
    _img_size_w = img_size_w;
}

void PatchEmbed::get_patch_size(vit_size& _patch_size_h, vit_size& _patch_size_w) const {
    _patch_size_h = patch_size_h;
    _patch_size_w = patch_size_w;
}

void PatchEmbed::get_grid_size(vit_size& _grid_size_h, vit_size& _grid_size_w) const {
    _grid_size_h = grid_size_h;
    _grid_size_w = grid_size_w;
}

vit_size PatchEmbed::get_in_chans() const {
    return in_chans;
}

vit_size PatchEmbed::get_embed_dim() const {
    return embed_dim;
}

vit_size PatchEmbed::get_num_patches() const {
    return num_patches;
}

vit_bool PatchEmbed::get_strict_img_size() const {
    return strict_img_size;
}

vit_bool PatchEmbed::get_dynamic_img_pad() const {
    return dynamic_img_pad;
}

vit_bool PatchEmbed::get_use_norm() const {
    return use_norm;
}

vit_size PatchEmbed::get_feat_ratio() const {
    return patch_size_h>=patch_size_w ? patch_size_h : patch_size_w;
}

void PatchEmbed::get_dynamic_feat_size(
    vit_size _img_size_h,
    vit_size _img_size_w,
    vit_size& _grid_size_h,
    vit_size& _grid_size_w
) const {
    _grid_size_h = _img_size_h / patch_size_h;
    _grid_size_w = _img_size_w / patch_size_w;
    if (dynamic_img_pad == true) {
        if (_img_size_h % patch_size_h != 0) {
            ++_grid_size_h; // to compensate the floor of integer division
        }
        if (_img_size_w % patch_size_w != 0) {
            ++_grid_size_w; // to compensate the floor of integer division
        }
    }
}

void PatchEmbed::move_c2d(Conv2d& _c2d) {
    c2d = std::move(_c2d);
}

void PatchEmbed::move_norm(LayerNorm& _norm) {
    norm = std::move(_norm);
}

void PatchEmbed::to_ofstream(std::ofstream& os) const {
    assert( os.is_open() );

    os.write( (char*) &img_size_h, sizeof(vit_size));
    os.write( (char*) &img_size_w, sizeof(vit_size));
    os.write( (char*) &patch_size_h, sizeof(vit_size));
    os.write( (char*) &patch_size_w, sizeof(vit_size));

    os.write( (char*) &grid_size_h, sizeof(vit_size));
    os.write( (char*) &grid_size_w, sizeof(vit_size));
    os.write( (char*) &num_patches, sizeof(vit_size));

    os.write( (char*) &in_chans, sizeof(vit_size));
    os.write( (char*) &embed_dim, sizeof(vit_size));

    os.write( (char*) &strict_img_size, sizeof(vit_bool));
    os.write( (char*) &dynamic_img_pad, sizeof(vit_bool));
    os.write( (char*) &use_norm, sizeof(vit_bool));

    c2d.to_ofstream(os);
    if (use_norm == true) {
        norm.to_ofstream(os);
    }
}

void PatchEmbed::from_ifstream(std::ifstream& is) {
    assert( is.is_open() );

    is.read( (char*) &img_size_h, sizeof(vit_size));
    is.read( (char*) &img_size_w, sizeof(vit_size));
    is.read( (char*) &patch_size_h, sizeof(vit_size));
    is.read( (char*) &patch_size_w, sizeof(vit_size));

    is.read( (char*) &grid_size_h, sizeof(vit_size));
    is.read( (char*) &grid_size_w, sizeof(vit_size));
    is.read( (char*) &num_patches, sizeof(vit_size));

    is.read( (char*) &in_chans, sizeof(vit_size));
    is.read( (char*) &embed_dim, sizeof(vit_size));

    is.read( (char*) &strict_img_size, sizeof(vit_bool));
    is.read( (char*) &dynamic_img_pad, sizeof(vit_bool));
    is.read( (char*) &use_norm, sizeof(vit_bool));

    c2d.from_ifstream(is);
    if (use_norm == true) {
        norm.from_ifstream(is);
    }
}

void PatchEmbed::forward(const PictureBatch& p_in, Tensor& x_out) const {
    if (strict_img_size == true) {
        assert(p_in.get_H() == img_size_h);
        assert(p_in.get_W() == img_size_w);
    }

    PictureBatch p_conv;
    if (dynamic_img_pad == true) {
        vit_size h_pad = patch_size_h - (p_in.get_H()%patch_size_h);
        vit_size w_pad = patch_size_w - (p_in.get_W()%patch_size_w);
        if (h_pad == patch_size_h && w_pad == patch_size_w) {
            c2d.forward(p_in, p_conv);
        } else {
            p_in.get_pad(p_conv, p_in.get_H()+h_pad, p_in.get_W()+w_pad);
            c2d.forward(p_conv, p_conv);
        }
    } else {
        assert(p_in.get_H() % patch_size_h == 0);
        assert(p_in.get_W() % patch_size_w == 0);
        c2d.forward(p_in, p_conv);
    }

    p_conv.flatten_to_tensor(x_out);

    if (use_norm == true) {
        norm(x_out);
    }
}
