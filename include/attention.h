#ifndef __ATTENTION_H__
#define __ATTENTION_H__

#include "datatypes.h"
#include "modules.h"



class Attention {
private:
    vit_size dim;
    vit_size num_heads;
    vit_size head_dim;
    vit_float scale;
    vit_bool use_qk_norm;

    Linear q_gen;
    Linear k_gen;
    Linear v_gen;
    LayerNorm q_norm; // subject to use_qk_norm
    LayerNorm k_norm; // subject to use_qk_norm
    Linear proj;
public:
    Attention(vit_size _dim, vit_size _num_heads=8, vit_bool use_qkv_bias=false, vit_bool _use_qk_norm=false);
    Attention(const Attention& att) = delete;
    Attention(Attention&& attn);
    ~Attention();

    Attention& operator= (const Attention& attn) = delete;
    Attention& operator= (Attention&& attn);

    vit_size get_dim() const;
    vit_size get_num_heads() const;
    vit_size get_head_dim() const;
    vit_float get_scale() const;
    vit_bool get_use_qk_norm() const;

    void move_qkv_gen(Linear& _q_gen, Linear& _k_gen, Linear& _v_gen);
    void move_norms(LayerNorm& _q_norm, LayerNorm& _k_norm);
    void move_proj(Linear& _proj);

    void to_ofstream(std::ofstream& os) const;
    void from_ifstream(std::ifstream& is);

    void forward(const Tensor& x_in, Tensor& x_out) const;

    void single_head_attention(
        const Tensor& query, const Tensor& key, const Tensor& value, vit_float _scale, Tensor& x_out
    ) const;

    void multi_head_attention(
        const Tensor& query, const Tensor& key, const Tensor& value, vit_float _scale,
        Tensor& x_out, vit_size _num_heads, vit_size _head_dim
    ) const;
};



#endif // __ATTENTION_H__
