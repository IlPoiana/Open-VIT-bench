#ifndef __BLOCK_H__
#define __BLOCK_H__

#include "datatypes.h"
#include "modules.h"
#include "mlp.h"
#include "attention.h"



class Block {
private:
    vit_size dim;
    vit_size num_heads;
    vit_float mlp_ratio;

    LayerNorm norm1;
    Attention attn;
    LayerScale ls1;
    LayerNorm norm2;
    Mlp mlp;
    LayerScale ls2;
public:
    Block(
        vit_size _dim, vit_size _num_heads, vit_float _mlp_ratio,
        vit_bool use_qkv_bias, vit_bool use_qk_norm, vit_float scale_val, vit_float (*act)(vit_float val)
    );
    Block(const Block& b) = delete;
    Block(Block&& b);
    ~Block();

    Block& operator= (const Block& b) = delete;
    Block& operator= (Block&& b);

    vit_size get_dim() const;
    vit_size get_num_heads() const;
    vit_float get_mlp_ratio() const;

    void move_norm1(LayerNorm& _norm1);
    void move_attn(Attention& _attn);
    void set_ls1_val(vit_float val);
    void move_norm2(LayerNorm& _norm2);
    void move_mlp(Mlp& _mlp);
    void set_ls2_val(vit_float val);

    void to_ofstream(std::ofstream& os) const;
    void from_ifstream(std::ifstream& is);

    void forward(const Tensor& x_in, Tensor& x_out) const;

    void timed_forward(const Tensor& x_in, Tensor& x_out, vit_float& attn_time, vit_float& mlp_time) const;
};



#endif // __BLOCK_H__
