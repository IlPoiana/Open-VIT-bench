#include "../include/block.h"

#include "../include/datatypes.h"
#include "../include/modules.h"
#include "../include/mlp.h"
#include "../include/attention.h"

#include <utility>
#include <assert.h>
#include <chrono>


Block::Block(
    vit_size _dim,
    vit_size _num_heads,
    vit_float _mlp_ratio,
    vit_bool use_qkv_bias,
    vit_bool use_qk_norm,
    vit_float scale_val, 
    vit_float (*act)(vit_float val)
) :
    norm1(_dim, 0.00001, true),
    attn(_dim, _num_heads, use_qkv_bias, use_qk_norm),
    ls1(_dim, scale_val),
    norm2(_dim, 0.00001, true),
    mlp(_dim, _dim*_mlp_ratio, _dim, act, true, false),
    ls2(_dim, scale_val)
{
    dim = _dim;
    num_heads = _num_heads;
    mlp_ratio = _mlp_ratio;
}

Block::Block(Block&& b) :
    norm1(std::move(b.norm1)),
    attn(std::move(b.attn)),
    ls1(b.ls1),
    norm2(std::move(b.norm2)),
    mlp(std::move(b.mlp)),
    ls2(b.ls2)
{
    dim = b.dim;
    num_heads = b.num_heads;
    mlp_ratio = b.mlp_ratio;
}

Block::~Block() {}

Block& Block::operator= (Block&& b) {
    dim = b.dim;
    num_heads = b.num_heads;
    mlp_ratio = b.mlp_ratio;

    norm1 = std::move(b.norm1);
    attn = std::move(b.attn);
    ls1 = b.ls1;
    norm2 = std::move(b.norm2);
    mlp = std::move(b.mlp);
    ls2 = b.ls2;

    return *this;
}

vit_size Block::get_dim() const {
    return dim;
}

vit_size Block::get_num_heads() const {
    return num_heads;
}

vit_float Block::get_mlp_ratio() const {
    return mlp_ratio;
}

layer_data Block::get_norm1()
{
    layer_data data(
        norm1.get_g(),
        norm1.get_bias(),
        norm1.get_eps(),
        norm1.get_use_bias()
    );
    return data;
}
layer_shape Block::get_norm1_shape()
{
    layer_shape data(      
        norm1.get_g_dim(), // get_in_chans() //should be for single element case
        norm1.get_g_dim()
    );
    return data;
}

attn_data Block::get_attention()
{
    attn_data data(
        attn.get_q_gen(),
        attn.get_k_gen(),
        attn.get_v_gen(),
        attn.get_q_norm(), // subject to use_qk_norm
        attn.get_k_norm(), // subject to use_qk_norm
        attn.get_proj(),
        attn.get_dim(),
        attn.get_num_heads(),
        attn.get_head_dim(),
        attn.get_scale(),
        attn.get_use_qk_norm()
    );
    return data;
}
attn_shape Block::get_attention_shape()
{
    attn_shape data(
        attn.get_q_gen_shape(),
        attn.get_k_gen_shape(),
        attn.get_v_gen_shape(),
        attn.get_q_norm_shape(),
        attn.get_k_norm_shape(),
        attn.get_proj_shape()
    );
    return data;
}
// TO DO

scale_data Block::get_ls1()
{
    scale_data data(
        ls1.get_dim(),
        ls1.get_val()
    );
    return data;
}
layer_data Block::get_norm2()
{
    layer_data data(
        norm2.get_g(),
        norm2.get_bias(),
        norm2.get_eps(),
        norm2.get_use_bias()
    );
    return data;
}
layer_shape Block::get_norm2_shape()
{
    layer_shape data(      
        norm2.get_g_dim(), // get_in_chans() //should be for single element case
        norm2.get_g_dim()
    );
    return data;}
mlp_data Block::get_mlp()
{
    mlp_data data(
        mlp.get_in_features(),
        mlp.get_hidden_features(),
        mlp.get_out_features(),
        mlp.get_use_norm(),
        mlp.get_fc1(),
        mlp.get_norm(),
        mlp.get_act(),
        mlp.get_fc2()
    );
    return data;
}
mlp_shape Block::get_mlp_shape()
{
    mlp_shape data(
        mlp.get_fc1_shape(),
        mlp.get_norm_shape(),
        mlp.get_fc2_shape()
    );
    return data;
}
scale_data Block::get_ls2()
{
    scale_data data(    
        ls2.get_dim(),
        ls2.get_val()
    );
    return data;
}

//
void Block::move_norm1(LayerNorm &_norm1)
{
    norm1 = std::move(_norm1);
}

void Block::move_attn(Attention& _attn) {
    attn = std::move(_attn);
}

void Block::set_ls1_val(vit_float val) {
    ls1.set_val(val);
}

void Block::move_norm2(LayerNorm& _norm2) {
    norm2 = std::move(_norm2);
}

void Block::move_mlp(Mlp& _mlp) {
    mlp = std::move(_mlp);
}

void Block::set_ls2_val(vit_float val) {
    ls2.set_val(val);
}

void Block::to_ofstream(std::ofstream& os) const {
    assert( os.is_open() );

    os.write( (char*) &dim, sizeof(vit_size));
    os.write( (char*) &num_heads, sizeof(vit_size));
    os.write( (char*) &mlp_ratio, sizeof(vit_float));

    norm1.to_ofstream(os);
    attn.to_ofstream(os);
    ls1.to_ofstream(os);
    norm2.to_ofstream(os);
    mlp.to_ofstream(os);
    ls2.to_ofstream(os);
}

void Block::from_ifstream(std::ifstream& is) {
    assert( is.is_open() );

    is.read( (char*) &dim, sizeof(vit_size));
    is.read( (char*) &num_heads, sizeof(vit_size));
    is.read( (char*) &mlp_ratio, sizeof(vit_float));

    norm1.from_ifstream(is);
    attn.from_ifstream(is);
    ls1.from_ifstream(is);
    norm2.from_ifstream(is);
    mlp.from_ifstream(is);
    ls2.from_ifstream(is);
}

void Block::forward(const Tensor& x_in, Tensor& x_out) const {
    Tensor y;
    y.copy_tensor(x_in);
    norm1(y);
    attn.forward(y, y);
    ls1(y);
    y += x_in;

    x_out.copy_tensor(y);
    norm2(x_out);
    mlp.forward(x_out, x_out);
    ls2(x_out);
    x_out += y;
}

//Encoder block forward 
void Block::timed_forward(
    const Tensor& x_in,
    Tensor& x_out,
    vit_float& attn_time,
    vit_float& mlp_time
) const {
    auto start_time = std::chrono::high_resolution_clock::now();
    Tensor y;
    y.copy_tensor(x_in);
    norm1(y);
    attn.forward(y, y);
    ls1(y);
    y += x_in;
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    attn_time = elapsed.count();

    start_time = std::chrono::high_resolution_clock::now();
    x_out.copy_tensor(y);
    norm2(x_out);
    mlp.forward(x_out, x_out);
    ls2(x_out);
    x_out += y;
    end_time = std::chrono::high_resolution_clock::now();
    elapsed = end_time - start_time;
    mlp_time = elapsed.count();
}
