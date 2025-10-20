#include "../include/mlp.h"

#include "../include/datatypes.h"
#include "../include/modules.h"

#include <utility>
#include <assert.h>


Mlp::Mlp(
    vit_size _in_features,
    vit_size _hidden_features,
    vit_size _out_features,
    vit_float (*_act)(vit_float val),
    vit_bool use_bias,
    vit_bool _use_norm
) :
    fc1(_in_features, _hidden_features, use_bias),
    act(_act),
    norm(_hidden_features, 0.00001, true),
    fc2(_hidden_features, _out_features, use_bias)
{
    in_features = _in_features;
    hidden_features = _hidden_features;
    out_features = _out_features;
    use_norm = _use_norm;
}

Mlp::Mlp(Mlp&& mlp) :
    fc1(std::move(mlp.fc1)),
    act(mlp.act),
    norm(std::move(mlp.norm)),
    fc2(std::move(mlp.fc2))
{
    in_features = mlp.in_features;
    hidden_features = mlp.hidden_features;
    out_features = mlp.out_features;
    use_norm = mlp.use_norm;
}

Mlp::~Mlp() {}

Mlp& Mlp::operator= (Mlp&& mlp) {
    in_features = mlp.in_features;
    hidden_features = mlp.hidden_features;
    out_features = mlp.out_features;
    use_norm = mlp.use_norm;

    fc1 = std::move(mlp.fc1);
    act = mlp.act;
    norm = std::move(mlp.norm);
    fc2 = std::move(mlp.fc2);

    return *this;
}

vit_size Mlp::get_in_features() const {
    return in_features;
}

vit_size Mlp::get_hidden_features() const {
    return hidden_features;
}

vit_size Mlp::get_out_features() const {
    return out_features;
}

vit_bool Mlp::get_use_norm() const {
    return use_norm;
}

//TO DO
linear_data Mlp::get_fc1()
{
    linear_data data(
            fc1.get_A(),
            fc1.get_b(),
            fc1.get_in_features(),
            fc1.get_out_features(),
            fc1.get_use_bias()  
        );
    return data;
}
linear_shape Mlp::get_fc1_shape()
{
    int mtx_shape[2];
    fc1.get_A_shape(mtx_shape);
    linear_shape data(
        mtx_shape[0],
        mtx_shape[1],
        fc1.get_b_dim()
    );
    return data;
}
vit_float (*Mlp::get_act() const)(vit_float)
{
    return act.get_act();
}

layer_data Mlp::get_norm(){
    layer_data data(
        norm.get_g(),
        norm.get_bias(),
        norm.get_eps(),
        norm.get_use_bias()
    );
    return data;
}
layer_shape Mlp::get_norm_shape(){
    layer_shape data(      
        norm.get_g_dim(), // get_in_chans() //should be for single element case
        norm.get_g_dim()
    );
    return data;
};
linear_data Mlp::get_fc2(){
    linear_data data(
            fc2.get_A(),
            fc2.get_b(),
            fc2.get_in_features(),
            fc2.get_out_features(),
            fc2.get_use_bias()  
        );
    return data;
}
linear_shape Mlp::get_fc2_shape()
{
    int mtx_shape[2];
    fc2.get_A_shape(mtx_shape);
    linear_shape data(
        mtx_shape[0],
        mtx_shape[1],
        fc2.get_b_dim()
    );
    return data;
};

//
void Mlp::move_fc1(Linear &_fc1)
{
    fc1 = std::move(_fc1);
}

void Mlp::set_act(const Activation _act) {
    act = _act;
}

void Mlp::move_norm(LayerNorm& _norm) {
    norm = std::move(_norm);
}

void Mlp::move_fc2(Linear& _fc2) {
    fc2 = std::move(_fc2);
}

void Mlp::to_ofstream(std::ofstream& os) const {
    assert( os.is_open() );

    os.write( (char*) &in_features, sizeof(vit_size));
    os.write( (char*) &hidden_features, sizeof(vit_size));
    os.write( (char*) &out_features, sizeof(vit_size));
    os.write( (char*) &use_norm, sizeof(vit_bool));

    fc1.to_ofstream(os);
    act.to_ofstream(os);
    if (use_norm == true) {
        norm.to_ofstream(os);
    }
    fc2.to_ofstream(os);
}

void Mlp::from_ifstream(std::ifstream& is) {
    assert( is.is_open() );

    is.read( (char*) &in_features, sizeof(vit_size));
    is.read( (char*) &hidden_features, sizeof(vit_size));
    is.read( (char*) &out_features, sizeof(vit_size));
    is.read( (char*) &use_norm, sizeof(vit_bool));

    fc1.from_ifstream(is);
    act.from_ifstream(is);
    if (use_norm == true) {
        norm.from_ifstream(is);
    }
    fc2.from_ifstream(is);
}

void Mlp::forward(const Tensor& x_in, Tensor& x_out) const {
    assert(x_in.get_C() == in_features);

    fc1(x_in, x_out);
    act(x_out);
    if (use_norm == true) {
        norm(x_out);
    }
    fc2(x_out, x_out);
}
