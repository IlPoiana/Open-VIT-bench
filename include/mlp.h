#ifndef __MLP_H__
#define __MLP_H__

#include "datatypes.h"
#include "modules.h"



class Mlp {
private:
    vit_size in_features;
    vit_size hidden_features;
    vit_size out_features;
    vit_bool use_norm;

    Linear fc1;
    Activation act;
    LayerNorm norm; // subject to use_norm
    Linear fc2;
public:
    Mlp(
        vit_size _in_features, vit_size _hidden_features, vit_size _out_features,
        vit_float (*_act)(vit_float val), vit_bool use_bias=true, vit_bool _use_norm=false
    );
    Mlp(const Mlp& mlp) = delete;
    Mlp(Mlp&& mlp);
    ~Mlp();

    Mlp& operator= (const Mlp& mlp) = delete;
    Mlp& operator= (Mlp&& mlp);

    vit_size get_in_features() const;
    vit_size get_hidden_features() const;
    vit_size get_out_features() const;
    vit_bool get_use_norm() const;

    void move_fc1(Linear& _fc1);
    void set_act(const Activation _act);
    void move_norm(LayerNorm& _norm);
    void move_fc2(Linear& _fc2);

    void to_ofstream(std::ofstream& os) const;
    void from_ifstream(std::ifstream& is);

    void forward(const Tensor& x_in, Tensor& x_out) const;
};



#endif // __MLP_H__
