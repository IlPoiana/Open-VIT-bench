#ifndef __CONV2D_H__
#define __CONV2D_H__

#include "datatypes.h"



class Conv2d {
private:
    vit_size in_channels;
    vit_size out_channels;

    vit_size kernel_h;
    vit_size kernel_w;
    vit_size stride_h;
    vit_size stride_w;
    vit_bool use_bias;

    PictureBatch kernel;
    RowVector bias; // subject to use_bias
public:
    Conv2d(
        vit_size _in_channels, vit_size _out_channels, vit_size _kernel_h,
        vit_size _kernel_w, vit_size _stride_h=1, vit_size _stride_w=1, vit_bool _use_bias=true
    );
    Conv2d(const Conv2d& c2d) = delete;
    Conv2d(Conv2d&& c2d);
    ~Conv2d();

    Conv2d& operator= (const Conv2d& c2d) = delete;
    Conv2d& operator= (Conv2d&& c2d);

    vit_size get_in_channels() const;
    vit_size get_out_channels() const;
    vit_size get_kernel_h() const;
    vit_size get_kernel_w() const;
    vit_size get_stride_h() const;
    vit_size get_stride_w() const;
    vit_bool get_use_bias() const;

    void move_kernel(PictureBatch& _kernel);
    void move_bias(RowVector& _bias);

    void to_ofstream(std::ofstream& os) const;
    void from_ifstream(std::ifstream& is);

    void forward(const PictureBatch& x_in, PictureBatch& x_out) const;
};



#endif // __CONV2D_H__
