#include "../include/datatypes.h"

#include <utility>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <math.h>

layer_data::layer_data(vit_float * i_g,vit_float * i_bias,vit_float i_eps,vit_bool i_use_bias){
    g = i_g;
    bias = i_bias;
    eps = i_eps;
    use_bias = i_use_bias;
}

layer_data::layer_data(){
    g = nullptr;
    bias = nullptr;
    eps = 0.0f;
    use_bias = false;
};

layer_shape::layer_shape(vit_size i_g, vit_size i_b): g_size(i_g), bias_size(i_b){}

scale_data::scale_data(vit_size i_dim, vit_float i_val): dim(i_dim), val(i_val){}

linear_data::linear_data(vit_float *i_a, vit_float * i_b, vit_size in_f, vit_size out_f, vit_bool i_use_bias):
in_features(in_f), out_features(out_f), use_bias(i_use_bias) {
    A = i_a;
    b = i_b;
}

linear_shape::linear_shape(vit_size a_r, vit_size a_c, vit_size b_s): a_row(a_r), a_col(a_c), b_size(b_s){}

attn_data::attn_data(
    linear_data q_g, linear_data k_g, linear_data v_g,
    layer_data q_n, layer_data k_n,
    linear_data i_proj,
    vit_size i_dim, 
    vit_size num_h, vit_size head_d,
    vit_float scal,
    vit_bool use_qk_n
):
    dim(i_dim),
    num_heads(num_h),
    head_dim(head_d),
    scale(scal),
    use_qk_norm( use_qk_n),
    q_gen(q_g.A,  q_g.b,  q_g.in_features,  q_g.out_features,  q_g.use_bias),
    k_gen(k_g.A,  k_g.b,  k_g.in_features,  k_g.out_features,  k_g.use_bias),
    v_gen(v_g.A,  v_g.b,  v_g.in_features,  v_g.out_features,  v_g.use_bias),
    proj(i_proj.A,  i_proj.b,  i_proj.in_features,  i_proj.out_features,  i_proj.use_bias),
    q_norm(q_n.g, q_n.bias, q_n.eps, q_n.use_bias),
    k_norm(k_n.g, k_n.bias, k_n.eps, k_n.use_bias)
{}

attn_shape::attn_shape(linear_shape q_gen_s, linear_shape k_gen_s, linear_shape v_gen_s, layer_shape q_norm_s, layer_shape k_norm_s, linear_shape proj_s):
    q_gen_shape(q_gen_s.a_col,q_gen_s.a_row,q_gen_s.b_size),
    k_gen_shape(k_gen_s.a_col,k_gen_s.a_row,k_gen_s.b_size),
    v_gen_shape(v_gen_s.a_col,v_gen_s.a_row,v_gen_s.b_size),
    proj_shape(proj_s.a_col,proj_s.a_row, proj_s.b_size),
    q_norm_shape(q_norm_s.g_size, q_norm_s.bias_size),
    k_norm_shape(k_norm_s.g_size, k_norm_s.bias_size)
{}

//TO CHECK the activation
mlp_data::mlp_data(vit_size in_f, vit_size hidden_f, vit_size out_f, vit_bool use_n, linear_data i_fc1, layer_data i_norm, vit_float (* act)(vit_float i_val), linear_data i_fc2):
    in_features(in_f),
    hidden_features(hidden_f),
    out_features(out_f),
    use_norm(use_n),
    fc1(i_fc1.A, i_fc1.b, i_fc1.in_features, i_fc1.out_features, i_fc1.use_bias),
    fc2(i_fc2.A, i_fc2.b, i_fc2.in_features, i_fc2.out_features, i_fc2.use_bias),
    norm(i_norm.g, i_norm.bias, i_norm.eps, i_norm.use_bias)
{
    activaction = act;
}

mlp_shape::mlp_shape(linear_shape fc1_s, layer_shape norm_s, linear_shape fc2_s):
    fc1_shape(fc1_s.a_row, fc1_s.a_col, fc1_s.b_size),
    fc2_shape(fc2_s.a_row, fc2_s.a_col, fc2_s.b_size),
    norm_shape(norm_s.g_size, norm_s.bias_size)
{}

// blocks_data::blocks_data()
// {
//     dim = 0;
//     num_heads = 0;
//     mlp_ratio = 0.0f;
// }


blocks_data::blocks_data(vit_size i_dim, vit_size num_h, vit_float mlp_r, layer_data i_norm1, attn_data attn, scale_data i_ls1, layer_data i_norm2, mlp_data i_mlp, scale_data i_ls2) :
    dim(i_dim),
    num_heads(num_h),
    mlp_ratio(mlp_r),
    norm1(i_norm1.g,i_norm1.bias, i_norm1.eps, i_norm1.use_bias),
    norm2(i_norm2.g,i_norm2.bias, i_norm2.eps, i_norm2.use_bias),
    ls1(i_ls1.dim, i_ls1.val),
    ls2(i_ls2.dim, i_ls2.val),
    attention( attn.q_gen,  attn.k_gen,  attn.v_gen,  attn.q_norm, attn.k_norm,  attn.proj,  attn.dim,  attn.num_heads,  attn.head_dim, attn.scale, attn.use_qk_norm),
    mlp(i_mlp.in_features,i_mlp.hidden_features, i_mlp.out_features, i_mlp.use_norm,
    i_mlp.fc1,i_mlp.norm,i_mlp.activaction,i_mlp.fc2)
{}

// blocks_shape::blocks_shape()
// {
// }

blocks_shape::blocks_shape(layer_shape norm1_s, attn_shape attention_s, mlp_shape mlp_s, layer_shape norm2_s) : 
    norm1_shape(norm1_s.g_size, norm1_s.bias_size),
    norm2_shape(norm2_s.g_size, norm2_s.bias_size),
    attention_shape(attention_s.q_gen_shape, attention_s.k_gen_shape, attention_s.v_gen_shape, attention_s.q_norm_shape, attention_s.k_norm_shape, attention_s.proj_shape),
    mlperc_shape(mlp_s.fc1_shape, mlp_s.norm_shape, mlp_s.fc2_shape)
{}


// RowVector class

RowVector::RowVector() {
    DIM = 0;
    data = nullptr;
}

RowVector::RowVector(vit_size _DIM) {
    DIM = _DIM;
    data = new vit_float[_DIM];
}

RowVector::RowVector(vit_float* _data, vit_size data_dim) {
    DIM = data_dim;
    data = new vit_float[data_dim];
    for (int i=0;i<data_dim;++i) {
        data[i] = _data[i];
    }
}

RowVector::RowVector(RowVector&& v) {
    DIM = v.DIM;
    data = v.data;

    v.DIM = 0;
    v.data = nullptr;
}

RowVector::~RowVector() {
    if (data != nullptr) {
        delete [] data;
    }
}

RowVector& RowVector::operator= (RowVector&& v) {
    DIM = v.DIM;
    if (data != nullptr) {
        delete [] data;
    }
    data = v.data;

    v.DIM = 0;
    v.data = nullptr;

    return *this;
}

RowVector RowVector::operator+ (const RowVector& v) const {
    assert(this->DIM == v.DIM);
    RowVector res(this->DIM);

    for (int i=0;i<this->DIM;++i) {
        res.data[i] = this->data[i] + v.data[i];
    }

    return res;
}

RowVector& RowVector::operator+= (const RowVector& v) {
    assert(this->DIM == v.DIM);

    for (int i=0;i<this->DIM;++i) {
        this->data[i] += v.data[i];
    }

    return *this;
}

vit_size RowVector::get_DIM() const {
    return DIM;
}

vit_float RowVector::at(vit_size i) const {
    assert(i<DIM);
    return data[i];
}

vit_float * RowVector::get_data() const {
    return data;
}

void RowVector::set(vit_size i, vit_float val) {
    assert(i<DIM);
    data[i] = val;
}

void RowVector::print() const {
    std::cout << "RowVector[" << DIM << "]:" << std::endl;
    std::cout << "   ";
    for (int i=0;i<DIM;++i) {
        //std::cout << this->at(i) << " ";
        printf("%.3f ", this->at(i));
    }
    std::cout << std::endl << std::endl;
}

void RowVector::to_ofstream(std::ofstream& os) const {
    assert( os.is_open() );
    vit_size ndim = 1;
    os.write( (char*) &ndim, sizeof(vit_size));

    os.write( (char*) &DIM, sizeof(vit_size));

    os.write( (char*) data, sizeof(vit_float)*DIM );
}

void RowVector::from_ifstream(std::ifstream& is) {
    assert( is.is_open() );
    vit_size ndim;
    is.read( (char*) &ndim, sizeof(vit_size));
    assert(ndim == 1);

    vit_size old_dim = DIM;
    is.read( (char*) &DIM, sizeof(vit_size));

    if (DIM != old_dim) {
        if (data != nullptr) {
            delete [] data;
        }
        data = new vit_float[DIM];
    }
    is.read( (char*) data, sizeof(vit_float)*DIM );
}



// Matrix class

Matrix::Matrix() {
    ROWS = COLS = 0;
    data = nullptr;
}

Matrix::Matrix(vit_size _ROWS, vit_size _COLS) {
    ROWS = _ROWS;
    COLS = _COLS;
    data = new vit_float[_ROWS*_COLS];
}

Matrix::Matrix(vit_float* _data, vit_size data_dim, vit_size _ROWS, vit_size _COLS) {
    assert(data_dim == _ROWS*_COLS);
    ROWS = _ROWS;
    COLS = _COLS;
    data = new vit_float[data_dim];
    for (int i=0;i<data_dim;++i) {
        data[i] = _data[i];
    }
}

Matrix::Matrix(vit_float** _data, vit_size _ROWS, vit_size _COLS) {
    ROWS = _ROWS;
    COLS = _COLS;
    data = new vit_float[_ROWS*_COLS];
    for (int i=0;i<_ROWS;++i) {
        for (int j=0;j<_COLS;++j) {
            data[j + (i*_COLS)] = _data[i][j];
        }
    }
}

Matrix::Matrix(Matrix&& m) {
    ROWS = m.ROWS;
    COLS = m.COLS;
    data = m.data;

    m.ROWS = 0;
    m.COLS = 0;
    m.data = nullptr;
}

Matrix::~Matrix() {
    if (data != nullptr) {
        delete [] data;
    }
}

Matrix& Matrix::operator= (Matrix&& m) {
    ROWS = m.ROWS;
    COLS = m.COLS;
    if (data != nullptr) {
        delete [] data;
    }
    data = m.data;

    m.ROWS = 0;
    m.COLS = 0;
    m.data = nullptr;

    return *this;
}

Matrix Matrix::operator+ (const Matrix& m) const {
    assert(this->ROWS == m.ROWS);
    assert(this->COLS == m.COLS);
    Matrix res(this->ROWS, this->COLS);

    for (int i=0;i<this->ROWS * this->COLS;++i) {
        res.data[i] = this->data[i] + m.data[i];
    }

    return res;
}

Matrix& Matrix::operator+= (const Matrix& m) {
    assert(this->ROWS == m.ROWS);
    assert(this->COLS == m.COLS);

    for (int i=0;i<this->ROWS * this->COLS;++i) {
        this->data[i] += m.data[i];
    }

    return *this;
}

vit_size Matrix::get_ROWS() const {
    return ROWS;
}

vit_size Matrix::get_COLS() const {
    return COLS;
}

vit_float *Matrix::get_data()
{
    return data;
}

vit_float Matrix::at(vit_size i, vit_size j) const {
    assert(i<ROWS);
    assert(j<COLS);
    return data[j + (i*COLS)];
}

void Matrix::set(vit_size i, vit_size j, vit_float val) {
    assert(i<ROWS);
    assert(j<COLS);
    data[j + (i*COLS)] = val;
}

void Matrix::print() const {
    std::cout << "Matrix[" << ROWS << "x" << COLS << "]:" << std::endl;
    for (int i=0;i<ROWS;++i) {
        std::cout << "   ";
        for (int j=0;j<COLS;++j) {
            //std::cout << this->at(i,j) << " ";
            printf("%7.3f ", this->at(i,j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void Matrix::to_ofstream(std::ofstream& os) const {
    assert( os.is_open() );
    vit_size dim = 2;
    os.write( (char*) &dim, sizeof(vit_size));

    os.write( (char*) &ROWS, sizeof(vit_size));
    os.write( (char*) &COLS, sizeof(vit_size));

    os.write( (char*) data, sizeof(vit_float)*ROWS*COLS );
}

void Matrix::from_ifstream(std::ifstream& is) {
    assert( is.is_open() );
    vit_size dim;
    is.read( (char*) &dim, sizeof(vit_size));
    assert(dim == 2);

    vit_size old_dim = ROWS*COLS;
    is.read( (char*) &ROWS, sizeof(vit_size));
    is.read( (char*) &COLS, sizeof(vit_size));

    if (ROWS*COLS != old_dim) {
        if (data != nullptr) {
            delete [] data;
        }
        data = new vit_float[ROWS*COLS];
    }
    is.read( (char*) data, sizeof(vit_float)*ROWS*COLS );
}



// Tensor class

Tensor::Tensor() {
    B = N = C = 0;
    data = nullptr;
}

Tensor::Tensor(vit_size _B, vit_size _N, vit_size _C) {
    B = _B;
    N = _N;
    C = _C;
    data = new vit_float[_B*_N*_C];
}

Tensor::Tensor(vit_float* _data, vit_size data_dim, vit_size _B, vit_size _N, vit_size _C) {
    assert(data_dim == _B*_N*_C);
    B = _B;
    N = _N;
    C = _C;
    data = new vit_float[data_dim];
    for (int i=0;i<data_dim;++i) {
        data[i] = _data[i];
    }
}

Tensor::Tensor(vit_float*** _data, vit_size _B, vit_size _N, vit_size _C) {
    B = _B;
    N = _N;
    C = _C;
    data = new vit_float[_B*_N*_C];
    for (int b=0;b<_B;++b) {
        for (int n=0;n<_N;++n) {
            for (int c=0;c<_C;++c) {
                data[c + (n*_C) + (b*_N*_C)] = _data[b][n][c];
            }
        }
    }
}

Tensor::Tensor(Tensor&& t) {
    B = t.B;
    N = t.N;
    C = t.C;
    data = t.data;

    t.B = 0;
    t.N = 0;
    t.C = 0;
    t.data = nullptr;
}

Tensor::~Tensor() {
    if (data != nullptr) {
        delete [] data;
    }
}

Tensor& Tensor::operator= (Tensor&& t) {
    B = t.B;
    N = t.N;
    C = t.C;
    if (data != nullptr) {
        delete [] data;
    }
    data = t.data;

    t.B = 0;
    t.N = 0;
    t.C = 0;
    t.data = nullptr;

    return *this;
}

Tensor Tensor::operator+ (const Tensor& t) const {
    assert(this->B == t.B);
    assert(this->N == t.N);
    assert(this->C == t.C);
    Tensor res(this->B, this->N, this->C);

    for (int i=0;i<this->B * this->N * this->C;++i) {
        res.data[i] = this->data[i] + t.data[i];
    }

    return res;
}

Tensor& Tensor::operator+= (const Tensor& t) {
    assert(this->B == t.B);
    assert(this->N == t.N);
    assert(this->C == t.C);

    for (int i=0;i<this->B * this->N * this->C;++i) {
        this->data[i] += t.data[i];
    }

    return *this;
}

vit_float * Tensor::get_data(){
    return data;
}

vit_size Tensor::get_B() const {
    return B;
}

vit_size Tensor::get_N() const {
    return N;
}

vit_size Tensor::get_C() const {
    return C;
}

vit_float Tensor::at(vit_size b, vit_size n, vit_size c) const {
    assert(b<B);
    assert(n<N);
    assert(c<C);
    return data[c + (n*C) + (b*N*C)];
}

void Tensor::set(vit_size b, vit_size n, vit_size c, vit_float val) {
    assert(b<B);
    assert(n<N);
    assert(c<C);
    data[c + (n*C) + (b*N*C)] = val;
}

void Tensor::copy_tensor(const Tensor& t) {
    vit_size dim = t.B * t.N * t.C;
    if (this->B * this->N * this->C != dim) {
        if (this->data != nullptr) {
            delete [] this->data;
        }
        this->B = t.B;
        this->N = t.N;
        this->C = t.C;
        this->data = new vit_float[dim];
    }

    for (int i=0;i<dim;++i) {
        this->data[i] = t.data[i];
    }
}

void Tensor::print() const {
    std::cout << "Tensor[" << B << "x" << N << "x" << C << "]:" << std::endl;
    for(int b=0;b<B;++b) {
        std::cout << "   B[" << b << "]" << std::endl;
        for (int n=0;n<N;++n) {
            std::cout << "   ";
            for (int c=0;c<C;++c) {
                // std::cout << this->at(b,n,c) << " ";
                printf("%7.3f ", this->at(b,n,c));
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}

void Tensor::to_ofstream(std::ofstream& os) const {
    assert( os.is_open() );
    vit_size dim = 3;
    os.write( (char*) &dim, sizeof(vit_size));

    os.write( (char*) &B, sizeof(vit_size));
    os.write( (char*) &N, sizeof(vit_size));
    os.write( (char*) &C, sizeof(vit_size));

    os.write( (char*) data, sizeof(vit_float)*B*N*C );
}

void Tensor::from_ifstream(std::ifstream& is) {
    assert( is.is_open() );
    vit_size dim;
    is.read( (char*) &dim, sizeof(vit_size));
    assert(dim == 3);

    vit_size old_dim = B*N*C;
    is.read( (char*) &B, sizeof(vit_size));
    is.read( (char*) &N, sizeof(vit_size));
    is.read( (char*) &C, sizeof(vit_size));

    if (B*N*C != old_dim) {
        if (data != nullptr) {
            delete [] data;
        }
        data = new vit_float[B*N*C];
    }
    is.read( (char*) data, sizeof(vit_float)*B*N*C );
}



// PictureBatch class

PictureBatch::PictureBatch() {
    B = C = H = W = 0;
    data = nullptr;
}

PictureBatch::PictureBatch(vit_size _B, vit_size _C, vit_size _H, vit_size _W) {
    B = _B;
    C = _C;
    H = _H;
    W = _W;
    data = new vit_float[_B*_C*_H*_W];
}

PictureBatch::PictureBatch(vit_float* _data, vit_size data_dim, vit_size _B, vit_size _C, vit_size _H, vit_size _W) {
    assert(data_dim == _B*_C*_H*_W);
    B = _B;
    C = _C;
    H = _H;
    W = _W;
    data = new vit_float[data_dim];
    for (int i=0;i<data_dim;++i) {
        data[i] = _data[i];
    }
}

PictureBatch::PictureBatch(PictureBatch&& pic) {
    B = pic.B;
    C = pic.C;
    H = pic.H;
    W = pic.W;
    data = pic.data;

    pic.B = 0;
    pic.C = 0;
    pic.H = 0;
    pic.W = 0;
    pic.data = nullptr;
}

PictureBatch::~PictureBatch() {
    if (data != nullptr) {
        delete [] data;
    }
}

PictureBatch& PictureBatch::operator= (PictureBatch&& pic) {
    B = pic.B;
    C = pic.C;
    H = pic.H;
    W = pic.W;
    if (data != nullptr) {
        delete [] data;
    }
    data = pic.data;

    pic.B = 0;
    pic.C = 0;
    pic.H = 0;
    pic.W = 0;
    pic.data = nullptr;

    return *this;
}

vit_size PictureBatch::get_B() const {
    return B;
}

vit_size PictureBatch::get_C() const {
    return C;
}

vit_size PictureBatch::get_H() const {
    return H;
}

vit_size PictureBatch::get_W() const {
    return W;
}

vit_float * PictureBatch::get_data() const {
    return data;
}

vit_float PictureBatch::at(vit_size b, vit_size c, vit_size h, vit_size w) const {
    assert(b<B);
    assert(c<C);
    assert(h<H);
    assert(w<W);
    return data[w + (h*W) + (c*H*W) + (b*C*H*W)];
}

void PictureBatch::flatten_to_tensor(Tensor& t) const {
    Tensor x(this->B, this->H * this->W, this->C);

    vit_float val;
    for (int i=0;i<this->B;++i) {
        for (int j=0;j<this->C;++j) {
            for (int k=0;k<this->H;++k) {
                for (int l=0;l<this->W;++l) {
                    val = this->at(i,j,k,l);
                    x.set(i, (k*this->W) + l, j, val);
                }
            }
        }
    }

    t = std::move(x);
}

void PictureBatch::get_pad(PictureBatch& pic, vit_size new_h, vit_size new_w) const {
    assert(new_h >= this->H);
    assert(new_w >= this->W);

    PictureBatch p(this->B, this->C, new_h, new_w);

    vit_float val;
    for (int i=0;i<this->B;++i) {
        for (int j=0;j<this->C;++j) {
            for (int k=0;k<new_h;++k) {
                for (int l=0;l<new_w;++l) {
                    if (k<this->H && l<this->W) {
                        val = this->at(i,j,k,l);
                    } else {
                        val = 0.0;
                    }
                    p.set(i, j, k, l, val);
                }
            }
        }
    }

    pic = std::move(p);
}

void PictureBatch::set(vit_size b, vit_size c, vit_size h, vit_size w, vit_float val) {
    assert(b<B);
    assert(c<C);
    assert(h<H);
    assert(w<W);
    data[w + (h*W) + (c*H*W) + (b*C*H*W)] = val;
}

void PictureBatch::print() const {
    std::cout << "Picture[" << B << "x" << C << "x" << H << "x" << W << "]:" << std::endl;
    for(int i=0;i<B;++i) {
        std::cout << "   B[" << i << "]" << std::endl;
        for (int j=0;j<H;++j) {
            std::cout << "   ";
            for (int k=0;k<W;++k) {
                std::cout << "[ ";
                for (int l=0;l<C;++l) {
                    //std::cout << this->at(i,l,j,k) << " ";
                    printf("%7.3f ", this->at(i,l,j,k));
                }
                std::cout << "]";
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}

void PictureBatch::to_ofstream(std::ofstream& os) const {
    assert( os.is_open() );
    vit_size dim = 4;
    os.write( (char*) &dim, sizeof(vit_size));

    os.write( (char*) &B, sizeof(vit_size));
    os.write( (char*) &C, sizeof(vit_size));
    os.write( (char*) &H, sizeof(vit_size));
    os.write( (char*) &W, sizeof(vit_size));

    os.write( (char*) data, sizeof(vit_float)*B*C*H*W );
}

void PictureBatch::from_ifstream(std::ifstream& is) {
    assert( is.is_open() );
    vit_size dim;
    is.read( (char*) &dim, sizeof(vit_size));
    assert(dim == 4);

    vit_size old_dim = B*C*H*W;
    is.read( (char*) &B, sizeof(vit_size));
    is.read( (char*) &C, sizeof(vit_size));
    is.read( (char*) &H, sizeof(vit_size));
    is.read( (char*) &W, sizeof(vit_size));

    if (B*C*H*W != old_dim) {
        if (data != nullptr) {
            delete [] data;
        }
        data = new vit_float[B*C*H*W];
    }
    is.read( (char*) data, sizeof(vit_float)*B*C*H*W );
}



// PredictionBatch class

PredictionBatch::PredictionBatch() {
    B = CLS = 0;
    classes = nullptr;
    prob = nullptr;
    prob_matrix = nullptr;
}

PredictionBatch::PredictionBatch(const Tensor& t) {
    // Please note: on Python code PredictionBatch is constructed from 2d tensors,
    // here it's constructed from 3d tensors whose second dimension is 1
    assert(t.get_N() == 1);

    B = t.get_B();
    CLS = t.get_C();
    prob_matrix = new vit_float[B * CLS];

    classes = new vit_size[B];
    prob = new vit_float[B];

    vit_float val;
    vit_size max_val_index;
    vit_float max_val;
    vit_float cumulative;

    for(int i=0;i<B;++i) {
        max_val = std::exp( t.at(i,0,0) );
        max_val_index = 0;
        cumulative = 0.0;
        for (int cls=0;cls<CLS;++cls) {
            val = std::exp( t.at(i,0,cls) );
            cumulative += val;
            prob_matrix[cls + (i*CLS)] = val;
            if (val>max_val) {
                max_val = val;
                max_val_index = cls;
            }
        }
        for (int cls=0;cls<CLS;++cls) {
            val = prob_matrix[cls + (i*CLS)];
            val /= cumulative;
            prob_matrix[cls + (i*CLS)] = val;
        }
        classes[i] = max_val_index;
        prob[i] = max_val / cumulative;
    }
}

PredictionBatch::PredictionBatch(PredictionBatch&& pred) {
    B = pred.B;
    CLS = pred.CLS;
    classes = pred.classes;
    prob = pred.prob;
    prob_matrix = pred.prob_matrix;

    pred.B = 0;
    pred.CLS = 0;
    pred.classes = nullptr;
    pred.prob = nullptr;
    pred.prob_matrix = nullptr;
}

PredictionBatch::~PredictionBatch() {
    if (classes != nullptr) {
        delete [] classes;
    }
    if (prob != nullptr) {
        delete [] prob;
    } 
    if (prob_matrix != nullptr) {
        delete [] prob_matrix;
    } 
}

PredictionBatch& PredictionBatch::operator= (PredictionBatch&& pred) {
    B = pred.B;
    CLS = pred.CLS;
    if (classes != nullptr) {
        delete [] classes;
    }
    if (prob != nullptr) {
        delete [] prob;
    }
    if (prob_matrix != nullptr) {
        delete [] prob_matrix;
    }
    classes = pred.classes;
    prob = pred.prob;
    prob_matrix = pred.prob_matrix;

    pred.B = 0;
    pred.CLS = 0;
    pred.classes = nullptr;
    pred.prob = nullptr;
    pred.prob_matrix = nullptr;

    return *this;
}

vit_size PredictionBatch::get_B() const {
    return B;
}

vit_size PredictionBatch::get_CLS() const {
    return CLS;
}

vit_size PredictionBatch::get_prediction_class(vit_size i) const {
    assert(i<B);
    return classes[i];
}

vit_float PredictionBatch::get_prediction_class_probability(vit_size i) const {
    assert(i<B);
    return prob[i];
}

vit_float PredictionBatch::get_probability_of_class(vit_size i, vit_size cls) const {
    assert(i<B);
    assert(cls<CLS);
    return prob_matrix[cls + (i*CLS)];
}

void PredictionBatch::get_prediction_probability_tensor(Tensor &out) const {
    Tensor t(B, 1, CLS);

    for (int i=0;i<B;++i) {
        for (int cls=0;cls<CLS;++cls) {
            t.set(i, 0, cls, get_probability_of_class(i, cls));
        }
    }

    out = std::move(t);
}

void PredictionBatch::get_predictions(int *out) const {
    for (int i=0;i<B;++i) {
        out[i] = classes[i];
    }
}


void PredictionBatch::print() const {
    std::cout << "Prediction[" << B << "], " << CLS << " classes:" << std::endl;
    for(int i=0;i<B;++i) {
        std::cout << "   B[" << i << "]: class " << classes[i] << ", prob ";
        printf("%7.3f\n", prob[i]);
    }
    std::cout << std::endl;

    std::cout << "   Probability Matrix[" << B << "x" << CLS << "]:" << std::endl;
    for (int i=0;i<B;++i) {
        std::cout << "   ";
        for (int cls=0;cls<CLS;++cls) {
            //std::cout << this->at(i,j) << " ";
            //printf("%7.3f ", this->get_probability_of_class(i,cls));
            printf("%13.9f ", this->get_probability_of_class(i,cls));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void PredictionBatch::to_ofstream(std::ofstream& os) const {
    assert( os.is_open() );

    os.write( (char*) &B, sizeof(vit_size));
    os.write( (char*) &CLS, sizeof(vit_size));

    os.write( (char*) classes, sizeof(vit_size)*B );
    os.write( (char*) prob, sizeof(vit_float)*B );
    os.write( (char*) prob_matrix, sizeof(vit_float)*B*CLS );
}

void PredictionBatch::from_ifstream(std::ifstream& is) {
    assert( is.is_open() );
    vit_size old_B;
    vit_size old_dim;

    old_B = B;
    old_dim = B*CLS;
    is.read( (char*) &B, sizeof(vit_size));
    is.read( (char*) &CLS, sizeof(vit_size));

    if (B != old_B) {
        if (classes != nullptr) {
            delete [] classes;
        }
        classes = new vit_size[B];
        if (prob != nullptr) {
            delete [] prob;
        }
        prob = new vit_float[B];
    }
    is.read( (char*) classes, sizeof(vit_size)*B );
    is.read( (char*) prob, sizeof(vit_float)*B );

    if (B*CLS != old_dim) {
        if (prob_matrix != nullptr) {
            delete [] prob_matrix;
        }
        prob_matrix = new vit_float[B*CLS];
    }
    is.read( (char*) prob_matrix, sizeof(vit_float)*B*CLS );
}

