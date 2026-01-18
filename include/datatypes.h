#ifndef __DATATYPES_H__
#define __DATATYPES_H__

#include <fstream>


typedef unsigned vit_size;
typedef float vit_float;
typedef bool vit_bool;
typedef enum { pool_token, pool_avg, pool_avgmax, pool_max } pool_type;

// vit_float * g, vit_float * bias, vit_float eps, vit_bool use_bias);

struct layer_data
{
    vit_float * g;
    vit_float * bias;
    vit_float eps;
    vit_bool use_bias;

    layer_data(vit_float * i_g,vit_float * i_bias,vit_float i_eps,vit_bool i_use_bias);
    layer_data();
};

struct layer_shape
{
    vit_size g_size;
    vit_size bias_size;

    layer_shape(vit_size i_g, vit_size i_b);
};

struct scale_data //doesn't need shape
{
    vit_size dim;
    vit_float val;

    scale_data(vit_size i_dim, vit_float i_val);
};

struct linear_data
{
    vit_float * A;
    vit_float * b;
    vit_size in_features;
    vit_size out_features;
    vit_bool use_bias;

    linear_data(vit_float * i_a, vit_float * i_b,vit_size in_f, vit_size out_f, vit_bool i_use_bias);
};

struct linear_shape
{
    vit_size a_row;
    vit_size a_col;
    vit_size b_size;

    linear_shape(vit_size a_r, vit_size a_c, vit_size b_s);
};



struct attn_data {
    linear_data q_gen;
    linear_data k_gen;
    linear_data v_gen;
    layer_data q_norm; // subject to use_qk_norm
    layer_data k_norm; // subject to use_qk_norm
    linear_data proj;

    vit_size dim;
    vit_size num_heads;
    vit_size head_dim;
    vit_float scale;
    vit_bool use_qk_norm;

    attn_data(
        linear_data q_g,
        linear_data k_g,
        linear_data v_g,
        layer_data q_n, 
        layer_data k_n, 
        linear_data i_proj,
        vit_size i_dim,
        vit_size num_h,
        vit_size head_d,
        vit_float scal,
        vit_bool use_qk_n
    );
};

struct attn_shape
{
    linear_shape q_gen_shape;
    linear_shape k_gen_shape;
    linear_shape v_gen_shape;
    layer_shape q_norm_shape;
    layer_shape k_norm_shape;
    linear_shape proj_shape;

    attn_shape(
        linear_shape q_gen_s,
        linear_shape k_gen_s,
        linear_shape v_gen_s,
        layer_shape q_norm_s,
        layer_shape k_norm_s,
        linear_shape proj_s
    );
};

struct mlp_data
{
    vit_size in_features;
    vit_size hidden_features;
    vit_size out_features;
    vit_bool use_norm;

    linear_data fc1;
    layer_data norm;
    vit_float (*activaction) (vit_float val);
    linear_data fc2;

    mlp_data(
        vit_size in_f,
        vit_size hidden_f,
        vit_size out_f,
        vit_bool use_n,
        linear_data i_fc1,
        layer_data i_norm,
        vit_float (*act) (vit_float i_val),
        linear_data i_fc2
    );
};

struct mlp_shape
{
    linear_shape fc1_shape;
    layer_shape norm_shape;
    linear_shape fc2_shape;

    mlp_shape(
        linear_shape fc1_s,
        layer_shape norm_s,
        linear_shape fc2_s
    );
};

struct blocks_data
{
    vit_size dim;
    vit_size num_heads;
    vit_float mlp_ratio;

    layer_data norm1;
    attn_data attention;
    scale_data ls1;
    layer_data norm2;
    mlp_data mlp;
    scale_data ls2;

    // blocks_data();
    blocks_data(
        vit_size i_dim,
        vit_size num_h,
        vit_float mlp_r,
        layer_data i_norm1,
        attn_data attn,
        scale_data i_ls1,
        layer_data i_norm2,
        mlp_data i_mlp,
        scale_data i_ls2
    );
};

struct blocks_shape
{
    layer_shape norm1_shape;
    attn_shape attention_shape;
    mlp_shape mlperc_shape;
    layer_shape norm2_shape;

    // blocks_shape();
    blocks_shape(
        layer_shape norm1_s,
        attn_shape attention_s,
        mlp_shape mlp_s,
        layer_shape norm2_s
    );
};


class RowVector {
private:
    vit_size DIM;
    vit_float* data;
public:
    RowVector();
    RowVector(vit_size _DIM);
    RowVector(vit_float* _data, vit_size data_dim);
    RowVector(const RowVector& v) = delete;
    RowVector(RowVector&& v);
    ~RowVector();

    RowVector& operator= (const RowVector& v) = delete;
    RowVector& operator= (RowVector&& v);
    RowVector operator+ (const RowVector& v) const;
    RowVector& operator+= (const RowVector& v);

    vit_size get_DIM() const;
    vit_float at(vit_size i) const;
    vit_float * get_data() const;

    void set(vit_size i, vit_float val);

    void print() const;

    void to_ofstream(std::ofstream& os) const;
    void from_ifstream(std::ifstream& is);
};



class Matrix {
private:
    vit_size ROWS, COLS;
    vit_float* data;
public:
    Matrix();
    Matrix(vit_size _ROWS, vit_size _COLS);
    Matrix(vit_float* _data, vit_size data_dim, vit_size _ROWS, vit_size _COLS);
    Matrix(vit_float** _data, vit_size _ROWS, vit_size _COLS);
    Matrix(const Matrix& m) = delete;
    Matrix(Matrix&& m);
    ~Matrix();

    Matrix& operator= (const Matrix& m) = delete;
    Matrix& operator= (Matrix&& m);
    Matrix operator+ (const Matrix& m) const;
    Matrix& operator+= (const Matrix& m);

    vit_float * get_data();
    vit_size get_ROWS() const;
    vit_size get_COLS() const;
    vit_float at(vit_size i, vit_size j) const;

    void set(vit_size i, vit_size j, vit_float val);

    void print() const;

    void to_ofstream(std::ofstream& os) const;
    void from_ifstream(std::ifstream& is);
};



class Tensor {
private:
    vit_size B, N, C; // We will deal with three-dimensional tensors
    vit_float* data;
public:
    Tensor();
    Tensor(vit_size _B, vit_size _N, vit_size _C);
    Tensor(vit_float* _data, vit_size data_dim, vit_size _B, vit_size _N, vit_size _C);
    Tensor(vit_float*** _data, vit_size _B, vit_size _N, vit_size _C);
    Tensor(const Tensor& t) = delete;
    Tensor(Tensor&& t);
    ~Tensor();

    Tensor& operator= (const Tensor& t) = delete;
    Tensor& operator= (Tensor&& t);
    Tensor operator+ (const Tensor& t) const;
    Tensor& operator+= (const Tensor& t);

    vit_float * get_data();
    vit_size get_B() const;
    vit_size get_N() const;
    vit_size get_C() const;
    vit_float at(vit_size b, vit_size n, vit_size c) const;

    void set(vit_size b, vit_size n, vit_size c, vit_float val);
    void copy_tensor(const Tensor& t);

    void print() const;

    void to_ofstream(std::ofstream& os) const;
    void from_ifstream(std::ifstream& is);
};



class PictureBatch {
private:
    vit_size B, C, H, W;
    vit_float* data;
public:
    PictureBatch();
    PictureBatch(vit_size _B, vit_size _C, vit_size _H, vit_size _W);
    PictureBatch(
        vit_float* _data, vit_size data_dim, vit_size _B, vit_size _C, vit_size _H, vit_size _W
    );
    PictureBatch(const PictureBatch& pic) = delete;
    PictureBatch(PictureBatch&& pic);
    ~PictureBatch();

    PictureBatch& operator= (const PictureBatch& pic) = delete;
    PictureBatch& operator= (PictureBatch&& pic);

    vit_size get_B() const;
    vit_size get_C() const;
    vit_size get_H() const;
    vit_size get_W() const ;
    vit_float * get_data() const;
    vit_float at(vit_size b, vit_size c, vit_size h, vit_size w) const;

    void flatten_to_tensor(Tensor& t) const;
    void get_pad(PictureBatch& pic, vit_size new_h, vit_size new_w) const;

    void set(vit_size b, vit_size c, vit_size h, vit_size w, vit_float val);

    void print() const;

    void to_ofstream(std::ofstream& os) const;
    void from_ifstream(std::ifstream& is);
};


class PredictionBatch {
private:
    vit_size B;
    vit_size CLS;

    vit_size* classes;
    vit_float* prob;
    vit_float* prob_matrix;
public:
    PredictionBatch();
    PredictionBatch(const Tensor& t);
    PredictionBatch(const PredictionBatch& pred) = delete;
    PredictionBatch(PredictionBatch&& pred);
    ~PredictionBatch();

    PredictionBatch& operator= (const PredictionBatch& pred) = delete;
    PredictionBatch& operator= (PredictionBatch&& pred);

    vit_size get_B() const;
    vit_size get_CLS() const;
    vit_size get_prediction_class(vit_size i) const;
    vit_float get_prediction_class_probability(vit_size i) const;
    vit_float get_probability_of_class(vit_size i, vit_size cls) const;

    void get_prediction_probability_tensor(Tensor &out) const;
    void get_predictions(int *out) const;

    void print() const;

    void to_ofstream(std::ofstream& os) const;
    void from_ifstream(std::ifstream& is);
};



#endif // __DATATYPES_H__
