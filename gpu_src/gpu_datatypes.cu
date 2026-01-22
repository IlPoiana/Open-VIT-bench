#include "../gpu_include/gpu_datatypes.h"

#include <assert.h>
#include <sys/time.h>

mtx::mtx(float * f32_data, u_int16_t row, u_int16_t col): row_n(row), col_n(col)
{
    data = (half*)malloc(sizeof(half) * row * col);
    f32_to_f16(f32_data, data, row * col);
}

mtx::mtx(u_int row, u_int col){
    data = (half*)malloc(sizeof(half) * row * col);
}

mtx::~mtx(){
    free(data);
}

h_tensor::~h_tensor(){
    free(data);
}


h_tensor::h_tensor(float * f32_data, u_int16_t batch, u_int16_t channels, u_int16_t height, u_int16_t width):
    B(batch),
    C(channels),
    H(height),
    W(width)
{
    data = (half*)malloc(sizeof(half) * batch * channels * height * width);
    f32_to_f16(f32_data, data, batch * channels * height * width);
}
