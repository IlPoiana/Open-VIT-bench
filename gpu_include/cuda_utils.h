#include <vector>

// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                     \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)


struct picture_shape {
    int B;  // Batch
    int C;  // Channels
    int H;  // Height
    int W;  // Width

    picture_shape(int b, int c, int h, int w) : B(b), C(c), H(h), W(w) {}
};

struct conv_kernel_shape {
    int W;  // Batch
    int H;  // Channels
    int w_stride;  // Height
    int h_stride;  // Width

    conv_kernel_shape(int w, int h, int s_w, int s_h) : W(w), H(h), w_stride(s_w), h_stride(s_h) {}
    conv_kernel_shape();  
};

struct benchmark_time
{
    std::vector<float> preprocess;
    float kernel;

    benchmark_time(std::vector<float> pre, float &k);
};

void print_time(benchmark_time time);

__global__ void addScalarKernel(float* array, float val, int N);

void linearize(float * data, float * linearized_data, picture_shape input_img, conv_kernel_shape kernel);
