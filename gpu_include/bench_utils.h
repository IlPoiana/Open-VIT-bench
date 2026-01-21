#pragma once
#include <functional>
#include "./gpu_datatypes.h"

#ifndef WARM_UP
#define WARM_UP 20
#endif
#ifndef N
#define N 100
#endif

using namespace std;

struct benchmark_time {
    float avg_time;
    float variance;

    benchmark_time(float avg, float var) : avg_time(avg), variance(var) {}

    void print();

    //`params`: 0 - elements per thread, 1 - tokens per block
    void to_JSON(int batch, int params[]);
};


int get_arg(int argc, char** argv, const char* name, int default_val);

bool has_flag(int argc, char** argv, const char* flag);


float time_kernel(
    int warmup,
    int iters,
    cudaStream_t stream,
    std::function<void()> launch
);

benchmark_time time_kernel_variance(
    int warmup,
    int iters,
    cudaStream_t stream,
    std::function<void()> launch
);

benchmark_time time_cpu(
    int warmup,
    int iters,
    std::function<void()> func
);

float compare_results(Tensor &cpu, half * gpu);

float compare_predictions(PredictionBatch &cpu, int * gpu);