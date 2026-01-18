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

struct kernel_time {
    float time;

    kernel_time(float time) : time(time) {}

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

float time_cpu(
    int warmup,
    int iters,
    std::function<void()> func
);

float compare_results(Tensor &cpu, half * gpu);

float compare_predictions(PredictionBatch &cpu, int * gpu);