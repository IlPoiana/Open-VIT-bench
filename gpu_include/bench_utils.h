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

int get_arg(int argc, char** argv, const char* name, int default_val);

bool has_flag(int argc, char** argv, const char* flag);


float time_kernel(
    int warmup,
    int iters,
    cudaStream_t stream,
    std::function<void()> launch
);

float compare_results(Tensor &cpu, half * gpu);
