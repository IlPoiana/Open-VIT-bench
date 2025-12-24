#pragma once
#include <functional>

using namespace std;

int get_arg(int argc, char** argv, const char* name, int default_val);


float time_kernel(int warmup, int iters, function<void()> launch);
