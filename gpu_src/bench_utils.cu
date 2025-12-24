
#include "../gpu_include/bench_utils.h"

int get_arg(int argc, char** argv, const char* name, int default_val)
{
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], name) == 0) {
            return atoi(argv[i + 1]);
        }
    }
    return default_val;
}


float time_kernel(
    int warmup,
    int iters,
    function<void()> launch)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up
    for (int i = 0; i < warmup; ++i)
        launch();
    cudaDeviceSynchronize();

    float total_ms = 0.0f;
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(start);
        launch();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return total_ms / iters;
}