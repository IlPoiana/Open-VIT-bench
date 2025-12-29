
#include "../gpu_include/bench_utils.h"

int get_arg(int argc, char** argv, const char* name, int default_val){
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], name) == 0) {
            return atoi(argv[i + 1]);
        }
    }
    return default_val;
}

bool has_flag(int argc, char** argv, const char* flag){
    for (int i = 1; i < argc; ++i)
        if (strcmp(argv[i], flag) == 0)
            return true;
    return false;
}

float time_kernel(
    int warmup,
    int iters,
    cudaStream_t stream,
    std::function<void()> launch
){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up
    for (int i = 0; i < warmup; ++i)
        launch();
    cudaStreamSynchronize(stream);

    float total_ms = 0.0f;
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(start, stream);
        launch();
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return total_ms / iters;
}


// Returns the MRE of the cpu `y` Tensor and `gpu_y`. Attention! There is a tolerance instroduced to avoid division by zero
float compare_results(Tensor &y, half * gpu_y){
    float tolerance = 1e-3f;
    double avg = 0;
    float gpu_val;
    float total_elem_num = y.get_B() * y.get_N() * y.get_C();
    for(u_int b = 0; b < y.get_B(); b++){
        for(u_int t = 0; t < y.get_N(); t++){
            for(u_int c = 0; c < y.get_C(); c++){
                assert(!isnanf( y.at(b,t,c)));
                assert(!isnanf( __half2float(gpu_y[c + y.get_C() * t + y.get_C() * y.get_N() * b])));
                gpu_val = __half2float(gpu_y[c + y.get_C() * t + y.get_C() * y.get_N() * b]);
                avg += 
                    (
                        (double)abs(y.at(b,t,c) - gpu_val)
                        /
                        (double)max(abs(y.at(b,t,c)), tolerance)
                    )
                    / total_elem_num;
            }
        }
    }
    return float(avg);
}