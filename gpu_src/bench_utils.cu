#include "../gpu_include/bench_utils.h"

void benchmark_time::print() {
    cout << "   Kernel time  : " << avg_time << " +"<< variance<< " ms\n";
}

void benchmark_time::to_JSON(int batch, int params[]){
    int elements_per_th = params[0];
    int tokens_per_block = params[1];

    cout << "{\n"
        << "\"batch\":" << batch << ",\n"
        << "\"params\": {\n" 
            << "\"tokens_per_block\":" << tokens_per_block << ",\n"
            << "\"elements_per_th\":" << elements_per_th << "\n"
        << "},\n"
        << "\"time\": {\n" 
            << "\"time\":"      << avg_time << ",\n"
            << "\"variance\":"  << variance << "\n"
        << "}\n"
        << "}\n";
}


int get_arg(int argc, char** argv, const char* name, int default_val){
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

benchmark_time time_kernel_variance(
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
    std::vector<float> times(iters); // Store individual execution times

    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(start, stream);
        launch();
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        times[i] = ms; // Save the time for variance calculation
        total_ms += ms;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    float average_time = total_ms / iters;

    // Compute variance
    float variance = 0.0f;
    for (float time : times) {
        variance += (time - average_time) * (time - average_time);
    }
    variance /= iters;

    return benchmark_time(average_time, variance);
}


benchmark_time time_cpu(
    int warmup,
    int iters,
    std::function<void()> func
){
    // Warm-up
    for (int i = 0; i < warmup; ++i)
        func();

    std::vector<float> times(iters);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iters; ++i) {
        auto iter_start = std::chrono::high_resolution_clock::now();
        func();
        auto iter_end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<float, std::milli> duration = iter_end - iter_start;
        times[i] = duration.count();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    
    // Calculate average
    float sum = 0.0f;
    for (float time : times) {
        sum += time;
    }
    float average = sum / iters;

    // Calculate variance
    float variance_sum = 0.0f;
    for (float time : times) {
        variance_sum += (time - average) * (time - average);
    }
    float variance = variance_sum / iters;

    return benchmark_time(average, variance);
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

float compare_predictions(PredictionBatch &cpu, int * gpu){
    double avg = 0;
    int total_elem_num = cpu.get_B();
    for(u_int b = 0; b < cpu.get_B(); b++){
        avg += 
            (cpu.get_prediction_class(b) == gpu[b] ? 1.0 : 0.0)
            / total_elem_num;
    }
                               
    return float(avg);
}