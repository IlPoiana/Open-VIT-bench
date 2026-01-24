#include "../gpu_include/gpu_block.h"


#define B 2
#define T 7
#define C 9
#define K 10
#define M 9
#define EPSILON 1e-5

// Returns the MRE of the cpu `y` Tensor and `gpu_y`. Attention! There is a tolerance instroduced to avoid division by zero
float compare_results(Tensor &y, float * gpu_y){
    float tolerance = 1e-3;
    double avg = 0;
    float gpu_val;
    float total_elem_num = y.get_B() * y.get_N() * y.get_C();
    for(u_int b = 0; b < y.get_B(); b++){
        for(u_int t = 0; t < y.get_N(); t++){
            for(u_int c = 0; c < y.get_C(); c++){
                assert(!isnanf( y.at(b,t,c)));
                assert(!isnanf( gpu_y[c + y.get_C() * t + y.get_C() * y.get_N() * b]));
                gpu_val = gpu_y[c + y.get_C() * t + y.get_C() * y.get_N() * b];
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


// from A: MxN (row-major) to B: NxM (row-major)
template <class cls>
void transpose_out_of_place(const cls* in, cls* out, std::size_t rows, std::size_t cols) {
    for (std::size_t i = 0; i < rows; ++i) {
        const cls* Ai = in + i * cols;
        for (std::size_t j = 0; j < cols; ++j) {
            out[j * rows + i] = Ai[j];
        }
    }
}


void cpu_gpu_comparison(bool fused_mlp, bool debug){ 
    u_int batch = 4,tokens = 197,channels = 768,hidden = 3072;
    float scale = 1.0f;// 4 * 1e-5;
    int num_heads = 12; 
    
    if(debug){
        batch = 4,tokens = 4,channels = 768, hidden = 40;
        num_heads = 12;
    }   

    u_int input_elements_number = batch * tokens * channels;
    vector<float> h_x(input_elements_number);

    cout << "Tensor: [" << batch << ","<< tokens << "," << channels << "]" << endl;
    cout << "fc1: [" << channels << ","<< hidden << "]" << endl;
    cout << "fc2: [" << hidden << ","<< channels << "]" << endl;

    //Descriptors
    bool attn_init = false; //false == Initialize randomly the attn_descriptor

    //Generate all the descriptors, except attn (need weights for cuDNN)
    if(debug) cout << "constructor" << endl;
    GpuBlock gpu_block(
        batch, tokens, channels, hidden,
        fused_mlp,
        EPSILON, scale, num_heads
    );

    //Generate random data for all the weights in the encoder block, initialize also the cuDNN attn weights descriptor
    if(debug) cout << "random gen"<< endl;
    gpu_block.random_data(attn_init);
    gpu_block.download_x(h_x.data()); //for debug purposes
    if(debug) cout << "cpu x tensor" << endl;
    Tensor x(h_x.data(), input_elements_number, batch, tokens, channels);
    if(debug)  x.print();
    
    if(debug) cout << "forward" << endl;
    gpu_block.forward(debug);
    
    if(debug) cout << "download gpu result" << endl;
    gpu_block.download_x(h_x.data());

    // -- CPU REFERENCE --
    if(debug) cout << "CPU reference" << endl;
    Tensor y(batch,tokens,channels);
    Block cpu_block(channels, num_heads, hidden / channels, true, false, scale, GELU);
    
    if(debug) cout << "to CPU" << endl;
    gpu_block.to_CPU(cpu_block, debug);
    
    cpu_block.forward(x,y);
    if(debug) {
        Tensor gpu_x(h_x.data(), input_elements_number, batch, tokens, channels);
        cout << "gpu output" << endl; gpu_x.print();
        y.print();
    }
    cout << "GPU/CPU avg. difference: " << compare_results(y, h_x.data()) * 100 << " %" << endl;
    gpu_block.mark_shared_weights();
    gpu_block.mark_shared_buffers(); //Tell to the destructor to free device pointers

}

int main() {
    bool fused_mlp = false;
    bool debug = false;
    cpu_gpu_comparison(fused_mlp, debug);
    
    return 0;
}