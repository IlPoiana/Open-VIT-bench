#include "../gpu_include/gpu_patch_embedder.h"
#include "../include/vision_transformer.h"

#define STREAM_N 8

/*
----
This tests includes the patch embedder + position embedder.
Basically the entire part before the encoder blocks
----
*/

double compare_results(Tensor &y, half * gpu_y){
    double avg = 0;
    for(u_int b = 0; b < y.get_B(); b++){
        for(u_int t = 0; t < y.get_N(); t++){
            for(u_int c = 0; c < y.get_C(); c++){
                assert(!isnanf( y.at(b,t,c)));
                assert(!isnanf( __half2float(gpu_y[c + y.get_C() * t + y.get_C() * y.get_N() * b])));
                avg += (double)abs(y.at(b,t,c) - __half2float(gpu_y[c + y.get_C() * t + y.get_C() * y.get_N() * b]));
                
            }
        }
    }
    return avg / (double(y.get_B()) * y.get_N() * y.get_C());
}

Tensor cpu_baseline(
    convolution_dim &conv_dim,
    float * conv_weights,
    float * conv_bias,
    float * positional_embeddings,
    float * x_data,
    bool debug

){
    convolution_dim cd = conv_dim;
  
    u_int tokens = (cd.y_height * cd.y_width);
    PictureBatch k(conv_weights, cd.channels* cd.embeddings * cd.Ho * cd.Wo, cd.embeddings, cd.channels , cd.Ho , cd.Wo);
    if(debug){
        cout << "### k" << endl;
        k.print();
    }

    RowVector b(conv_bias, cd.embeddings);
    if(debug) {
        cout << "### b" << endl;
        b.print();
    }   

    Conv2d c2d(cd.channels,cd.embeddings, cd.Ho, cd.Wo, cd.Ho, cd.Wo, true);
    c2d.move_kernel(k);
    c2d.move_bias(b);
    bool c2d_bias = true, strict_img_size = true, dynamic_img_pad = false, use_norm = false; 
    PatchEmbed pe(
        cd.height, cd.width, cd.Ho, cd.Wo, cd.channels, cd.embeddings,
        c2d_bias, strict_img_size, dynamic_img_pad, use_norm
    ); //use norm set to true ==> use_pre_norm = false
    pe.move_c2d(c2d);

    Matrix pos_emb(positional_embeddings, cd.embeddings * (tokens + 1), (tokens + 1), cd.embeddings);
    if(debug){
        cout << "### positional embeddings" << endl;
        pos_emb.print();
    }
    // float cls_t[8] = {1.0, 0.0, 2.0, -1.0, 0.0, 0.5, 0.7, 1.0}; 
    // RowVector cls_token(cls_t,embeddings);
    vector<float> cls_tokens_f(cd.embeddings, 0.0f);
    RowVector cls_token(cls_tokens_f.data(),cd.embeddings); // all zeros
    if(debug){
        cout << "### class token" << endl;
        cls_token.print();
    }
    VisionTransformer cpu_vit(
        cd.height, cd.width,
        cd.Ho, cd.Wo, cd.channels,
        100, pool_token, cd.embeddings,
        12, 2, 4, true, false, 1.0
    );  
    cpu_vit.move_cls_token(cls_token);
    cpu_vit.move_pos_embed(pos_emb);

    PictureBatch x(x_data, cd.batch * cd.channels * cd.height * cd.width, cd.batch, cd.channels, cd.height, cd.width);
    if(debug){
        cout << "### x" << endl;
        x.print();
    }

    Tensor t;
    pe.forward(x, t);
    if(debug){
        cout << "### t" << endl;
        t.print();
    }

    Tensor y;
    cpu_vit.position_embed(t,y);
    if(debug){
        cout << "### y" << endl;
        y.print();
    }

    return y;
}

void cpu_gpu_comparison(){
    u_int batch = 2, channels = 3, height = 9, width = 9;
    u_int Ho = 3, Wo = 3, embeddings = 8; 
    u_int tokens = (height / Ho) * (width / Wo); //should be 9
    cout << "Test Patch Embed" << endl;
    
    vit_float k_data[8*3*3*3] = {
        -88.730, -93.081,  19.316,
        -27.174,  49.940,  37.719,
        -39.020,  86.735,  43.056,

        -11.937,  40.547, -17.649,
         25.842,  84.624,   7.097,
          8.233,  90.428,  84.885,

         84.913,  55.120,  74.480,
        -2.241,  38.433, -17.530,
         58.655, -54.583,  -4.333,



         29.413,  25.298, -53.512,
        -64.908, -53.491, -96.911,
         88.593,  45.439, -22.601,

         30.451, -98.619, -90.090,
         37.800,   8.083, -73.115,
        -99.743,  59.051, -69.488,

        -47.737,  88.094, -83.411,
         71.357,  15.275,  67.462,
         38.708, -64.405,  27.294,



         12.083, -52.698,  74.245,
         19.907,  30.086,  71.090,
        -46.041, -58.810,  63.740,

        -54.943, -24.221,  -8.640,
         98.345,  50.003, -39.889,
         27.074,  -5.487,  82.346,

         94.753,  66.510, -47.038,
         95.468, -66.575,  22.357,
         98.754,  55.686,   4.822,



        -30.965, -49.466, -20.177,
        -29.233, -78.812,  15.309,
        -83.495,  99.341,  40.782,

        -94.861,  65.642, -38.582,
          8.328,  44.350, -25.638,
        -84.217,  75.322,  -6.415,

         46.725,   2.152,  46.278,
         -8.779,  57.138,  56.135,
        -74.049, -88.487,  53.438,



         -2.630, -72.963,  56.555,
         11.010, -47.636,  10.358,
        -75.197,  32.915, -73.366,

        -31.381,  -3.444,  70.352,
        -20.552, -74.075, -70.557,
         62.374, -59.349,  76.740,

         77.752, -31.352,   8.564,
        -74.004,  28.564,   9.775,
        -33.609,  25.998,  63.140,



        -99.265, -57.720,  28.233,
        -75.953, -29.514, -16.720,
         42.485, -97.906, -66.119,

        -92.978, -59.572,  24.809,
        -61.123,  -7.783, -32.180,
         88.633,  87.414, -91.674,

        -83.951, -31.502, -61.103,
         98.899, -39.559,  74.920,
         -0.475, -30.656, -40.841,



         46.087,  83.957,  99.166,
        -82.996, -93.025, -99.918,
         69.672,  -1.628, -71.019,

         59.040,  -5.567, -23.223,
         17.478, -75.988,  -1.184,
         48.975,  22.450,   6.595,

         17.817,  81.462,  -3.339,
         17.241,   5.225,  38.536,
        -20.782, -45.011,  76.046,



        -27.813, -64.398,  28.924,
        -54.210, -49.066,  16.549,
        -97.581,  31.564, -75.405,

          6.526,  61.920, -12.924,
         11.075,  -3.247, -20.403,
        -84.149,  27.273,   3.218,

          0.449,  11.900,  -8.930,
        -75.704, -79.089, -27.307,
        -86.749,  86.090, -33.020
    };
    PictureBatch k(k_data, 8*3*3*3, 8, 3, 3, 3);
    cout << "### k" << endl;
    k.print();

    vit_float b_data[8] = {-96.814, 9.515, -28.606, 84.045, 68.013, -97.364, -24.707, -3.075};
    RowVector b(b_data, 8);
    cout << "### b" << endl;
    b.print();

    // vit_float ng_data[8] = {58.576, -63.604, 39.352, 44.173, -6.129, 22.081, -23.061, -15.240};
    // RowVector ng(ng_data, 8);
    // cout << "### ng" << endl;
    // ng.print();

    // vit_float nb_data[8] = {-16.154, 4.198, -34.143, 2.215, -72.475, -53.607, -8.689, -30.214};
    // RowVector nb(nb_data, 8);
    // cout << "### nb" << endl;
    // nb.print();

    vit_float x_data[2*3*9*9] = {
          4.331, -80.581,  20.891,  59.280, -55.102, -52.729, -86.093,  72.703, -28.591,
         36.849, -69.260, -70.523, -12.793,  17.444, -65.791,  32.285,  84.994, -98.686,
         43.771,  29.459,  88.093,   8.496,  83.935, -64.277,  14.534,   5.508, -21.926,
        -53.408, -92.816, -29.048,  35.933, -12.533, -27.686, -78.971,  -8.073,  11.023,
        -90.314, -87.864, -60.096,   5.151,  32.543, -83.185,  46.169,  61.969,  45.266,
         64.916,  68.820,   7.936, -96.336, -99.154, -55.148,  88.678,  -6.596,  62.691,
         77.528, -11.259,  71.784,  11.277,  90.984,   1.591,  41.496,  18.751,  51.403,
        -35.560, -32.069,  86.941,  55.848, -86.699, -10.894,  23.669,  -1.278, -46.756,
        -19.328,  70.710,  84.120, -67.115, -72.694, -54.554,  -6.338, -45.485,  98.850,

        -81.852, -27.812, -78.824,  94.563,  41.157, -12.367,  13.588,  67.943,  40.364,
        -92.329, -99.156,  86.991, -93.383, -46.160, -90.091,  15.718,  15.657,  69.040,
         62.681,  44.965,   7.159,  56.774, -92.153, -37.702,  83.942,  -7.332, -59.305,
        -68.979, -45.227, -92.204, -83.414, -30.902,  57.136, -78.906,  12.470,   9.339,
        -55.045,  -7.929, -31.355,   3.835, -37.100,  79.473,  92.108,  46.796, -30.676,
         88.872,  90.636,  54.969,  16.338, -75.786,  45.240,  46.523, -79.954, -76.263,
         53.660, -67.246,  -7.755,  94.126, -29.318, -31.994, -94.726,  57.293,  62.970,
        -21.722, -30.768,  92.464,  47.660,  27.629, -18.000,  67.905, -14.915, -75.160,
        -21.458,   0.850,   9.546, -38.282,  88.526,   7.535, -71.701,  73.679, -26.444,

        38.213, -78.117, -76.558, -67.867,  97.676,  58.044,  46.521,  -3.886, -70.774,
          7.551,  -8.768,  -0.171,   5.159,  76.543, -56.519,  76.110,  80.876,  17.192,
         15.610,  62.008, -95.821,  68.182, -79.625,  34.017,  17.694, -47.620,  34.172,
        -18.202, -36.839,   5.733,  48.756,  -5.878,  87.252, -28.882,  97.228,  13.068,
        -57.135,  72.437,  36.921,  57.213,  35.288,  19.898,  80.545,  89.963,  88.961,
         92.579, -34.539,  12.915, -96.821,  49.423,  87.221, -39.861,  36.366, -50.769,
        -85.008,  38.237, -78.583, -80.427,  81.129,  22.857,  81.531,  60.476, -46.508,
         20.100,   4.174,  46.550,  -6.862,  33.979, -12.110, -92.874,  44.329, -70.896,
         69.364,  42.249,  32.805,  -5.522, -45.768,  80.226,  22.682,  41.706,  90.035,



        -37.820,  33.077,   0.256,  37.946, -51.970, -67.732, -99.723, -47.815,  35.256,
         -6.184,  38.110,  88.777,  55.116,  -5.896,  44.042, -93.272,  31.644, -55.087,
         27.236, -13.559, -48.797, -48.809, -94.204,  35.054, -97.819,  80.703,   3.400,
        -93.295, -54.977,  89.360,  45.993,  53.748,  12.328, -23.969,  97.564, -17.448,
        -72.213, -86.207, -94.119,  79.184, -78.604,  39.427,  99.433,   9.213,  -0.235,
         52.742,  48.590, -96.871,  81.923,  56.694,  -6.146, -23.039,  80.763, -77.348,
        -14.965,  89.828,  74.151,  49.083, -33.078,  41.293,  33.850,  84.793,  94.480,
         21.612, -88.861,  82.155,   3.442,  68.481,  27.021, -40.103,  52.852, -97.400,
         81.265,  52.820, -19.356,  79.941, -96.902,  85.890, -47.348, -31.633,  43.943,

        -98.821, -69.902, -15.087,   8.972,  22.061, -40.141, -92.931, -11.749,  28.189,
          2.109,  37.483,  18.695,  54.627,  65.276,  86.224,  72.363, -59.992,  81.345,
         -0.855,  43.162,   5.204,  27.923,  82.022, -31.387, -67.215, -53.621,  -5.972,
          9.339,   0.832,  90.834,  69.378,  26.955,  -8.716,  78.409, -69.764, -15.182,
        -29.223, -64.675,  86.129, -33.505, -56.260,  10.077,  62.332, -49.303,  15.040,
         18.065,  62.164,  35.867,   9.539, -82.569,  32.096,  53.012,  99.149, -34.830,
         -2.312,  50.620, -11.212, -50.428,  82.843,  34.432,  68.442,   7.451,  -3.007,
         42.609, -52.782, -73.495, -70.345,  58.981, -42.240,  39.222, -20.131, -78.952,
        -68.380,   6.147,  59.459,  -1.574, -73.983,   0.998,  80.763,  54.310, -27.961,

         48.271, -21.585, -60.041,  20.350, -99.212,   4.651, -19.649,  80.559, -21.151,
         56.659,  66.687,  76.469,  72.017, -22.844,  90.344,  23.728,  70.194,  11.536,
         84.477,  44.972,  88.878,  11.603, -96.580,  42.984,  37.546, -10.372,  50.727,
         -2.900,  88.169,  20.322,   0.090, -75.514, -69.116, -23.171, -54.631,  24.190,
         55.183,  97.461,  63.399,  81.415,  12.416,  85.283, -64.218,  13.327,  66.687,
        -81.501, -49.371,  15.856, -66.861, -79.121,  99.704,  -3.138,  77.537, -39.014,
          7.350,  16.695,  72.435, -77.277, -74.559, -38.696,  -9.179,   5.915,   9.098,
         63.440, -81.730,  52.738, -43.638, -39.950,  87.473, -12.099,  82.361, -21.801,
         44.867,   6.878,  98.599, -64.293,  96.124,  -4.289,   5.212,  39.614,  60.995
    };
    PictureBatch x(x_data, 2*3*9*9, 2, 3, 9, 9);
    cout << "### x" << endl;
    x.print();

    Conv2d c2d(3, 8, 3, 3, 3, 3, true);
    c2d.move_kernel(k);
    c2d.move_bias(b);
    bool c2d_bias = true, strict_img_size = true, dynamic_img_pad = false, use_norm = false; 
    PatchEmbed pe(9, 9, 3, 3, 3, 8, c2d_bias, strict_img_size, dynamic_img_pad, use_norm); //use norm set to true ==> use_pre_norm = false
    pe.move_c2d(c2d);
    
    // Position embeddings, embeddings * (tokens + cls_token) = 8 * (9 + 1)
    vit_float pos_e[10*8] = {
        0.0, 1.0, 2.0, 3.0, 0.0, 0.5, 0.1, 0.2,
        0.0, 1.0, 2.0, 3.0, 0.0, 0.5, 0.1, 0.2,
        5.0, 1.0, 2.0, 3.0, 0.0, 0.5, 0.1, 0.2,
        0.0, 1.0, 2.0, 4.0, 0.0, 0.5, 0.1, 0.2,
        0.0, 1.0, 2.0, 3.0, 0.0, 0.5, 0.1, 0.2,
        0.0, 1.0, 2.0, 3.0, 0.0, 0.5, 0.1, 0.2,
        0.0, 1.0, 2.0, 3.0, 0.0, 0.5, 0.1, 0.2,
        1.2, 1.0, 2.0, 3.0, 0.0, 0.5, 0.1, 0.3,
        0.0, 1.0, 2.0, 3.0, 0.9, 1.5, 0.1, 0.2,
        0.0, 1.0, 2.0, 3.0, 0.0, 0.5, 0.1, 0.2
    };
    Matrix pos_emb(pos_e, embeddings * 10, 10, embeddings);
    // float cls_t[8] = {1.0, 0.0, 2.0, -1.0, 0.0, 0.5, 0.7, 1.0}; 
    // RowVector cls_token(cls_t,embeddings);
    RowVector cls_token(embeddings); // all zeros

    VisionTransformer cpu_vit(
        height, width,
        Ho, Wo, channels,
        100, pool_token, embeddings,
        12, 2, 4, true, false, 1.0
    );  
    cpu_vit.move_cls_token(cls_token);
    cpu_vit.move_pos_embed(pos_emb);
    // cpu_vit.move_patch_embed(pe);

    // -- COMPARISON --
    // CPU reference
    Tensor t;
    pe.forward(x, t);
    cout << "### t" << endl;
    t.print();

    Tensor y;
    cpu_vit.position_embed(t,y);
    cout << "### y" << endl;
    y.print();

    //GPU reference
    /*
    d_pic: [B,C,H,W]
    d_x: [B,T+1,C]
    */
    half * gpu_pic, * gpu_out;
    half * gpu_bias, * gpu_pos_emb, * gpu_conv_weights;
    gpu_pic = (half *)malloc(sizeof(half) * batch * channels * height * width);
    gpu_out = (half *)malloc(sizeof(half) * batch * (tokens + 1) * embeddings);
    f32_to_f16(x_data, gpu_pic, batch * channels * height * width);

    gpu_bias = (half *)malloc(sizeof(half) * embeddings);
    gpu_pos_emb = (half *)malloc(sizeof(half) * (tokens + 1) * embeddings);
    gpu_conv_weights = (half *)malloc(sizeof(half) * channels * embeddings * Ho * Wo);

    f32_to_f16(b_data, gpu_bias, embeddings);
    f32_to_f16(pos_e, gpu_pos_emb, embeddings * (tokens + 1));
    f32_to_f16(k_data, gpu_conv_weights, channels * embeddings * Ho * Wo);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));
    CUDNN_CHECK(cudnnSetStream(handle, stream));

    convolution_dim conv_dim(batch,channels,height,width,embeddings, Ho, Wo);

    // Create the instance
    GpuPatchEmbedder gpu_pe(
      stream,
      handle,
      conv_dim
    );
    cudaStreamSynchronize(stream);
    // Load the weights
    gpu_pe.load_weights_data(gpu_conv_weights, gpu_bias, gpu_pos_emb, false);
    cudaStreamSynchronize(stream); /*Need to sync cause every op is async*/
    gpu_pe.load_pics(gpu_pic);
    cudaStreamSynchronize(stream); /*Need to sync cause every op is async*/
    gpu_pe.forward(gpu_out, false, true); //Copying on host directly
    cudaStreamSynchronize(stream);
    cout << "CPU/GPU avg. value difference: " << compare_results(y,gpu_out) << endl;
}

void gpu_comparison(bool debug){
    u_int batch = 64, channels = 3, height = 224, width = 224;
    u_int Ho = 16, Wo = 16, embeddings = 768;
    if(debug){
        batch = 8; channels = 3; height = 16; width = 16;
        Ho = 4; Wo = 4; embeddings = 10;  
    }
    convolution_dim conv_dim(batch,channels,height,width,embeddings,Ho,Wo);
    u_int tokens = conv_dim.y_height * conv_dim.y_width;
	cout << "X: [" << batch << ","<< channels << ","<< height << ","<< width << "]" << endl;
    cout << "W: [" << embeddings << ","<< channels << ","<< Ho << ","<< Wo << "]" << endl;
    cout << "Final embedded tensor: [" << batch << ","<< tokens + 1 << ","<< embeddings << "]" << endl;
    cout << "debug: " << yesno(debug) << endl;
    u_int input_pic_elements_num = batch * channels * height * width;
    u_int conv_kernel_elements_num = channels * embeddings * Ho * Wo;
    // u_int flatten_elements_num = batch * tokens * embeddings;
    u_int embedded_elements_num = batch * (tokens + 1) * embeddings;
    
    float * h_pic, * h_out;
    float * h_bias, * h_pos_emb, * h_conv_weights;
    h_pic = (float *)malloc(sizeof(float) * input_pic_elements_num);
    h_bias = (float *)malloc(sizeof(float) * embeddings);
    h_pos_emb = (float *)malloc(sizeof(float) * embeddings * (tokens + 1));
    h_conv_weights = (float *)malloc(sizeof(float) * conv_kernel_elements_num);

    h_out = (float *)malloc(sizeof(float) * embedded_elements_num);

    // Random generation
    /*Host mem for gpu implementation should be pinned!*/
    half * gpu_pic, * gpu_out;
    half * gpu_bias, * gpu_pos_emb, * gpu_conv_weights;
    cudaHostAlloc(&gpu_pic,sizeof(half) * input_pic_elements_num, cudaHostAllocDefault);
    cudaHostAlloc(&gpu_out,sizeof(half) * embedded_elements_num,cudaHostAllocDefault);
    cudaHostAlloc(&gpu_bias,sizeof(half) * embeddings,cudaHostAllocDefault);
    cudaHostAlloc(&gpu_pos_emb,sizeof(half) * embeddings * (tokens + 1),cudaHostAllocDefault);
    cudaHostAlloc(&gpu_conv_weights,sizeof(half) * conv_kernel_elements_num,cudaHostAllocDefault);

    u_long seed = std::chrono::high_resolution_clock::now()
        .time_since_epoch()
        .count();
    rand_init(h_pic, input_pic_elements_num, 1.0f, seed);
    rand_init(h_bias, embeddings, 0.1f, seed);
    rand_init(h_pos_emb, (tokens + 1) * embeddings, 1.0f, seed);
    rand_init(h_conv_weights, conv_kernel_elements_num, 1.0f, seed);
    f32_to_f16(h_pic ,gpu_pic, input_pic_elements_num);
    f32_to_f16(h_bias,gpu_bias, embeddings);
    f32_to_f16(h_pos_emb ,gpu_pos_emb, (tokens + 1) * embeddings);
    f32_to_f16(h_conv_weights,gpu_conv_weights, conv_kernel_elements_num);


    // CPU REFERENCE
    Tensor y_cpu = cpu_baseline(conv_dim, h_conv_weights, h_bias, h_pos_emb, h_pic, debug);
    
    // GPU Single Stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));
    CUDNN_CHECK(cudnnSetStream(handle, stream));

    GpuPatchEmbedder gpu_pe(
      stream,
      handle,
      conv_dim
    );

    cudaStreamSynchronize(stream);
    gpu_pe.load_weights_data(gpu_conv_weights, gpu_bias, gpu_pos_emb, false);
    gpu_pe.load_pics(gpu_pic);
    gpu_pe.forward(gpu_out, false, debug); //Copying on host directly
    cudaStreamSynchronize(stream);
    if(debug) {
        cout << "gpu_out:" << endl;
        f16_to_f32(gpu_out, h_out, embedded_elements_num);
        Tensor gpu_debug(h_out, embedded_elements_num, batch, tokens + 1, embeddings);
        gpu_debug.print();
    }
    cout << " Comparison CPU/GPU: " << compare_results(y_cpu, gpu_out) << endl;

    // GPU Multi Stream

    //initialize the streams
    cudaStream_t streams[STREAM_N];
    cudnnHandle_t handles[STREAM_N];
    GpuPatchEmbedder sub_gpu_pe[STREAM_N];

    assert(batch % STREAM_N == 0);
    u_int minibatch = batch / STREAM_N; 
    conv_dim.batch = minibatch;
    cout << "minibatch:" << minibatch << endl;
    
    for(int i = 0; i< STREAM_N; i++){
        cudaStreamCreate(&streams[i]);

        CUDNN_CHECK(cudnnCreate(&handles[i]));
        CUDNN_CHECK(cudnnSetStream(handles[i], streams[i]));
        
        sub_gpu_pe[i] = GpuPatchEmbedder(
            streams[i],
            handles[i],
            conv_dim
        );

        sub_gpu_pe[i].set_weights_data(gpu_pe.d_w, gpu_pe.d_bias, gpu_pe.d_pos_emb);

    }   

    //Copy the respective input data and set the pointers to the weights data and run
    half * actual_pic = gpu_pic;
    half * actual_out = gpu_out;
    for(int i = 0; i< STREAM_N; i++){
        sub_gpu_pe[i].load_pics(actual_pic); //This should load a minibatch of pics
        // cudaStreamSynchronize(streams[i]);
        sub_gpu_pe[i].forward(debug);
        /*Should not be necessary, because on the same stream */
        // CUDA_CHECK(cudaMemcpyAsync(actual_out, sub_gpu_pe[i].d_x, sizeof(half) * minibatch * (tokens + 1) * embeddings, cudaMemcpyDeviceToHost, streams[i]));
    
        actual_pic += minibatch * channels * height * width; //go to the next minibatch of images 
        actual_out += minibatch * (tokens + 1) * embeddings;
    }
    
    for (size_t i = 0; i < STREAM_N; i++){
        cudaStreamSynchronize(streams[i]);
    }
    

    //compare the results
    cout << " Comparison CPU/GPU Streams: " << compare_results(y_cpu, gpu_out) << endl;


    sub_gpu_pe[0].free_weights(); /*Only one istance!*/

}

int main() {
    test_type test = GPU_COMPARISON;
    if(test == CPU_COMPARISON){
        cpu_gpu_comparison();
    }
    else{
        gpu_comparison(false);
    }

    return 0;
}