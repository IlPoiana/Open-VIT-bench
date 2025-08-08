#include "../gpu_include/gpu_conv2d.h"
#include "../include/utils.h"

#include <iostream>
#include <assert.h>
#include <vector>
#include <map>

#include <sys/time.h>
using namespace std;

/*
BENCHMARK FILE FOR CONV2D
GATHER THE TIME FOR THE ENTIRE APPLICATION: 
- GPU MEM ALLOCATION
- STREAM ALLOCATION
- KERNEL EXECUTION

*/

benchmark_time timed_single_channel(vit_float * x_data, vit_float * k_data, picture_shape x_shape, conv_kernel_shape kernel_shape, vit_float * bias) {
	// cout << "Test 2D Convolution" << endl;
	vector<float> preprocess_time;
	float kernel_time = 0;
	benchmark_time time = {preprocess_time, kernel_time};

	struct timeval start={0,0};
	struct timeval stop={0,0};    
	float elapsed_time;

    PictureBatch k(
        k_data,
        kernel_shape.in_channels * kernel_shape.out_channels * kernel_shape.H * kernel_shape.W,
        kernel_shape.in_channels,
        kernel_shape.out_channels,
        kernel_shape.H,
        kernel_shape.W
    );
    // cout << "### kernel" << endl;
    // k.print();

    // cout << "### bias" << endl;
    // b.print();

    //create the class and initialize the stream
	gettimeofday(&start, (struct timezone*)0);
    GPUPictureBatch x(
        x_data,
        x_shape.B * x_shape.C * x_shape.H * x_shape.W,
        x_shape,
        false,
        kernel_shape
    );
	gettimeofday(&stop, (struct timezone*)0);
	elapsed_time = ((stop.tv_sec-start.tv_sec)*1.e6+(stop.tv_usec-start.tv_usec));
	time.preprocess.push_back(elapsed_time);

	//create the class and initialize the stream
	gettimeofday(&start, (struct timezone*)0);
    PictureBatch t(x_data,
        x_shape.B * x_shape.C * x_shape.H * x_shape.W,
        x_shape.B,
        x_shape.C,
        x_shape.H,
        x_shape.W
    );
	gettimeofday(&stop, (struct timezone*)0);
	elapsed_time = ((stop.tv_sec-start.tv_sec)*1.e6+(stop.tv_usec-start.tv_usec));
	time.preprocess.push_back(elapsed_time);

	// t.print();

    GPUConv2d c2d(kernel_shape, true);
    c2d.set_stream(x.get_stream());
    c2d.create_handle();
    c2d.bind_handle();

    c2d.move_kernel(k); //load it into the GPU

    RowVector b(bias, c2d.get_out_channels());
    c2d.move_bias(b); // load it into the GPU


    PictureBatch y;
    c2d.timed_forward(x, y, 0, time); // compute the forward and save it to y
	// cout << "finished forward" << endl;
    // vector<string> preprocess_names = {"gpu_allocation","cpu_allocation"};
    // print_json_time(time,preprocess_names);

    c2d.kernel_free();
    //stream should be already free
    x.data_free();
    // y.print();
    return time;
}

benchmark_time timed_unified_channel(vit_float * x_data, vit_float * k_data, picture_shape x_shape, conv_kernel_shape kernel_shape, vit_float * bias)  {
	// cout << "Test 2D Convolution" << endl;

	vector<float> preprocess_time;
	float kernel_time = 0;
	benchmark_time time = {preprocess_time, kernel_time};

    PictureBatch k(
        k_data,
        kernel_shape.in_channels * kernel_shape.out_channels * kernel_shape.H * kernel_shape.W,
        kernel_shape.in_channels,
        kernel_shape.out_channels,
        kernel_shape.H,
        kernel_shape.W
    );
    // cout << "### kernel" << endl;
    // k.print();

    // vit_float b_data[4] = {-5.247, -10.884, -60.649, 51.270};
    // RowVector b(b_data, 4);
    // cout << "### bias" << endl;
    // b.print();


    //create the class and initialize the stream
    GPUPictureBatch x(
        x_data,
        x_shape.B * x_shape.C * x_shape.H * x_shape.W,
        x_shape,
        true,
        kernel_shape,
        time
    );
    GPUConv2d c2d(kernel_shape, true);
    c2d.set_stream(x.get_stream());
    c2d.create_handle();
    c2d.bind_handle();

    c2d.move_kernel(k); //load it into the GPU
    
    RowVector b(bias, c2d.get_out_channels());
    c2d.move_bias(b); // load it into the GPU


    PictureBatch y;
    c2d.timed_forward(x, y, 1,time); // compute the forward and save it to y
    // cout << "finished forward" << endl;
    // vector<string> preprocess_names = {"linearize"};
    // print_json_time(time,preprocess_names);

    c2d.kernel_free();
    //stream should be already free
    x.data_free();
    // cout<< y.get_B()<< " | " <<y.get_C() << " | " <<y.get_H() * y.get_W() <<endl; 
    // y.print();
    return time;
}

benchmark_time timed_parallelized(vit_float * x_data, vit_float * k_data, picture_shape x_shape, conv_kernel_shape kernel_shape, vit_float * bias)  {
    // cout << "Test 2D Convolution" << endl;

	vector<float> preprocess_time;
	float kernel_time = 0;
	benchmark_time time = {preprocess_time, kernel_time};

    PictureBatch k(
        k_data,
        kernel_shape.in_channels * kernel_shape.out_channels * kernel_shape.H * kernel_shape.W,
        kernel_shape.in_channels,
        kernel_shape.out_channels,
        kernel_shape.H,
        kernel_shape.W
    );
    // cout << "### kernel" << endl;
    // k.print();

    // vit_float b_data[4] = {-5.247, -10.884, -60.649, 51.270};
    // RowVector b(b_data, 4);
    // cout << "### bias" << endl;
    // b.print();


    //create the class and initialize the stream
    GPUPictureBatch x(
        x_data,
        x_shape.B * x_shape.C * x_shape.H * x_shape.W,
        x_shape,
        true,
        kernel_shape,
        time
    );
    GPUConv2d c2d(kernel_shape, true);
    c2d.set_stream(x.get_stream());
    c2d.create_handle();
    c2d.bind_handle();

    c2d.move_kernel(k); //load it into the GPU

    RowVector b(bias, c2d.get_out_channels());
    c2d.move_bias(b); // load it into the GPU


    PictureBatch y;
    c2d.timed_forward(x, y, 2,time); // compute the forward and save it to y
    // cout << "finished forward" << endl;
    // vector<string> preprocess_names = {"linearize","create_streams","malloc"};
    // print_json_time(time,preprocess_names);

    c2d.kernel_free();
    //stream should be already free
    x.data_free();
    // cout<< y.get_B()<< " | " <<y.get_C() << " | " <<y.get_H() * y.get_W() <<endl; 
    return time;
}


int main(int argc, char* argv[]){
    //This will be fetched from a config file
    const u_int runs_n = 2;
    const u_int warm_up = 1;
	
    assert(argc > 3);
    
    //loading the batch
    //path to the cpic directory
    const string cpic_path = argv[1]; //data/pic_$i.cpic
    PictureBatch pic;
    load_cpic(cpic_path, pic);
    
    //loading the kernel
    const string cvit_path = argv[2]; //models/vit_1.cvit
    VisionTransformer vit;
    load_cvit(cvit_path, vit);
    // printf("%d,%d\n", pic.get_B(), pic.get_C());
    //decide what kernel you want to bench
	int level = atoi(argv[3]);
    // printf("%s - %d\n", argv[3],level);
    // return 0;
	assert(level < 3);

    int k_s[6];
    vit.get_kernel_shape(k_s);
    conv_kernel_shape kernel_shape(k_s);
    picture_shape x_shape = {(int)pic.get_B(),(int)pic.get_C(),(int)pic.get_H(),(int)pic.get_W()};
    
    benchmark_time single_time;
    switch (level){
		case 0:
            cout << "[" << endl;
            for(u_int idx = 0; idx < warm_up + runs_n; idx++){
                single_time = timed_single_channel(pic.get_data(),vit.get_conv2d_kernel(), x_shape,kernel_shape, vit.get_conv2d_bias());
                if(idx >= warm_up){
                    cout << "{" << endl;
                    print_json_time(single_time,{"gpu_allocation","cpu_allocation"});
                    (idx != warm_up + runs_n - 1) ? (cout << "}," << endl) : cout << "}" << endl;
                }
            }
            cout << "]" << endl;
            break;
		case 1:
            cout << "[" << endl;
            for(u_int idx = 0; idx < warm_up + runs_n; idx++){
                single_time = timed_unified_channel(pic.get_data(),vit.get_conv2d_kernel(), x_shape,kernel_shape, vit.get_conv2d_bias());
                if(idx >= warm_up){
                    cout << "{" << endl;
                    print_json_time(single_time,{"linearize"});
                    (idx != warm_up + runs_n - 1) ? (cout << "}," << endl) : cout << "}" << endl;cout << "}," << endl;
                }
            }
            cout << "]" << endl;
			break;
		default:
			cout << "[" << endl;
            for(u_int idx = 0; idx < warm_up + runs_n; idx++){
                single_time = timed_parallelized(pic.get_data(),vit.get_conv2d_kernel(), x_shape,kernel_shape, vit.get_conv2d_bias());
                if(idx >= warm_up){
                    cout << "{" << endl;
                    print_json_time(single_time,{"linearize","create_streams","malloc"});
                    (idx != warm_up + runs_n - 1) ? (cout << "}," << endl) : cout << "}" << endl;
                }
            }
            cout << "]" << endl;
            break;
	}

    //final print (json like)
    /*
    _info.txt
    dataset dimension: 8 batches
    minimum batch size: 4 pictures
    maximum batch size: 16 pictures
    channel dimension: 3
    picture height: 224
    picture width: 224
    pixel minimum value: 0.0
    pixel maximum value: 1.0
    dataset dimension: 8 batches
    minimum batch size: 4 pictures
    maximum batch size: 16 pictures
    channel dimension: 3
    picture height: 224
    picture width: 224
    pixel minimum value: 0.0
    pixel maximum value: 1.0
    */

    /* "time": {
        "preprocess": 1,
        "kernel": 1
    }
    */
	return 0;
}



