mkdir gpu_out
ml cuDNN/8.9.2.26-CUDA-12.1.1
make clean
make test_bin/test_gpu_layer.exe ARCH=-arch=sm_80