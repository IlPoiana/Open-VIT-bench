mkdir gpu_out
ml cuDNN/8.9.2.26-CUDA-12.1.1
make clean
# export CUDNN_LOGINFO_DBG=3
# export CUDNN_LOGDEST_DBG=stdout
make test_bin/test_cudnn_conv2d.exe ARCH=-arch=sm_80