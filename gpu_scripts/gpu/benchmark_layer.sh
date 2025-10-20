#!/bin/bash

#SBATCH --job-name=layer
#SBATCH --output=gpu_out/benchmark_layer%j.out  
#SBATCH --error=gpu_out/benchmark_layer_error%j.err
#SBATCH --partition=edu-short #edu-long
#SBATCH --nodes=1
#SBATCH --gres=gpu:a30.24:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
ml cuDNN/8.9.2.26-CUDA-12.1.1
srun nsys profile -o gpu_out/gpu_layer test_bin/test_gpu_layer.exe
