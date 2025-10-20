#!/bin/bash

#SBATCH --job-name=mlp_bench
#SBATCH --output=gpu_out/benchmark_mlp%j.out  
#SBATCH --error=gpu_out/benchmark_mlp_error%j.err
#SBATCH --partition=edu-medium #edu-long
#SBATCH --nodes=1
#SBATCH --gres=gpu:a30.24:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
ml cuDNN/8.9.2.26-CUDA-12.1.1
srun nsys profile -o gpu_out/gpu_mlp test_bin/test_gpu_mlp.exe
