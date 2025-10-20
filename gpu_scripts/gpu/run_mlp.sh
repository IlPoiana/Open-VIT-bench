#!/bin/bash

#SBATCH --job-name=mlp
#SBATCH --output=gpu_out/mlp%j.out  
#SBATCH --error=gpu_out/mlp_error%j.err
#SBATCH --partition=edu-short #edu-long
#SBATCH --nodes=1
#SBATCH --gres=gpu:a30.24:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

export CUBLASLT_LOG_LEVEL=2
ml cuDNN/8.9.2.26-CUDA-12.1.1
srun test_bin/test_gpu_mlp.exe