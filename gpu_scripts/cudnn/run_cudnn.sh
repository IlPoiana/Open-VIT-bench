#!/bin/bash

#SBATCH --job-name=cudnn_0
#SBATCH --output=gpu_out/cudnn%j.out  
#SBATCH --error=gpu_out/cudnn_error%j.err
#SBATCH --partition=edu-short #edu-long
#SBATCH --nodes=1
#SBATCH --gres=gpu:a30.24:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
mkdir gpu_out
ml cuDNN/8.9.2.26-CUDA-12.1.1
make clean
make obj/cudnn_backend_conv
srun obj/cudnn_backend_conv 