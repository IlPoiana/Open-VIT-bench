#!/bin/bash

#SBATCH --job-name=block
#SBATCH --output=gpu_out/block%j.out  
#SBATCH --error=gpu_out/block_error%j.err
#SBATCH --partition=edu-short #edu-long
#SBATCH --nodes=1
#SBATCH --gres=gpu:a30.24:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
ml cuDNN/8.9.2.26-CUDA-12.1.1
# export CUDNN_LOGINFO_DBG=3
# export CUDNN_LOGDEST_DBG=stdout
srun test_bin/test_gpu_block.exe