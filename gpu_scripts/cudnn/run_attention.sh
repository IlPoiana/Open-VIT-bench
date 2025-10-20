#!/bin/bash

#SBATCH --job-name=mha
#SBATCH --output=gpu_out/attention%j.out  
#SBATCH --error=gpu_out/attention_error%j.err
#SBATCH --partition=edu-short #edu-long
#SBATCH --nodes=1
#SBATCH --gres=gpu:a30.24:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
ml cuDNN/8.9.2.26-CUDA-12.1.1
# export CUDNN_LOGINFO_DBG=3
# export CUDNN_LOGDEST_DBG=stdout
# export CUDNN_FRONTEND_LOG_INFO=1
# export CUDNN_FRONTEND_LOG_FILE=stdout 
srun test_bin/test_cudnn_attention.exe