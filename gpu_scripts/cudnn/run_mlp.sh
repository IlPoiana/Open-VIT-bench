#!/bin/bash

#SBATCH --job-name=mlp
#SBATCH --output=gpu_out/mlp%j.out  
#SBATCH --error=gpu_out/mlp_error%j.err
#SBATCH --partition=edu-short #edu-long
#SBATCH --nodes=1
#SBATCH --gres=gpu:a30.24:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
ml cuDNN/8.9.2.26-CUDA-12.1.1
export CUDNN_LOGINFO_DBG=1
export CUDNN_LOGERR_DBG=1
export CUDNN_LOGDEST_DBG=stderr
# export CUDNN_FRONTEND_LOG_INFO=1
# export CUDNN_FRONTEND_LOG_FILE=stdout 
srun test_bin/test_cudnn_mlp.exe