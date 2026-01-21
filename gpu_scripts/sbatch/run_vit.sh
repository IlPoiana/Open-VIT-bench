#!/bin/bash

#SBATCH --job-name=vit
#SBATCH --output=gpu_out/block%j.out  
#SBATCH --error=gpu_out/block_error%j.err
#SBATCH --partition=edu-long 
#SBATCH --nodes=1
#SBATCH --gres=gpu:a30.24:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
ml cuDNN/8.9.2.26-CUDA-12.1.1
make 
srun test_bin/block_bench.exe --batch 128 --kernel 3 --mlp_type 1