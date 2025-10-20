#!/bin/bash

#SBATCH --job-name=my_job
#SBATCH --output=my_job%j.out  
#SBATCH --error=my_error_%j.err
#SBATCH --partition=edu-short #edu-long
#SBATCH --nodes=1
#SBATCH --gres=gpu:a30.24:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

nvidia-smi

