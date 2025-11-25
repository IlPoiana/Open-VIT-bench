ml cuDNN/8.9.2.26-CUDA-12.1.1
srun --job-name=block --partition=edu-medium --nodes=1 --gres=gpu:a30.24:1 --ntasks-per-node=1 --cpus-per-task=1 bash