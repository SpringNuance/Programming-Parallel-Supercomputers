#!/bin/bash -l
#SBATCH --gpus-per-node 1
#SBATCH --time=00:05:00
#SBATCH --output=BA.out
#SBATCH --account=courses
#SBATCH --partition=courses-gpu

nvcc -o BA Blocking_Async.cu

time srun BA
