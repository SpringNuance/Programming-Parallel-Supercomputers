#!/bin/bash -l
#SBATCH --gres=gpu:teslap100:1
#SBATCH --time=00:05:00
#SBATCH --output=BA.out

module load gcc cuda 

nvcc -o BA Blocking_Async.cu

time srun BA
