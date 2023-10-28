#!/bin/bash -l
#SBATCH --gres=gpu:teslap100:1
#SBATCH --time=00:05:00
#SBATCH --output=vecAdd_GPU.out
#SBATCH --partition=courses

module load gcc cuda 

nvcc -o vecAdd_GPU vecAdd_GPU.cu

time srun vecAdd_GPU
