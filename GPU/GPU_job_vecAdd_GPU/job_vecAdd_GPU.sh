#!/bin/bash -l
#SBATCH --gpus-per-node 1
#SBATCH --time=00:05:00
#SBATCH --output=vecAdd_GPU.out
#SBATCH --account=courses
#SBATCH --partition=courses-gpu

nvcc -o vecAdd_GPU vecAdd_GPU.cu

time srun vecAdd_GPU
