#!/bin/bash -l
#SBATCH --nodes 1  #Number of nodes, maximally 2
#SBATCH --ntasks 4 #Number of tasks; nodes x number of GPUs
#SBATCH --gres=gpu:teslap100:4 #Number of GPUs
#SBATCH -t 00:00:59
#SBATCH --partition courses 
#SBATCH --account courses 

module load gcc cuda cmake openmpi

nvcc -o HC hello_class.cu -lmpi

srun ./HC
