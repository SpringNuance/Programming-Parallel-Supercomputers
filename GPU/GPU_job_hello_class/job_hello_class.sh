#!/bin/bash -l
#SBATCH --nodes 2  #Number of nodes, maximally 2
#SBATCH --ntasks-per-node 4 #Number of tasks per node
#SBATCH --gpus-per-node 4 #Number of GPUs per node
#SBATCH -t 00:00:59
#SBATCH --partition courses-gpu 
#SBATCH --account courses 

module purge
module load openmpi

nvcc -o HC hello_class.cu -lmpi

srun ./HC
