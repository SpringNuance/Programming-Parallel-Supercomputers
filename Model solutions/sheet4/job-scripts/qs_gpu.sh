#!/bin/bash -l
##Triton
#SBATCH -A courses
#SBATCH -p courses-gpu
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=500M
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

module purge
module load gcc/11.3.0 cmake/3.26.3 openmpi/4.1.5

srun ../build/quicksort-gpu


