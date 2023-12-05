#!/bin/bash -l
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=500M
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=vecAdd_CPU.out
#SBATCH --partition courses
#SBATCH --account courses

module load gcc/8.4.0

gcc -o vecAdd_CPU vecAdd_CPU.c

time srun vecAdd_CPU
