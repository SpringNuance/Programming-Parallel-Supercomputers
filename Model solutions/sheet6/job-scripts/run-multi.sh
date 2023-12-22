#!/bin/bash
#SBATCH --gres=gpu:teslap100:4
#SBATCH -t 00:00:59
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p courses
#SBATCH -A courses

module load gcc cuda cmake openmpi

srun ./reduce-multi reduce-multi.result