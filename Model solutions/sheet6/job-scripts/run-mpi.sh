#!/bin/bash
#SBATCH --gres=gpu:teslap100:4
#SBATCH -t 00:00:59
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH -p courses
#SBATCH -A courses

module load gcc cuda cmake openmpi

srun ./reduce-mpi reduce-mpi.result