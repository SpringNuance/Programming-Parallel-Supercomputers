#!/bin/bash
#### I guess you know how this works by now... adjust as needed.
#SBATCH --gres=gpu:teslap100:4
#SBATCH -t 00:00:59
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --partition courses
#SBATCH --account courses

module load gcc cuda cmake openmpi

srun ./reduce-multi reduce-multi.result