#!/bin/bash
#### I guess you know how this works by now... adjust as needed.
#SBATCH --gpus-per-node 4
#SBATCH -t 00:00:59
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --partition courses-gpu
#SBATCH --account courses

module purge
module load gcc/11.3.0 cmake/3.26.3 openmpi/4.1.5

srun ../build/reduce-multi reduce-multi.result
