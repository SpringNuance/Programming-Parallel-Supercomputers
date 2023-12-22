#!/bin/bash -l
##Triton
#SBATCH -A courses
#SBATCH -p courses
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=500M
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4

module purge
module load openmpi/4.0.5


time srun ../build/quicksort-distributed-rdma


