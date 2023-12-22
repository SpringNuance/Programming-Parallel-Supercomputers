#!/bin/bash -l

##Note this will fail if CUDA aware MPI does not work

##Triton
#SBATCH -A courses
#SBATCH -p courses-gpu
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=500M
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2

module purge
export OMPI_MCA_opal_warn_on_missing_libcuda=0

module load gcc/11.3.0 cmake/3.26.3 openmpi/4.1.5

time srun ../build/quicksort-distributed-gpu



