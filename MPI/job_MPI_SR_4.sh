#!/bin/bash -l
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=500M
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --output=MPI_SR_4.out

#module load gcc/8.4.0
module load openmpi/3.1.4

mpicc -o SR4 MPI_SR_4.c

time srun SR4
