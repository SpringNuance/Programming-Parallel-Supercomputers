#!/bin/bash -l
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=500M
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --output=Comm_1.out

#module load gcc/8.4.0
module load openmpi/3.1.4

mpicc -o Comm_1 Comm_1.c

time srun Comm_1
