#!/bin/bash -l
#SBATCH --time=00:01:00
#SBATCH --mem-per-cpu=500M
#SBATCH --nodes=1
#### This program can only bu run with two cores
#SBATCH --ntasks-per-node=2
#SBATCH --output=MPI_SR_1.out
### You can comment this line out if you have a full Triton account
#SBATCH --partition courses
#SBATCH --account courses


#module load gcc/8.4.0
module load openmpi/3.1.4

mpicc -o SR1 MPI_SR_1.c

time srun SR1
