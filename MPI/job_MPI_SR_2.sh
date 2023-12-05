#!/bin/bash -l
#SBATCH --time=00:01:00
#SBATCH --mem-per-cpu=500M
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=MPI_SR_2.out
### You can comment this line out if you have a full Triton account
#SBATCH --partition courses
#SBATCH --account courses

#module load gcc/8.4.0
module load openmpi/3.1.4

mpicc -o SR2 MPI_SR_2.c

time srun SR2
