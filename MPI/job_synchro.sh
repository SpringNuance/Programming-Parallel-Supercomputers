#!/bin/sh
#SBATCH --account courses
#SBATCH --time=2
#SBATCH --partition courses
#SBATCH -n 2
#SBATCH --output=synchro.out

#module load gcc/8.4.0
module load openmpi/3.1.4

mpicc -o synchro.x synchro.c
srun ./synchro.x
