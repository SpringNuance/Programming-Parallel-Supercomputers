#!/bin/bash -l
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=500M
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --output=Coll_1.out

#module load gcc/8.4.0
module load openmpi/3.1.4

mpicc -o Coll_1 Coll_1.c

time srun Coll_1
