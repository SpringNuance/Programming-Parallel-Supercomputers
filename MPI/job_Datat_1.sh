#!/bin/bash -l
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=500M
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=Datat_1.out

#module load gcc/8.4.0
module load openmpi/3.1.4

mpicc -o Datat_1 Datat_1.c

time srun Datat_1
