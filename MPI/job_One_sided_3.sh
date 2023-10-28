#!/bin/bash -l
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=500M
#### Works only for two processes
#### Now forcing to run on different nodes...
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --output=One_sided_3.out

#module load gcc/8.4.0
module load openmpi/3.1.4

mpicc -o One_sided_3 One_sided_3.c

time srun One_sided_3
