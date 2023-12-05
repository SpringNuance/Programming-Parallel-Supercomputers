#!/bin/bash -l
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=500M
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --output=One_sided_1.out
#### You can comment this line out if you have a full Triton account
#SBATCH --partition courses
#SBATCH --account courses

#module load gcc/8.4.0
module load openmpi/3.1.4

mpicc -o One_sided_1 One_sided_1.c

time srun One_sided_1
