#!/bin/bash -l
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=500M
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --output=Split_1.out
#### You can comment this line out if you have a full Triton account
#SBATCH --partition courses
#SBATCH --account courses

#module load gcc/8.4.0
module load openmpi/3.1.4

mpicc -o Split_1 Split_1.c

time srun Split_1
