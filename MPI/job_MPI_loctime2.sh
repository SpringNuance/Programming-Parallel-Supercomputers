#!/bin/bash -l 
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=500M
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --output=loctime.out
#### You can comment this line out if you have a full Triton account
#SBATCH --partition courses
#SBATCH --account courses

#module load gcc/8.4.0
module load mpich/3.4.2

mpicc -o loctime MPI_loctime.c

time srun loctime
