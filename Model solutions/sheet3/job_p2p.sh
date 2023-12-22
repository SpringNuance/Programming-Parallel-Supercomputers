#!/bin/bash -l
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=500M
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --output=p2p.out
####For temporary user accounts, comment out this line.
#SBATCH -p courses
#SBATCH -A courses
####

module load gcc/9.2.0
module load openmpi/3.1.4

mpicc -o p2p point2point.c

time srun p2p 2000 100 1
