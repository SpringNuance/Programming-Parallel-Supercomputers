#!/bin/bash -l
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=500M
#SBATCH --nodes=1
##### This program can only be run with two cores
#SBATCH --ntasks-per-node=2
#SBATCH --output=MPI_SR_1.out
#### You can comment this line out if you have a full Triton account
#### SBATCH --partition courses
#### SBATCH --account courses


#module load gcc/8.4.0
module load openmpi/3.1.4

# mpicc is a compiler, and it outputs a binary file SR1 from C source sile MPI_SR_1
mpicc -o SR1 MPI_SR_1.c

# time is a linux command to measure the runnig time
# Unlike traditional computers, where a program is launched, there is only 1 process. 
# But in MPI paradigm, many processes are launched based on number of tasks
# They communicate via each other via MPI messages
time srun SR1
