#!/bin/bash -l
#### Job submission script example to a CPU queue using hybrid programming model
#### If you have a temporary course account, then you must uncomment
##SBATCH --partition courses
##SBATCH --account courses
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=500M
#### Number of nodes, number of MPI processes is nodes x ntasks
#SBATCH --nodes=2
#### Number of MPI processes per node
#SBATCH --ntasks-per-node=2
#### Number of openMP threads, number of total CPUs is nodes x ntasks x cpus
#SBATCH --cpus-per-task=4
#SBATCH --output=hello.out

export OMP_PROC_BIND=TRUE
#module load gcc/8.4.0
module load openmpi/3.1.4

mpicc -fopenmp -o hello_class hello_class.c

time srun hello_class
