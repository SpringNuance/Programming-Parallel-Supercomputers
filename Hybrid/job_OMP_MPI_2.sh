#!/bin/bash -l
#### Job submission script example to a CPU queue using hybrid programming model
#### If you have a temporary course account, then you must uncomment
##SBATCH --partition courses
##SBATCH --account courses
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=500M
#### Number of nodes, number of MPI processes is nodes x ntasks
#SBATCH --nodes=1
#### Number of MPI processes per node
#SBATCH --ntasks-per-node=4
#### Number of openMP threads, number of total CPUs is nodes x ntasks x cpus
#SBATCH --cpus-per-task=4
#SBATCH --output=OMP_MPI_2.out

export OMP_PROC_BIND=TRUE
#module load gcc/8.4.0
module load openmpi/3.1.4

mpicc -fopenmp -o OMP_MPI_2 OMP_MPI_2.c

time srun OMP_MPI_2
