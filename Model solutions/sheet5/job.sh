#!/bin/bash -l
#### Job submission script example to a CPU queue using hybrid programming model
#### If you have a temporary course account, then you must uncomment
##SBATCH --partition=courses
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=500M
#### Number of nodes
#SBATCH --nodes=1
#### Number of MPI processes per node
#SBATCH --ntasks-per-node=1
####Number of openMP threads per MPI process
#SBATCH --cpus-per-task=4
#SBATCH --output=DE.out
#SBATCH --constrain=hsw

export OMP_PROC_BIND=TRUE
module load gcc/9.2.0
module load openmpi/3.1.4

rm HEAT_RESTART.dat

time srun heat_mpi 4000 4000 1000
