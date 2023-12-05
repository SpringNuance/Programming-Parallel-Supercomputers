#!/bin/bash -l
#### Job submission script example to a CPU queue using hybrid programming model
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=500M
#### Number of nodes, number of MPI processes is nodes x ntasks
#SBATCH --nodes=3
#### Number of MPI processes per node
#SBATCH --ntasks-per-node=4
#SBATCH --output=MPIs_MPI_3.out
#### You can comment these two lines out if you have a full Triton account
#SBATCH --partition courses
#SBATCH --account courses

#module load gcc/8.4.0
module load openmpi/3.1.4

mpicc -o MPIs_MPI_3 MPIs_MPI_3.c

time srun MPIs_MPI_3
