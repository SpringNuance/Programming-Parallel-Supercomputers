#!/bin/bash -l
#### Job submission script example to a CPU queue using hybrid programming model
#### (Remember for Slurm #SBATCH, one # is active, two or
#### more ## is commented out)
####
#### If you have permanent Triton account (not made just for this
#### course), you can comment this out (comment = two `##`) and
#### maybe you will have shorter queue time.
#SBATCH -A courses
#SBATCH -p courses
#### Standard parameters
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=500M
#### Number of nodes
#SBATCH --nodes=1
#### Number of MPI processes per node
#SBATCH --ntasks-per-node=2
#### Number of openMP threads per MPI process
#SBATCH --cpus-per-task=4
#SBATCH --output=prog.out

export OMP_PROC_BIND=TRUE
module purge   # unload all current modules
#module load gcc/8.4.0
module load openmpi/3.1.4

mpicc -fopenmp -o prog prog.c

time srun prog
