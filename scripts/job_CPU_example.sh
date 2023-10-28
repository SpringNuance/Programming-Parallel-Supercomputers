#!/bin/bash -l
#### Job submission script example to a CPU queue.
#### (Remember for Slurm #SBATCH, one # is active, two or
#### more ## is commented out)
####
#### If you have permanent Triton account (not made just for this
#### course), you can comment this out (comment = two `##`) and
#### maybe you will have shorter queue time.
#SBATCH --account courses
#SBATCH --partition courses
#### Standard parameters
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=500M
#### For a small MPI job:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2 #No of tasks in one node, cf. --ntasks
#### For a large MPI job:
##SBATCH --exclusive       #Allocate whole node
##SBATCH --constraint=XXX  #Require certain type of nodes
##SBATCH --nodes=2         #Use two nodes
##SBATCH --ntasks=YYY      #Total amount of tasks: YYY=nodes*n_CPUs_on_XXX
####End of large MPI job.
##SBATCH --output=prog.out #You can optionally name output, otherwise slurm.jobid

module purge   # unload all current modules
#Load MPI module
module load openmpi

mpicc -o prog prog.c

time srun prog
