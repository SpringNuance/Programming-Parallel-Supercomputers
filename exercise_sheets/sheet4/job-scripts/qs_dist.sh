#!/bin/bash -l
#### Job submission script example.
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
#### For a large MPI job:
#SBATCH --nodes=2         #Use two nodes
#SBATCH --ntasks=8        #Eight tasks
####End of large MPI job.
#SBATCH --output=prog.out

module purge
module load openmpi/3.1.4


srun ./build/quicksort-distributed


