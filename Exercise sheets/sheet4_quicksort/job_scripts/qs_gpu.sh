#!/bin/bash -l
#### Job submission script example.
#### (Remember for Slurm #SBATCH, one # is active, two or
#### more ## is commented out)
####
#### If you have permanent Triton account (not made just for this
#### course), you can comment this out (comment = two `##`) and
#### maybe you will have shorter queue time.

####SBATCH --account courses
####SBATCH --partition courses-gpu
#### Standard parameters
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=500M
#SBATCH --nodes=1         #Use one node
#SBATCH --ntasks=1        #One task
#SBATCH --gpus-per-task=1
#SBATCH --output=prog.out



module purge
module load gcc/11.3.0 cmake/3.26.3 openmpi/4.1.5

time srun ../build/quicksort-gpu


