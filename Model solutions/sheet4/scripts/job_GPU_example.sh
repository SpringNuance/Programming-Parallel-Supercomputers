#!/bin/bash -l
#### For multi-GPU jobs, in --gres:...:N, choose N>1. Single GPU, N==1.
#### (Remember for Slurm #SBATCH, one # is active, two or
#### more ## is commented out)
####
#### If you have permanent Triton account (not made just for this
#### course), you can comment this out (comment = two `##`) and
#### maybe you will have shorter queue time.
#SBATCH -p courses
#SBATCH -A courses
#### Standard parameters
#SBATCH --gres=gpu:teslap100:4
#SBATCH --time=00:05:00
#### For MPI
####One node, four GPUs
#SBATCH --nodes 1
#SBATCH --ntasks 4 #nodes x N
#SBATCH --output=GPUcode.out

module purge   # unload all current modules
#module load openmpi
module load gcc cuda 

nvcc -o GPUcode GPUcode.cu

time srun GPUcode
