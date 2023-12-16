#!/bin/bash -l
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=500M
#SBATCH --nodes=1
##### This program can only be run with two cores
#SBATCH --ntasks-per-node=9
#SBATCH --cpus-per-task=1
#SBATCH --output=prog.out
#### You can comment this line out if you have a full Triton account
#### SBATCH --partition courses
#### SBATCH --account courses

module load openmpi/3.1.4

mpicc -lm -g -o advec_wave_2D_skel advec_wave_2D_skel.c

# 5 arguments: nprocx, nprox, domain_nx, domain_ny, iterations
time srun advec_wave_2D_skel 3 3 6 6 10  