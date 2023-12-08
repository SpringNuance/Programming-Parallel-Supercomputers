#!/bin/bash
#### I guess you know how this works by now... adjust as needed.
#SBATCH --gpus-per-node 4
#SBATCH -t 00:00:59
#SBATCH --ntasks-per-node 4
#SBATCH --nodes 2
#SBATCH --output mpi_output.txt
####SBATCH --partition courses-gpu
####SBATCH --account courses

module purge
export OMPI_MCA_opal_warn_on_missing_libcuda=0
module load gcc/11.3.0 cmake/3.26.3 openmpi/4.1.5

srun ../build/reduce-mpi reduce-mpi.result
