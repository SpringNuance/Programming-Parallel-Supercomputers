#!/bin/bash -l
#SBATCH --time=00:01:00
#SBATCH --mem-per-cpu=500M
#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=P2P_samenode_%a.out
#SBATCH --array=1-3
#### You can comment this line out if you have a full Triton account
#### SBATCH --partition courses
#### SBATCH --account courses

#module load gcc/8.4.0
module load openmpi/3.1.4

# Compile the program for each array task with a unique name
mpicc -o P2P_samenode_$SLURM_ARRAY_TASK_ID P2P.c

# Run the program with the task-specific executable
time srun P2P_samenode_$SLURM_ARRAY_TASK_ID
