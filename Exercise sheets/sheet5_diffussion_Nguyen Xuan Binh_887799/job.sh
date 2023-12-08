#!/bin/bash -l
#### Job submission script example to a CPU queue using hybrid programming
#### model
#### (Remember for Slurm #SBATCH, one # is active, two or
#### more ## is commented out)
####
#### If you have permanent Triton account (not made just for this
#### course), you can comment this out (comment = two `##`) and
#### maybe you will have shorter queue time.
#SBATCH --partition courses
#SBATCH --account courses
#### General resource parameters:
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=500M
#### Number of nodes, number of MPI processes is nodes x ntasks
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
####Specify output file, otherwise slurm-<jobid>.out generated
##SBATCH --output=DE.out
####Special resource allocation, do not use unless instructed
##SBATCH --reservation pps_course_session1

module purge   # unload all current modules
module load openmpi/3.1.4

time srun heat_mpi bottle.dat 100000
