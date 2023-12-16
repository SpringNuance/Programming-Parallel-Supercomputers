#!/bin/bash -l

./running_build.sh

mkdir -p mpi_results 
cd mpi_results

sbatch ../job-scripts/run-mpi.sh