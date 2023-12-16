#!/bin/bash -l

./make_heat_mpi.sh

mkdir -p heat_default
cd heat_default
sbatch ../heat_default.sh