#!/bin/bash -l

./make_heat_mpi.sh

mkdir -p heat_bottle_timestep
cd heat_bottle_timestep
sbatch ../heat_bottle_timestep.sh