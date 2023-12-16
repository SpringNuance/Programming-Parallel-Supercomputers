#!/bin/bash -l

./make_heat_mpi.sh

mkdir -p heat_default_timestep_dim
cd heat_default_timestep_dim
sbatch ../heat_default_timestep_dim.sh