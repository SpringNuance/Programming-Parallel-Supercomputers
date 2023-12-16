#!/bin/bash -l

./make_heat_mpi.sh

mkdir -p heat_bottle
cd heat_bottle
sbatch ../heat_bottle.sh