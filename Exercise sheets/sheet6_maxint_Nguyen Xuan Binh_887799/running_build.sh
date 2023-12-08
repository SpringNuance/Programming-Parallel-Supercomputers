#!/bin/bash -l

module load gcc cuda cmake openmpi
mkdir -p build 
cd build
cmake .. && make -j
cd .. 