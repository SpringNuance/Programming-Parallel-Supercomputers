#!/bin/bash -l

module load gcc cuda cmake openmpi
# rm -rf build
mkdir -p build 
cd build
cmake .. && make -j
cd .. 