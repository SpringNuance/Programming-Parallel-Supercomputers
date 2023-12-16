#!/bin/bash -l

module load gcc/11.3.0 cmake/3.26.3 openmpi/4.1.5
mkdir -p build 
cd build
cmake .. && make -j
cd ..