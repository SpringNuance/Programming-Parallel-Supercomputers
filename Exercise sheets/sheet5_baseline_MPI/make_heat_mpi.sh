#!/bin/bash -l

module load gcc/11.3.0 openmpi/4.1.5
make clean
make
