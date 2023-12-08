#!/bin/bash -l

./running_build.sh

mkdir -p single_results 
cd single_results
sbatch ../job-scripts/run-single.sh