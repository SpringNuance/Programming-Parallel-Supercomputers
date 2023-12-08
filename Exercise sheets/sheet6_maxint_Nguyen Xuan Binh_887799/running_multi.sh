#!/bin/bash -l

./running_build.sh

mkdir -p multi_results 
cd multi_results
sbatch ../job-scripts/run-multi.sh