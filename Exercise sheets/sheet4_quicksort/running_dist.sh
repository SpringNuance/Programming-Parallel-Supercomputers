#!/bin/bash -l

./running_build.sh

mkdir -p qs_dist_job
cd qs_dist_job
sbatch ../job_scripts/qs_dist.sh