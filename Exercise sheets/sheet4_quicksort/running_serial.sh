#!/bin/bash -l

./running_build.sh

mkdir -p qs_serial_job
cd qs_serial_job
sbatch ../job_scripts/qs_serial.sh