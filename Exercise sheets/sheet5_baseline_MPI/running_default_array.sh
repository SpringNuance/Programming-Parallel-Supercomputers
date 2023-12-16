#!/bin/bash -l

# Compile the heat_mpi program only once
./make_heat_mpi.sh

# Create a parent directory
parent_dir="heat_default_array"
mkdir -p $parent_dir

# Loop over the desired --ntasks-per-node values
for tasks in 2 4 6 8 10 12 16 20 24; do
    # Create a subdirectory for this number of tasks within the parent directory
    dir_name="${parent_dir}/tasks_${tasks}"
    
    mkdir -p $dir_name
    rm -rf ${dir_name}/*

    # Move into the subdirectory
    cd $dir_name

    # Call the submission script with the current number of tasks
    sbatch ../../heat_default_array.sh ${tasks}

    # Return to the original directory
    cd ../..

    # A short delay to prevent any potential issues with rapid directory changes
    # sleep 1
done
