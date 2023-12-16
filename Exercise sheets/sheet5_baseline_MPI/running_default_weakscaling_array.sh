#!/bin/bash -l

# Compile the heat_mpi program only once
./make_heat_mpi.sh

# Create a parent directory
parent_dir="heat_default_weakscaling_array"
mkdir -p $parent_dir

# Define the grid sizes for each task
grid_sizes=(1000 2000 3000 4000 5000 6000 8000 10000 12000)
tasks_list=(2 4 6 8 10 12 16 20 24)

# Loop over the desired --ntasks-per-node values and corresponding grid sizes
for i in {0..8}; do
    # Extract the number of tasks and corresponding grid size
    tasks=${tasks_list[i]}
    grid=${grid_sizes[i]}
    
    # Create a subdirectory for this number of tasks within the parent directory
    dir_name="${parent_dir}/tasks_${tasks}_grid_${grid}"
    
    mkdir -p $dir_name
    rm -rf $dir_name/*

    # Move into the subdirectory
    cd $dir_name

    # Call the submission script with the current number of tasks and grid size
    sbatch ../../heat_default_weakscaling_array.sh $tasks $grid

    # Return to the original directory
    cd ../..

    # A short delay to prevent any potential issues with rapid directory changes
    # sleep 1
done
