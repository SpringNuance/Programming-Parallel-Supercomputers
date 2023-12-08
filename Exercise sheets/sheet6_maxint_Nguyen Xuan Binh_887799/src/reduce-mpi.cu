#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "errchk.h"
#include "reduce.cuh"

__global__ void reduce_kernel(int* arr, const size_t count) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;
    int localMax = INT_MIN;

    while (idx < count) {
        localMax = max(localMax, arr[idx]);
        idx += gridSize;
    }

    extern __shared__ int sdata[];
    sdata[tid] = localMax;
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s = s / 2) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) arr[blockIdx.x] = sdata[0];
}





int reduce(const int* base_arr, const size_t base_count)
{
  int nprocs, pid;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);

  const size_t count = base_count / nprocs;
  ERRCHK(count * nprocs == base_count);

  int* arr = (int*)malloc(count * sizeof(arr[0]));
  MPI_Scatter(base_arr, count, MPI_INT, arr, count, MPI_INT, 0, MPI_COMM_WORLD);

  // EXERCISE 6: Your code here
  // Input:
  //  arr    - An array of integers assigned to the current rank (process)
  //  count  - The number of integers in an array per rank (process). You can
  //           assume that count is divisible by the number
  //           of ranks, i.e. count * nprocs = base_count
  //
  // Return: the maximum integer in base_arr. The correct result needs to be
  //         returned only by rank 0. The return values of other ranks are
  //         not checked.
  //
  // Task: the integer array (base_arr) is now distributed across the ranks.
  //       Each rank holds a subset of the array stored in `arr` residing in
  //       host memory. Your task is to first map the rank (i.e. process id or pid)
  //       to a device id. Then, you should allocate memory on that device, transfer
  //       the data, and apply the reduction with the selected device, in a similar
  //       fashion as in task 2. Finally, you should combine the results of each
  //       process/device any way you like (for example using MPI_Gather or MPI_Reduce)
  //       and return the result.
    // Map the MPI process to a GPU device
    int num_devices;
    ERRCHK_CUDA(cudaGetDeviceCount(&num_devices));
    cudaSetDevice(pid % num_devices);

    // Allocate memory on the device and copy data
    int *d_arr, *d_partial_max;
    cudaMalloc(&d_arr, count * sizeof(int));
    cudaMemcpy(d_arr, arr, count * sizeof(int), cudaMemcpyHostToDevice);

    // Perform reduction on the GPU (similar to Task 2)
    int threadsPerBlock = 256;
    size_t current_count = count;
    int blocks;
    cudaMalloc(&d_partial_max, count * sizeof(int));

    while (current_count > 1) {
        blocks = (current_count + threadsPerBlock - 1) / threadsPerBlock;
        reduce_kernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_arr, current_count);

        // Prepare for the next iteration
        cudaMemcpy(d_partial_max, d_arr, blocks * sizeof(int), cudaMemcpyDeviceToDevice);
        current_count = blocks;
        cudaMemcpy(d_arr, d_partial_max, current_count * sizeof(int), cudaMemcpyDeviceToDevice);
    }

    // Copy the result back to the host
    int local_max;
    cudaMemcpy(&local_max, d_arr, sizeof(int), cudaMemcpyDeviceToHost);

    // Clean up device memory
    cudaFree(d_arr);
    cudaFree(d_partial_max);

    // Reduce results across MPI processes
    int global_max;
    MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    free(arr);

    // Only rank 0 will have the correct global maximum
    if (pid == 0) {
        return global_max;
    }
    return -1;
}
