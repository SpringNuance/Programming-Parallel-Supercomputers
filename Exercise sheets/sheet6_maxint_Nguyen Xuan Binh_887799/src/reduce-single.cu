#include <stdio.h>
#include <stdlib.h>

#include "errchk.h"
#include "reduce.cuh"

// The device kernel for finding the maximum integer in `arr`

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


// The host function for calling the reduction kernel

int reduce(const int* arr, const size_t initial_count) {
    int *d_arr, *d_partial_max;
    size_t size = initial_count * sizeof(int);

    cudaMalloc(&d_arr, size);
    cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks;
    size_t current_count = initial_count;

    // Allocate memory for partial results on the GPU
    cudaMalloc(&d_partial_max, size);  // Allocate enough memory for all partial results

    while (current_count > 1) {
        blocks = (current_count + threadsPerBlock - 1) / threadsPerBlock;

        // Call the kernel
        reduce_kernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_arr, current_count);

        // Copy the partial results back to d_partial_max
        cudaMemcpy(d_partial_max, d_arr, blocks * sizeof(int), cudaMemcpyDeviceToDevice);

        // Prepare for the next iteration
        current_count = blocks;
        cudaMemcpy(d_arr, d_partial_max, current_count * sizeof(int), cudaMemcpyDeviceToDevice);
    }

    // Copy the final result (single maximum value) back to the host
    int global_max;
    cudaMemcpy(&global_max, d_arr, sizeof(int), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_arr);
    cudaFree(d_partial_max);

    return global_max;
}
