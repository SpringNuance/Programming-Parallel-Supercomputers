#include <stdio.h>
#include <stdlib.h>

#include "errchk.h"
#include "reduce.cuh"

// The device kernel for finding the maximum integer in `arr`
  // EXERCISE 6: Your code here.
  // (You can reuse your implementation from task 1)

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
int reduce(const int* arr, const size_t count) {
    int num_devices;
    ERRCHK_CUDA(cudaGetDeviceCount(&num_devices));
    size_t dcount = count / num_devices;
    
    int* darr[num_devices], *d_partial_max[num_devices];
    int global_max = INT_MIN;

    // Allocate memory on each GPU and copy segments of the array
    for (int i = 0; i < num_devices; ++i) {
        cudaSetDevice(i);
        cudaMalloc(&darr[i], dcount * sizeof(int));
        cudaMemcpy(darr[i], &arr[i * dcount], dcount * sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc(&d_partial_max[i], dcount * sizeof(int));  // Enough memory for all partial results
    }

    // Perform reduction on each GPU
    int threadsPerBlock = 256;
    for (int i = 0; i < num_devices; ++i) {
        cudaSetDevice(i);
        size_t current_count = dcount;
        int blocks;

        while (current_count > 1) {
            blocks = (current_count + threadsPerBlock - 1) / threadsPerBlock;
            reduce_kernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(darr[i], current_count);

            // Prepare for the next iteration
            cudaMemcpy(d_partial_max[i], darr[i], blocks * sizeof(int), cudaMemcpyDeviceToDevice);
            current_count = blocks;
            cudaMemcpy(darr[i], d_partial_max[i], current_count * sizeof(int), cudaMemcpyDeviceToDevice);
        }
    }

    // Collect results from each GPU and perform final reduction on the host
    int* h_partial_max = (int*)malloc(num_devices * sizeof(int));
    for (int i = 0; i < num_devices; ++i) {
        cudaSetDevice(i);
        cudaMemcpy(&h_partial_max[i], darr[i], sizeof(int), cudaMemcpyDeviceToHost);
        global_max = max(global_max, h_partial_max[i]);

        cudaFree(darr[i]);
        cudaFree(d_partial_max[i]);
    }
    free(h_partial_max);

    return global_max;
}

