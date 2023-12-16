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

int reduce(const int* base_arr, const size_t base_count) {
    int nprocs, pid;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    const size_t count_per_process = base_count / nprocs;
    ERRCHK(count_per_process * nprocs == base_count);

    int* local_arr = (int*)malloc(count_per_process * sizeof(int));
    MPI_Scatter(base_arr, count_per_process, MPI_INT, local_arr, count_per_process, MPI_INT, 0, MPI_COMM_WORLD);

    int num_devices;
    cudaGetDeviceCount(&num_devices);
    cudaSetDevice(pid % num_devices);

    int *d_arr, local_max = INT_MIN;
    cudaMalloc(&d_arr, count_per_process * sizeof(int));
    cudaMemcpy(d_arr, local_arr, count_per_process * sizeof(int), cudaMemcpyHostToDevice);

    // Perform reduction on the GPU
    int threadsPerBlock = 256;
    int blocks = (count_per_process + threadsPerBlock - 1) / threadsPerBlock;
    reduce_kernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_arr, count_per_process);
    cudaDeviceSynchronize();

    // Assuming the reduce kernel leaves the result in the first element of d_arr
    cudaMemcpy(&local_max, d_arr, sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_arr);
    free(local_arr);

    int global_max;
    MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (pid == 0) {
        return global_max;
    }
    return -1;
}
