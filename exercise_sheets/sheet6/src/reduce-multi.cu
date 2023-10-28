#include <stdio.h>
#include <stdlib.h>

#include "errchk.h"
#include "reduce.cuh"

// The device kernel for finding the maximum integer in `arr`
__global__ void
reduce_kernel(int* arr, const size_t count)
{
  // EXERCISE 6: Your code here.
  // (You can reuse your implementation from task 1)
}

// The host function for calling the reduction kernel
int
reduce(const int* arr, const size_t count)
{
  // Do not modify: helper code for distributing `arr` to multiple GPUs
  int num_devices;
  ERRCHK_CUDA(cudaGetDeviceCount(&num_devices));
  size_t dcount = count / num_devices;
  ERRCHK(dcount * num_devices ==
         count); // Require count divisible with num_devices

  int* darr[num_devices];
  cudaStream_t streams[num_devices];
  const size_t bytes = dcount * sizeof(darr[0][0]);
  for (int i = 0; i < num_devices; ++i) {
    cudaSetDevice(i);
    cudaMalloc(&darr[i], bytes);
    cudaMemcpy(darr[i], &arr[i * dcount], bytes, cudaMemcpyHostToDevice);
    cudaStreamCreate(&streams[i]);
  }

  // EXERCISE 6: Your code here
  // Input:
  //  darr[num_devices] - An array of integers distributed among the available devices.
  //                      For example, darr[0] is the array resident on device #0.
  //  dcount            - The number of integers in an array per device. You can
  //                      assume that the initial count is divisible by the number
  //                      of devices, i.e. dcount * num_devices = count
  //
  // Return: the maximum integer across all device arrays in darr[]
  //
  // Task: the data is now stored in an array of array pointers (see darr explanation above).
  //       Your task is to apply the reduction function implemented in task 1
  //       on all devices in parallel, combine their results, and return the
  //       final result. You can do the final reduction step on the host, but
  //       otherwise use the GPU resources relatively efficiently.
  //
  // Feel free to use CUDA streams to improve the efficiency of your code
  // (not required for full points)
  return -1;
}
