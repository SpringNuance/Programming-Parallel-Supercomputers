#include <stdio.h>
#include <stdlib.h>

#include "errchk.h"
#include "reduce.cuh"

// The device kernel for finding the maximum integer in `arr`
__global__ void
reduce_kernel(const int* in, const size_t count, int* out)
{
  // EXERCISE 6: Your code here.
  // (You can reuse your implementation from task 1)
  const int curr = threadIdx.x + blockIdx.x * blockDim.x;

  extern __shared__ int smem[];
  if (curr < count)
    smem[threadIdx.x] = in[curr];
  else
    smem[threadIdx.x] = INT_MIN;

  __syncthreads();

  int offset = blockDim.x / 2;
  while (offset > 0) {
    if (threadIdx.x < offset) {
      const int a       = smem[threadIdx.x];
      const int b       = smem[threadIdx.x + offset];
      smem[threadIdx.x] = a > b ? a : b;
    }

    offset /= 2;
    __syncthreads();
  }

  if (!threadIdx.x)
    out[blockIdx.x] = smem[threadIdx.x];
}

static void swap_ptrs(int*** a, int*** b)
{
  int** tmp = *a;
  *a = *b;
  *b = tmp;
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

  int* darr0[num_devices], *darr1[num_devices];
  cudaStream_t streams[num_devices];
  const size_t bytes = dcount * sizeof(darr0[0][0]);
  for (int i = 0; i < num_devices; ++i) {
    cudaSetDevice(i);
    cudaMalloc(&darr0[i], bytes);
    cudaMalloc(&darr1[i], bytes);
    cudaStreamCreate(&streams[i]);
  }

  int** in = darr0;
  int** out = darr1;

  for (int i = 0; i < num_devices; ++i) {
    cudaSetDevice(i);
    cudaMemcpy(in[i], &arr[i * dcount], bytes, cudaMemcpyHostToDevice);
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
  do {
    const unsigned tpb = 128;
    const size_t smem  = tpb * sizeof(in[0][0]);
    const unsigned bpg = (unsigned)ceil((double)dcount / tpb);
    for (int i = 0; i < num_devices; ++i) {
      cudaSetDevice(i);
      reduce_kernel<<<bpg, tpb, smem, streams[i]>>>(in[i], dcount, out[i]);
    }
    swap_ptrs(&in, &out);
    dcount = bpg;
  } while (dcount > 1);

  int results[num_devices];
  for (int i = 0; i < num_devices; ++i) {
    cudaSetDevice(i);
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
    cudaMemcpy(&results[i], in[i], 1 * sizeof(in[0][0]),
               cudaMemcpyDeviceToHost);
    cudaFree(darr0[i]);
    cudaFree(darr1[i]);
  }
  for (int i = 1; i < num_devices; ++i)
    results[0] = results[0] > results[i] ? results[0] : results[i];

  return results[0];
}
