#include <stdio.h>
#include <stdlib.h>

#include "errchk.h"
#include "reduce.cuh"

// The device kernel for finding the maximum integer in `arr`
__global__ void
reduce_kernel(const int* in, const size_t count, int* out)
{
  // EXERCISE 6: Your code here
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

static void swap_ptrs(int** a, int** b)
{
  int* tmp = *a;
  *a = *b;
  *b = tmp;
}

// The host function for calling the reduction kernel
int
reduce(const int* arr, const size_t initial_count)
{
  // EXERCISE 6: Your code here
  // Input:
  //  arr           - An array of integers
  //  initial_count - The number of integers in `arr`
  //
  // Return: the maximum integer in `arr`
  //
  // Task: allocate memory on the GPU, transfer `arr` into
  // the allocated space, and apply `reduce_kernel` iteratively on it
  // to find the maximum integer. Finally, move the result back to host
  // memory and return it.

  // Check that we are running on a single GPU
  int num_devices;
  ERRCHK_CUDA(cudaGetDeviceCount(&num_devices));
  ERRCHK(num_devices == 1);

  // Allocate device memory and copy data from host to device
  const size_t bytes = initial_count * sizeof(arr[0]);
  int *darr0, *darr1;
  cudaMalloc(&darr0, bytes);
  cudaMalloc(&darr1, bytes);

  int* in = darr0;
  int* out = darr1;
  cudaMemcpy(in, arr, bytes, cudaMemcpyHostToDevice);

  const unsigned tpb = 128;
  const size_t smem  = tpb * sizeof(arr[0]);

  size_t count = initial_count;
  do {
    const unsigned bpg = (unsigned)ceil((double)count / tpb);
    reduce_kernel<<<bpg, tpb, smem>>>(in, count, out);
    ERRCHK_CUDA_KERNEL();

    swap_ptrs(&in, &out);
    count = bpg;
  } while (count > 1);

  int result;
  cudaMemcpy(&result, in, sizeof(in[0]), cudaMemcpyDeviceToHost);

  cudaFree(darr0);
  cudaFree(darr1);
  return result;
}
