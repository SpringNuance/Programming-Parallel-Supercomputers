#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "errchk.h"
#include "reduce.cuh"

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

static void swap_ptrs(int** a, int** b)
{
  int* tmp = *a;
  *a = *b;
  *b = tmp;
}

int
reduce_device(const int* arr, const size_t initial_count)
{
  // EXERCISE 6: Your code here.
  // (You can reuse your implementation from task 2)

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

int
reduce(const int* base_arr, const size_t base_count)
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
  int devices_per_node;
  ERRCHK_CUDA(cudaGetDeviceCount(&devices_per_node));

  cudaSetDevice(pid % devices_per_node);
  int candidate = reduce_device(arr, count);

  /*
  int results[nprocs];
  MPI_Gather(&candidate, 1, MPI_INT, results, 1, MPI_INT, 0, MPI_COMM_WORLD);
  for (int i = 1; i < nprocs; ++i)
    results[0] = results[0] > results[i] ? results[0] : results[i];

  free(arr);
  return results[0];
  */
  int result;
  MPI_Reduce(&candidate, &result, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

  free(arr);
  return result;  
}
