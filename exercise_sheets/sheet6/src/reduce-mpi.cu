#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "errchk.h"
#include "reduce.cuh"

__global__ void
reduce_kernel(int* arr, const size_t count)
{
  // EXERCISE 6: Your code here.
  // (You can reuse your implementation from task 1)
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
  return -1;
}
