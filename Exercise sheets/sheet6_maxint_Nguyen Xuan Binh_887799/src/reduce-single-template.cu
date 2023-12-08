#include <stdio.h>
#include <stdlib.h>

#include "errchk.h"
#include "reduce.cuh"

// The device kernel for finding the maximum integer in `arr`
__global__ void
reduce_kernel(int* arr, const size_t count)
{
  // EXERCISE 6: Your code here
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
  return -1;
}
