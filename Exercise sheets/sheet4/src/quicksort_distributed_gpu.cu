
#include <mpi.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
void quicksort_distributed(float pivot, int start, int end, float* &data,  MPI_Comm comm)
{
	/**
        Exercise 4: Your code here
        Input:
                pivot: a pivot value based on which to split the array in to less and greater elems
                start: starting index of the range to be sorted
                end: exclusive ending index of the range to be sorted
                data: array of floats allocated on the GPU to sort in range start till end
                      the array is the same for each MPI process
                comm: the communicator of the MPI processes
        Return:
                upon return each MPI process should have their array sorted
        Task:
                to sort the array using the idea of quicksort in a stable manner using each MPI process
                a sort is stable if it maintains the relative order of elements with equal values
                truly split the work among the process i.e. don't have each process independently sort the array
		furthermore to do communication try to use CUDA aware MPI instead of first loading the data to host
		and then doing the communication 
        Hint:
		!!You should be able to combine and copy your solutions in quicksort_gpu.cu and quicksort_distributed.cu!!

**/

}
