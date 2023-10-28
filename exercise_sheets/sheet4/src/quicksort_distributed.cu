
#include <mpi.h>
void quicksort_distributed(float pivot, int start, int end, float* &data,  MPI_Comm comm)
{
/**
        Exercise 4: Your code here
        Input:
                pivot: a pivot value based on which to split the array in to less and greater elems
                start: starting index of the range to be sorted
                end: exclusive ending index of the range to be sorted
                data: array of floats to sort in range start till end
		      the array is the same for each MPI process
		comm: the communicator of the MPI processes
        Return:
                upon return each MPI process should have their array sorted
        Task:   
                to sort the array using the idea of quicksort in a stable manner using each MPI process
                a sort is stable if it maintains the relative order of elements with equal values
		truly split the work among the process i.e. don't have each process independently sort the array
	Hint:
		why is the communicator given to you?
		maybe it could be a good MPI_Comm_split?
		could you reuse your quicksort implementation for the base case of a single process?

**/
}
