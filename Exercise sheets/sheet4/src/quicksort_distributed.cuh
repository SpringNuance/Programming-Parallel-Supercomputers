#include <mpi.h>
void quicksort_distributed(float pivot, int start, int end, float* &data, MPI_Comm comm);