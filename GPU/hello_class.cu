#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>


int
main(void)
{
// Check how many MPIs we have
  MPI_Init(NULL, NULL);
  int nprocs, pid;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  printf("Number of MPI processes %d, Hello from rank: %d\n", nprocs,pid);


// Check how many GPUs we have
  int num_devices;
  cudaGetDeviceCount(&num_devices);
  printf("Number of devices %d rank %d sees\n", num_devices,pid);

}

