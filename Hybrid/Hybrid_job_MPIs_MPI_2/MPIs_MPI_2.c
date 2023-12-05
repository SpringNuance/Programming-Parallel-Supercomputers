#include <mpi.h>
#include <stdio.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

// Simple extension of MPIs_MPI_1.c, which also stores some silly
// data to the shared memory window, and then demonstrates the 
// referencing scheme to neighbor's shares of the window.

int main(int argc, char** argv) {

  MPI_Init(NULL, NULL);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  MPI_Info info=MPI_INFO_NULL;
  MPI_Comm sharedcomm;
  // Split the communicator into the number of nodes
  MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,0,info,&sharedcomm);
  int new_procno, new_nprocs;
  MPI_Comm_size(sharedcomm,&new_nprocs);
  MPI_Comm_rank(sharedcomm,&new_procno);
  // Comment out to see if split operation was successful
  // printf("Old Comm: rank %d nprocs %d\n",world_rank,world_size);
  // printf("New Comm: rank %d nprocs %d\n",new_procno,new_nprocs);
  
  // Allocate shared memory window; now contiguous data
  // window_size is the LOCAL window size; dispp it the displacement unit.
  MPI_Win node_window;
  MPI_Aint window_size; double *window_data; int dispp=sizeof(double);
  window_size = sizeof(double);
  // Make shared mem allocation
  MPI_Win_allocate_shared(window_size,dispp,MPI_INFO_NULL,sharedcomm,&window_data,&node_window);

  // Fence to start a local store epoch
  MPI_Win_fence(0,node_window);
  // Fill values into the local portion
  window_data[0]=1.0*new_procno+1.0*world_rank;
  // Fence to start a remote load epoch
  MPI_Win_fence(0,node_window);
  // No process that tries to peek to the left can have rank 0; it would then
  // try to go outside the window.
  if (new_procno > 0) printf("Old %d New %d Left neighbor's value =%lf\n",world_rank,new_procno,window_data[-1]);
  // No process that tries to peek right can have highest rank; it would also
  // try to go outside the window.
  if (new_procno < new_nprocs-1) printf("Old %d, New %d Right neighbor's value=%lf\n",world_rank,new_procno,window_data[1]);

  /* Finalize the MPI environment. */
  MPI_Finalize();
}

