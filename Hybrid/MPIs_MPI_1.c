#include <mpi.h>
#include <stdio.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

// Program that splits the communicator into n-code blocks, and
// reserves a contiguous shared memory window for each node's procs,
// and prints the address of each local window (from the beginning 
// of the window)

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

  // Check what was accomplished
  for (int p=1; p<new_nprocs; p++) {
    MPI_Aint window_sizep; int distp; int windowp_unit; double *winp_addr;
    MPI_Win_shared_query(node_window,p,&window_sizep,&dispp,&winp_addr );
    distp = (size_t)winp_addr-(size_t)window_data;
    if (new_procno==0)
      printf("Distance %d to zero: %ld\n",p,(long)distp);
  }

  MPI_Win_fence(0,node_window);
  window_data[0]=1.0*new_procno+1.0*world_rank;
  MPI_Win_fence(0,node_window);
  if (new_procno > 0) printf("Old %d New %d Left neighbor's value =%lf\n",world_rank,new_procno,window_data[-1]);
  if (new_procno < new_nprocs-1) printf("Old %d, New %d Right neighbor's value=%lf\n",world_rank,new_procno,window_data[1]);
  printf("Old %d New %d My own is %lf\n",world_rank,new_procno,window_data[0]);

  /* Finalize the MPI environment. */
  MPI_Finalize();
}

