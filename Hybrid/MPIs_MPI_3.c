#include <mpi.h>
#include <stdio.h>

// Program that communicates data over nodes with standard MPI
// and through shared memory inside a node.
// The processes exchange their ranks to the right neighbor with
// periodic bcs.

int main(int argc, char** argv) {

  MPI_Init(NULL, NULL);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  MPI_Info info=MPI_INFO_NULL;
  MPI_Status status;
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
  double update_data, recv_data;
  window_size = sizeof(double);
  MPI_Win_allocate_shared(window_size,dispp,MPI_INFO_NULL,sharedcomm,&window_data,&node_window);

  // We should first accumulate data from inter-node processes
  if (new_procno == new_nprocs-1 && world_rank != world_size-1) {
    update_data=1.0*world_rank;
    MPI_Send(&update_data,1,MPI_DOUBLE,world_rank+1,0,MPI_COMM_WORLD);
    }

  // Boundary communication
  if (world_rank == world_size-1) {
      update_data=1.0*world_rank;
      MPI_Send(&update_data,1,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
    }

  if (new_procno == 0 && world_rank != 0) {
    MPI_Recv(&recv_data,1,MPI_DOUBLE,world_rank-1,0,MPI_COMM_WORLD,&status);
  }

  // Boundary communication
  if (world_rank == 0) {
    MPI_Recv(&recv_data,1,MPI_DOUBLE,world_size-1,0,MPI_COMM_WORLD,&status);
  }
  // Storing to shared mem starts.
  // The data to put into right neighbor's shared mem zone
  // is the process's global rank.
  MPI_Win_fence(0,node_window); // Starting remote store epoch               
  // First store the data that was acquired through inter-node comms
  if (new_procno == 0 || world_rank == 0 || world_rank==world_size-1) window_data[0]=recv_data;
  if (new_procno < new_nprocs-1) {
    window_data[1]=1.0*world_rank; // Remote intranode stores 
  }

  MPI_Win_fence(0,node_window); // Starting local load epoch to check what we have             
  printf("Old %d New %d has the value of %lf\n",world_rank,new_procno,window_data[0]);

  MPI_Win_free(&node_window);

  /* Finalize the MPI environment. */
  MPI_Finalize();
}

