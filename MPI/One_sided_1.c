#include <mpi.h>
#include <stdio.h>

/* Example program of one-sided communication.                                        
   Window is created on existing buffer with MPI_Win_create()                         
   Active synchronization with MPI_Win_fence()is used                                 
   Communicating the rank of the left neighbor to the right neighbor 
   using MPI_Put */

int main(int argc, char** argv) {

/* Initialize the MPI execution environment */
MPI_Init(&argc,&argv);

// Data to be communicate, now only one integer
int data;
MPI_Win window;

int world_rank;
MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

/* Get the communicator size - how many processes you have "ordered" */
int world_size;
MPI_Comm_size(MPI_COMM_WORLD, &world_size);

// Create window and assume that we do not care about the info handle
MPI_Win_create(&data, sizeof(int), sizeof(int), MPI_INFO_NULL,
                 MPI_COMM_WORLD, &window);

// Use active synchronization with fences; assertion 0 adequate here.
MPI_Win_fence(0, window); 

// Purpose: send around information about left neighbor to the right neighbor
if (world_rank < world_size-1) {
// Scenario: you are the origin. Next highest rank process is the target.
// Put your data(rank) to the window exposed by the next in rank processor 
// (equivalent of "sending to the right")
  printf("Rank %d here and sending %d to %d\n",world_rank,world_rank,world_rank+1);
  MPI_Put(&world_rank, 1, MPI_INT, world_rank+1, 0, 1, MPI_INT, window);
}
else {
  //The Last rank has to send to the 0th rank
  MPI_Put(&world_rank, 1, MPI_INT, 0, 0 , 1, MPI_INT, window);
  printf("Last rank, %d, here and sending %d to rank 0\n",world_rank,world_rank);
}

MPI_Win_fence(0, window);

// Free the window
MPI_Win_free(&window);

 printf("I'm rank %d and my neighbor %d put data %d to my window\n", world_rank, data,data);

/* Finalize the MPI environment. */
MPI_Finalize();

}
