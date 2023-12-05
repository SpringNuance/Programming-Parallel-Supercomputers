#include <mpi.h>
#include <stdio.h>

/* Example program of one-sided communication.                                        
   Window is created on existing buffer with MPI_Win_create()                         
   passive synchronization with MPI_Win_lock()... MPI_Win_unlock is used         
   Communicating data from rank 0 from 1 using MPI_Put 

   Rank 1 is a target, and Rank 0 is the origin. */

int main(int argc, char** argv) {

/* Initialize the MPI execution environment */
MPI_Init(&argc,&argv);

// Data to be communicated, now only one integer
int buf=10;
MPI_Win win;

int rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

/* Get the communicator size - how many processes you have "ordered" */
int world_size;
MPI_Comm_size(MPI_COMM_WORLD, &world_size);

if (rank == 0) {
  /* Locally change the value of data, which you now want rank 1 to know */
   buf=100;
   printf("Rank %d has new data to send: %d\n",rank,buf);   
   /* Rank 0 will be the origin for all processes, so it does not need a window on its own; allocating an empty window */
   MPI_Win_create(NULL,0,1,MPI_INFO_NULL,MPI_COMM_WORLD,&win);
   /* Request lock of process 1 */
   MPI_Win_lock(MPI_LOCK_EXCLUSIVE,1,0,win);
   MPI_Put(&buf,1,MPI_INT,1,0,1,MPI_INT,win);
   /* Block until put succeeds */
   MPI_Win_unlock(1,win);
   /* Free the window */
   MPI_Win_free(&win);
 }
 else {
   printf("Rank %d has data %d\n",rank,buf);
   /* Rank 1 is the target process, and creates a window to expose */
   MPI_Win_create(&buf,sizeof(int),sizeof(int),MPI_INFO_NULL, MPI_COMM_WORLD, &win);
   /* No sync calls on the target process! */
   MPI_Win_free(&win);
   printf("Rank %d has data %d\n",rank,buf);   
 }

/* Finalize the MPI environment. */
MPI_Finalize();

}
