#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>  // Include for malloc and free

int main(int argc, char** argv) {

  /* Blocking send and receive in between *two* processes */
  /* First send and receive */
  /* Start with bufsize 1, then keep on increasing it */
  int my_id, your_id, bufsize=100000;    // runs through
  //int my_id, your_id, bufsize=65536;    // hangs already
  int sendbuf[bufsize],recvbuf[bufsize];
  MPI_Status status;

  printf("Initializing MPI...\n");
  printf("size of int= %d \n",sizeof(my_id));

  /* Initialize the MPI execution environment */
  MPI_Init(&argc,&argv);

  /* Ranks */
  MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

  your_id = 1 - my_id;
  printf("Rank %d here, hello!\n",my_id);

  printf("Initiating send to rank %d\n",your_id);
  /* First send...*/

  // Allocate the buffer for MPI_Bsend
  int mpi_buffer_size = bufsize * sizeof(int) + MPI_BSEND_OVERHEAD;
  void* mpi_buffer = malloc(mpi_buffer_size);
  MPI_Buffer_attach(mpi_buffer, mpi_buffer_size);

  MPI_Bsend(&sendbuf,bufsize,MPI_INT,your_id,0,MPI_COMM_WORLD); 

  printf("Waiting to receive from rank %d\n",your_id);  
  /* Then receive */
  MPI_Recv(&recvbuf,bufsize,MPI_INT,your_id,0,MPI_COMM_WORLD,&status);

  printf("Receive of %d numbers finished from %d\n",sizeof(recvbuf)/4,your_id);

  // Detach the buffer
  MPI_Buffer_detach(&mpi_buffer, &mpi_buffer_size);
  free(mpi_buffer);

  /* Finalize the MPI environment. */
  MPI_Finalize();
}

