#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {

  // Example code of using non-blocking Isend and Irecv
  // Similar to MPI_SR_1.c 
  // The new thing here is that in between send and wait you could do something else, as the communication is not blocking.
  int my_id, your_id, bufsize=10000;
  int sendbuf[bufsize],recvbuf[bufsize];
  MPI_Status status;
  MPI_Request srequest, rrequest;

  /* Initialize the MPI execution environment */
  MPI_Init(&argc,&argv);

  /* Ranks */
  MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

  your_id = 1 - my_id;

  sendbuf[5]=10*my_id-100;

  /* First post a receive */
  printf("Rank %d issuing ireceive for a message from %d...",my_id,your_id);  
  MPI_Irecv(&recvbuf,bufsize,MPI_INT,your_id,0,MPI_COMM_WORLD,&rrequest);
  printf("... rank %d returned from ireceive ...",my_id);  

  printf("Rank %d Initiating send to rank %d ...",my_id,your_id);
  /* Then post a send...*/
  MPI_Isend(&sendbuf,bufsize,MPI_INT,your_id,0,MPI_COMM_WORLD,&srequest); 
  printf("... rank %d returned from send ... ",my_id);
  MPI_Wait(&srequest,MPI_STATUS_IGNORE);
  printf("... rank %d completed waiting of send request!\n ",my_id);

  MPI_Wait(&rrequest,MPI_STATUS_IGNORE);
  printf("...rank %d completed the waiting of ireceive: received this %d from  %d\n",my_id,recvbuf[5],your_id);

  /* Finalize the MPI environment. */
  MPI_Finalize();
}

