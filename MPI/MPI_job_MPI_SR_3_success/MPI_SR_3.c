#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {

  /* Blocking send and receive in between *two* processes */
  /* Issue send and receive in pairs; does the order matter? */

  int my_id, your_id, bufsize=65536;
  int sendbuf[bufsize],recvbuf[bufsize];
  MPI_Status status;

  /* Initialize the MPI execution environment */
  MPI_Init(&argc,&argv);

  /* Ranks */
  MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

  your_id = 1 - my_id;
  printf("Rank %d here, hello!\n",my_id);

  if (my_id == 0) {
  /* Receive */
  printf("Sending %d integers to and receiving from rank %d\n",sizeof(sendbuf)/sizeof(int),your_id);
  MPI_Send(&sendbuf,bufsize,MPI_INT,your_id,0,MPI_COMM_WORLD);  
  MPI_Recv(&recvbuf,bufsize,MPI_INT,your_id,0,MPI_COMM_WORLD,&status);
  printf("Rank %d done\n",my_id);
  } else {
  /* Switch the order of sends and receives here. Do you see a difference? Why?*/
  // MPI_Send(&sendbuf,bufsize,MPI_INT,your_id,0,MPI_COMM_WORLD);  
  // MPI_Recv(&recvbuf,bufsize,MPI_INT,your_id,0,MPI_COMM_WORLD,&status);  
  MPI_Recv(&recvbuf,bufsize,MPI_INT,your_id,0,MPI_COMM_WORLD,&status);
  MPI_Send(&sendbuf,bufsize,MPI_INT,your_id,0,MPI_COMM_WORLD);
  printf("Rank %d done\n",my_id);
  }

  /* Finalize the MPI environment. */
  MPI_Finalize();
}

