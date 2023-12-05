#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {

  /* Blocking send and receive in between *two* processes */
  /* Issue send and receive in pairsi; does the order matter? */

  int my_id, your_id, bufsize=10000;
  int sendbuf[bufsize],recvbuf[bufsize];
  MPI_Status status;
  int *v,flag;
  double start,end,tickres;  

  /* Initialize the MPI execution environment */
  MPI_Init(&argc,&argv);

  /* Ranks */
  MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

  /* Check the time resolution of MPI_Wtime in seconds */
  tickres=MPI_Wtick();

  /* Check for global timing */
  MPI_Comm_get_attr(MPI_COMM_WORLD,MPI_WTIME_IS_GLOBAL,&v,&flag);

  /* Does Wtime itself have overhead ? */
  start=MPI_Wtime();
  end=MPI_Wtime();

  if (my_id == 0) {
    printf("Does key value MPI_WTIME_IS_GLOBAL exist (1 for yes)?: %d\n",flag);
    printf("Is it set (1 for yes)?: %d\n",*v);
    printf("Wtime resolution is %12.5e seconds\n",tickres);
    printf("Wtime overhead is roughtly %12.5e seconds\n",end-start);
   }  

  your_id = 1 - my_id;

  /* What are we actually timing here? */
  if (my_id == 0) {
  start=MPI_Wtime();
  MPI_Send(&sendbuf,bufsize,MPI_INT,your_id,0,MPI_COMM_WORLD);  
  MPI_Recv(&recvbuf,bufsize,MPI_INT,your_id,0,MPI_COMM_WORLD,&status);
  end=MPI_Wtime();
  } else {
  start=MPI_Wtime();
  MPI_Recv(&recvbuf,bufsize,MPI_INT,your_id,0,MPI_COMM_WORLD,&status);
  MPI_Send(&sendbuf,bufsize,MPI_INT,your_id,0,MPI_COMM_WORLD);
  end=MPI_Wtime();
  }

  printf("Rank %d measured time elapsed in comms: %12.5e\n",my_id,end-start);

  MPI_Barrier(MPI_COMM_WORLD);
  start=MPI_Wtime();
  printf("Rank %d has time %12.12e\n",my_id,start);
  MPI_Barrier(MPI_COMM_WORLD);

  /* Finalize the MPI environment. */
  MPI_Finalize();
}

