#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {

  // Blocking send and receive: processes send to next higher rank
  // and receive from the next lower rank. 

  int my_id, next_id, prev_id, comm_size, bufsize=500000;
  int sendbuf[bufsize],recvbuf[bufsize];
  double start,end;
  MPI_Status status;

  /* Initialize the MPI execution environment */
  MPI_Init(&argc,&argv);

  /* Ranks */
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  /* Make the processes to greet the class */
  printf("Hello class from processor %s, rank %d out of %d processes\n",
         processor_name, my_id, comm_size);

  next_id = my_id+1;
  prev_id = my_id-1;

  if (my_id !=comm_size-1) {
    /* Send */
    start=MPI_Wtime();
    MPI_Send(&sendbuf,bufsize,MPI_INT,next_id,0,MPI_COMM_WORLD);
  }

  if (my_id != 0) {
    /* Receive */ 
    MPI_Recv(&recvbuf,bufsize,MPI_INT,prev_id,0,MPI_COMM_WORLD,&status);
    end=MPI_Wtime();
    if (my_id != comm_size-1) printf("Time elapsed %e for %d\n",my_id,end-start);
  }

  /* Finalize the MPI environment. */
  MPI_Finalize();

}

