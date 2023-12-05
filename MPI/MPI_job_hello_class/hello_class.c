#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {

  double start, end;

  /* Initialize the MPI execution environment */
  MPI_Init(NULL, NULL);

  start=MPI_Wtime();

  /* Get the communicator size - how many processes you have "ordered" */
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  /* Get the rank of the process */
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  /* Where are we assigned to run? */
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  /* Make the processes to greet the class */
  printf("Hello class from processor %s, rank %d out of %d processes\n",
	 processor_name, world_rank, world_size);

  end=MPI_Wtime();

  printf("Time taken by the process %e\n",end-start);

  /* Finalize the MPI environment. */
  MPI_Finalize();
}

