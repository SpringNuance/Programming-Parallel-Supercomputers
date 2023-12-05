#include <mpi.h>
#include <stdio.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

int main(int argc, char** argv) {

  int prov,req=MPI_THREAD_MULTIPLE;
  /* Initialize the MPI execution environment */
  MPI_Init_thread(NULL, NULL,req,&prov);

  if( prov < req ) {
     printf("Error: MPI implementation does not support the threading level requested.\n");
     MPI_Abort(MPI_COMM_WORLD,0); }

  /* Get the communicator size - how many processes you have "ordered" */
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  /* Get the rank of the process */
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  /* Print out the support that is provided */
  if (world_rank==0) printf("Provided support: %d\n",prov);

  /* Where are we assigned to run? */
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  /* Make the processes to greet the class */
  printf("Hello class from node %s, rank %d out of %d MPI processes\n",
	 processor_name, world_rank, world_size);

  #if defined(_OPENMP)
  #pragma omp parallel
    printf("Hello class from OMP thread %d.\n", omp_get_thread_num());
  #else
    printf("Hello class!.\n");
  #endif

  /* Finalize the MPI environment. */
  MPI_Finalize();
}

