#include <mpi.h>
#include <stdio.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

int main(int argc, char** argv) {

  // Program using highest level of thread support; Communicating from
  // thread to another thread inside omp parallel region.

  int prov,req=MPI_THREAD_MULTIPLE;
  /* Initialize the MPI execution environment */
  MPI_Init_thread(NULL, NULL,req,&prov);

  if( prov < req ) {
     printf("Error: MPI implementation does not support the threading level requested.\n");
     MPI_Abort(MPI_COMM_WORLD,0); }

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int idth,nth,recvid=100;
  int next_id, prev_id;
  MPI_Status status;

  next_id = world_rank+1;
  prev_id = world_rank-1;

  // Plenty of private variables need to be declared
#pragma omp parallel private(idth,nth,recvid,status) 
  {
  idth = omp_get_thread_num();
  nth= omp_get_num_threads();

  /* Do communication "in chain" from prev rank to next rank threads*/
  /* Exlude first and last for simplicity */
  if (world_rank != world_size-1) {
    MPI_Send(&idth,1,MPI_INT,next_id,idth,MPI_COMM_WORLD);
  }
  if (world_rank != 0) {
    MPI_Recv(&recvid,1,MPI_INT,prev_id,idth,MPI_COMM_WORLD,&status);
  }
  printf("Hi, this is rank %d, thread %d! I received from thread %d\n",world_rank,idth, recvid);
}

  MPI_Finalize();
}

