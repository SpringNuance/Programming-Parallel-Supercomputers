#include <mpi.h>
#include <stdio.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

int main(int argc, char** argv) {

  // Program using FUNNELED level support, to enable concurrent
  // computations with other threads, while the master threads
  // are communicating. Does not do anything sensible, but demonstrates
  // the usage of this programming style.

  int prov,req=MPI_THREAD_FUNNELED;
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
  int nsize=10;
  int a[nsize];

  next_id = world_rank+1;
  prev_id = world_rank-1;

  // Plenty of private variables need to be declared
#pragma omp parallel private(a,idth,nth,recvid,status) 
  {
  idth = omp_get_thread_num();
  nth= omp_get_num_threads();

  #pragma omp master
  {
    if (world_rank != world_size-1) MPI_Send(&idth,1,MPI_INT,next_id,0,MPI_COMM_WORLD);
    if (world_rank != 0) MPI_Recv(&recvid,1,MPI_INT,prev_id,0,MPI_COMM_WORLD,&status);
    printf("Hi, this is rank %d, thread %d! I sent to rank %d and received from rank %d\n",world_rank,idth,next_id,prev_id);
  }
  for (int i=0;i<nsize;i++) {
    a[i]++;
  }
  printf("Hi, this is rank %d, thread %d! I made some silly computations and got %d\n",world_rank,idth,a[0]);
  }

  MPI_Finalize();
}

