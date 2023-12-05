#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include<unistd.h>

int main(int argc, char** argv){

    int rank, size;
    double time, start, start_root, time_offset, end;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // as OpenMPI is not synchronizing time: determine time offset between the two ranks
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    if (rank == 0){
      MPI_Send(&start,1,MPI_DOUBLE,1,1, MPI_COMM_WORLD);
    } else {
      MPI_Recv(&start_root,1,MPI_DOUBLE,0,1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      time_offset=start-start_root;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank==0) {
      sleep(4);
      start = MPI_Wtime();
    } else {
      sleep(2);
      end = MPI_Wtime();
    }
    // determine time diff between start on rank 0 and end on rank 1.
    if (rank==0)
      MPI_Ssend(&start, 1, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD);
    else {
      MPI_Recv(&start, 1, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      time=end-(start+time_offset);
      printf("Synced time = %12.5e \n",time);
      printf("Non-synced time = %12.5e \n",end-start);
    }

    MPI_Finalize();

    return 0;
}
