#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(&argc,&argv);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Ensure there are exactly two MPI processes
    if (world_size != 2) {
        fprintf(stderr, "World size must be two for %s\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int sizes[] = {1024, 10*1024, 100*1024, 1024*1024, 10*1024*1024}; // 1KB, 10KB, 100KB, 1MB, 10MB
    int num_sizes = 5;
    double start_time, end_time;
    char* buffer;

    for (int i = 0; i < num_sizes; i++) {
        int size = sizes[i];
        buffer = (char*) malloc(size);

        if (world_rank == 0) {
            // MPI Barrier for synchronization

            MPI_Barrier(MPI_COMM_WORLD);

            start_time = MPI_Wtime();

            MPI_Send(buffer, size, MPI_CHAR, 1, 0, MPI_COMM_WORLD);

            end_time = MPI_Wtime();

            printf("Send %d bytes took %f seconds\n", size, end_time - start_time);

        } else if (world_rank == 1) {

            // MPI Barrier for synchronization

            MPI_Barrier(MPI_COMM_WORLD);

            MPI_Recv(buffer, size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        free(buffer);
    }

    MPI_Finalize();
}

