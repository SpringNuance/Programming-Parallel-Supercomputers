#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int localsize = rank;
    int localdata[localsize];

    int* localsizes = (int*) malloc(nprocs * sizeof(int));
    int* offsets = (int*) malloc(nprocs * sizeof(int));
    int total_data = 0;

    // Gather the sizes from each process
    MPI_Allgather(&localsize, 1, MPI_INT, localsizes, 1, MPI_INT, MPI_COMM_WORLD);

    // Construct the offsets array
    for (int i = 0; i < nprocs; i++) {
        offsets[i] = total_data;
        total_data += localsizes[i];
    }

    int* alldata = (int*) malloc(total_data * sizeof(int));

    // Collect the data from different processes
    MPI_Allgatherv(localdata, localsize, MPI_INT, alldata, localsizes, offsets, MPI_INT, MPI_COMM_WORLD);

    free(localsizes);
    free(offsets);
    free(alldata);

    MPI_Finalize();
    return 0;
}
