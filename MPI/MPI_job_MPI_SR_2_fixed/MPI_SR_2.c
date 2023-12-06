#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int my_id, your_id, bufsize=1;
    int sendbuf[bufsize], recvbuf[bufsize];
    MPI_Status status;
    MPI_Request send_request, recv_request;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

    your_id = 1 - my_id;

    // Start a non-blocking receive
    MPI_Irecv(&recvbuf, bufsize, MPI_INT, your_id, 0, MPI_COMM_WORLD, &recv_request);

    // Start a non-blocking send
    MPI_Isend(&sendbuf, bufsize, MPI_INT, your_id, 0, MPI_COMM_WORLD, &send_request);

    // Wait for both operations to complete
    MPI_Wait(&send_request, MPI_STATUS_IGNORE);
    MPI_Wait(&recv_request, &status);

    printf("Communication with %d completed\n", your_id);

    MPI_Finalize();
    return 0;
}

