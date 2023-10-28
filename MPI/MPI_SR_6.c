#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 2-proc. test program to test MPI_Send versus MPI_BSend */

int main(argc,argv) int argc; char *argv[];
{
    int numtasks, rank, dest, source, rc, count;  
    char *inmsg;
    char *outmsg = "Testing testing";
    MPI_Status Stat;
    int bufsize = strlen(outmsg) * sizeof(char);
    char *buf = malloc(bufsize); 
    inmsg = (char *) malloc(10 * sizeof(char));

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        //double bstart = MPI_Wtime();
        MPI_Buffer_attach( buf, bufsize ); 
        double bstart = MPI_Wtime();
	MPI_Bsend(&outmsg, strlen(outmsg), MPI_CHAR, 1, 1, MPI_COMM_WORLD);
        double bend = MPI_Wtime();
        MPI_Buffer_detach(&buf, &bufsize); 
        //double bend = MPI_Wtime();
        printf("Rank %d: buffered send completed. Time: %e\n",rank,bend-bstart);
        
        double normalstart=MPI_Wtime();
        MPI_Send(&outmsg, strlen(outmsg), MPI_CHAR, 1, 2, MPI_COMM_WORLD);
        double normalend =  MPI_Wtime();
        printf("Rank %d: normal send. Time: %e\n",rank,normalend-normalstart);
        
    } else {
        MPI_Recv(&inmsg, strlen(outmsg), MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &Stat);
        rc = MPI_Get_count(&Stat, MPI_CHAR, &count);
        printf("Rank %d: Received %d char(s) from Rank %d with tag %d \n", rank, count, Stat.MPI_SOURCE, Stat.MPI_TAG);
        MPI_Recv(&inmsg, strlen(outmsg), MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &Stat);
        rc = MPI_Get_count(&Stat, MPI_CHAR, &count);
        printf("Rank %d: Received %d char(s) from Rank %d with tag %d \n",
        rank, count, Stat.MPI_SOURCE, Stat.MPI_TAG);
    }
    MPI_Finalize();
}

