#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

//
// Example program splitting the communicator into a column
// communicator; assumption is a 2D domain decomposition into 
// at least four processes.
//
int main(int argc, char** argv) {

/* Initialize the MPI execution environment */
MPI_Init(&argc,&argv);

int comm_size;
MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
if(comm_size < 4)
    {
        printf("We need at least 4 MPI processes, not %d.\n", comm_size);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

int ncol=2;
int proc_c;
int myrank;
MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

// Generating "color" for the grouping. 
proc_c = myrank % ncol; 

// Generating some data that can show us we are doing the correct comms.
float data_c=10.*myrank;
 
MPI_Comm column_comm; 
MPI_Comm row_comm;

// Splitting the communicators using the column numbers.
// We keep the original ranks assigned by the default communicator.
MPI_Comm_split(MPI_COMM_WORLD, proc_c, myrank,&column_comm);

//Who are we in the subcommunicators?
int mynewrank;
MPI_Comm_rank(column_comm, &mynewrank);
// As the ordering is kept, then "0" is the first process in the subcomms.
MPI_Bcast(&data_c,1,MPI_FLOAT,0,column_comm);

printf("I am global rank %d, local rank %d, with proc_c %d and I have the data value %f\n",myrank,mynewrank,proc_c,data_c); 

MPI_Comm_free(&column_comm);

MPI_Finalize();

}
