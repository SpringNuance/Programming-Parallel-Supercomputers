#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// Example program that gathers uneven sized data from different processes
// using gather and gatherv

int main(int argc, char** argv) {

/* Initialize the MPI execution environment */
MPI_Init(&argc,&argv);

int rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

/* Get the communicator size - how many processes you have "ordered" */
int nprocs;
MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

// Example program to gather data from processes that possess an unequal sized data

// Local data, now different at different process: the sizes of each data blocks not yet know, but have to be queried...
int* localsizes;
int* offsets;
int* alldata;
// Just as an example, now setting the local size is the same as rank
// Could be whatever, as this is SPDM code.
int localsize=rank;
int localdata[localsize];
int root=0;

// the root process decides how much data will be coming:
// allocate arrays to contain size and offset information
// that will be collected from each process.
 if (rank==root) {
   localsizes = (int*) malloc( nprocs*sizeof(int) );
   offsets = (int*) malloc( nprocs*sizeof(int) );
 }
 // First gather the information of the size of data on each process,
 // to be able to construct global array.
 MPI_Gather(&localsize,1,MPI_INT,
	    localsizes,1,MPI_INT,root, MPI_COMM_WORLD);
 // Now the root can construct the offsets array
 if (rank==root) {
   int total_data = 0;
   for (int i=0; i<nprocs; i++) {
     offsets[i] = total_data;
     total_data += localsizes[i];
   }
   alldata = (int*) malloc( total_data*sizeof(int) );
 }
 // Now we are ready to collect the data from different processes on the root.
 MPI_Gatherv(localdata,localsize,MPI_INT,
	     alldata,localsizes,offsets,MPI_INT,root,MPI_COMM_WORLD);

/* Finalize the MPI environment. */
MPI_Finalize();

}
