#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// Example program that creates a vector data type on sender,
// the receiver receives contiguous data

int main(int argc, char** argv) {

/* Initialize the MPI execution environment */
MPI_Init(&argc,&argv);

int nprocs;
MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
 int sender=0;
 int receiver=1;
 double* source;
 double* target;

// Stride = the length of the data with a repeating pattern
// blocklen = the contiguous part of the data in this block
// count = How many such repeating patterns there are
// In this case, blocklenght = sice of double.
 int stride=3,count=1; 
 source = (double*) malloc(stride*count*sizeof(double));
 target = (double*) malloc(count*sizeof(double));
 MPI_Datatype newvectortype;
 if (nprocs==sender) {
   MPI_Type_vector(count,1,stride,MPI_DOUBLE,&newvectortype);
   MPI_Type_commit(&newvectortype);
   MPI_Send(source,1,newvectortype,receiver,0,MPI_COMM_WORLD);
   MPI_Type_free(&newvectortype);
 } else if (nprocs==receiver) {
   MPI_Recv(target,count,MPI_DOUBLE,sender,0,MPI_COMM_WORLD,
	    MPI_STATUS_IGNORE);
 }



}
