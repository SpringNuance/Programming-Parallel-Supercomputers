#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// Example program that creates a vector data type on sender,
// the receiver receives contiguous data

 int main(int argc, char** argv) {
 
/* Initialize the MPI execution environment */
 MPI_Init(&argc,&argv);
 
 int nprocs, rank;
 MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
 MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 int sender=0;
 int receiver=1;
 double* source;
 double* target;

// Stride = the length of the data with a repeating pattern
// blocklen = the contiguous part of the data in this block
// count = How many such repeating patterns there are
// In this case, blocklength = size of double.
 int stride=3,count=2; 
 source = (double*) malloc(stride*count*sizeof(double));
 target = (double*) malloc(count*sizeof(double));
 MPI_Datatype newvectortype;

 source[0]=3.1415;source[3]=2*source[0];
 source[1]=0.; source[2]=0.; source[4]=0.; source[5]=0.;
 if (rank==sender) {
   MPI_Type_vector(count,1,stride,MPI_DOUBLE,&newvectortype);
   MPI_Type_commit(&newvectortype);
   MPI_Send(source,1,newvectortype,receiver,0,MPI_COMM_WORLD);
   MPI_Type_free(&newvectortype);
 } else if (rank==receiver) {
   MPI_Recv(target,count,MPI_DOUBLE,sender,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
   printf("target= %e %e \n",target[0],target[1]);
 }

}
