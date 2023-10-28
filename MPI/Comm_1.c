#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// Example program that creates a 2D torus topology using MPI_Cart_create
// Program written assuming that 16 processes are launched, if some other
// processor count used, aborts.
// Define neighbors for 1st order von Neumann type comm. pattern

int main(int argc, char** argv) {

/* Initialize the MPI execution environment */
MPI_Init(&argc,&argv);

 int nprocs, oldrank;
 MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
 if(nprocs != 16)
   {
     printf("This application is meant to be run with 16 processes, not %d.\n", nprocs);
     MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
   }
 MPI_Comm_rank(MPI_COMM_WORLD, &oldrank); 
 if (oldrank==0) printf("We are creating a 2D torus topology for %d processes.\n",nprocs);

 MPI_Comm my_new_comm;
 int my_id;
 int dim = 2;
 const int dims[2] = {4,4};
 const int periodic[2] = {1,1};
 int coords[2];

 MPI_Cart_create(MPI_COMM_WORLD,dim,dims,periodic,1,&my_new_comm);
 MPI_Comm_rank(my_new_comm, &my_id);
 MPI_Cart_coords(my_new_comm, my_id,dim,coords);
 printf("My rank %d coords %d %d\n",my_id,coords[0],coords[1]);
 // Already know the ranks, this would be the transformation command
 // from coords to ranks otherwise.
 //MPI_Cart_rank(my_new_comm,coords,&my_id);

 int nghbr_left, nghbr_right, nghbr_up, nghbr_down;
 // Shift first in 0th (x) dimension, positive means "up" row-wise index
 MPI_Cart_shift(my_new_comm, 0, 1, &nghbr_up, &nghbr_down);

 // Shift first in 1st (y) dimension, positive means "right" col-wise index
 MPI_Cart_shift(my_new_comm, 1, 1, &nghbr_left, &nghbr_right);

 printf("[MPI process %d] I have left, right, up, down neighbors %d %d %d %d.\n", my_id,nghbr_left,nghbr_right,nghbr_up,nghbr_down);

 MPI_Finalize();
}
