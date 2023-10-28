#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char** argv) {

/* Initialize the MPI execution environment */
MPI_Init(&argc,&argv);

int myrank;
MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
float data_e=50.*myrank;
float data_o=-100.*myrank;

MPI_Comm comm_even;
{
  MPI_Group group_world,group_work;
  MPI_Comm_group(MPI_COMM_WORLD,&group_world);
  int even[] = {0,2};
  MPI_Group_incl( group_world,2,even,&group_work );
  MPI_Comm_create(MPI_COMM_WORLD,group_work,&comm_even );
  MPI_Group_free( &group_world ); MPI_Group_free( &group_work );
}

MPI_Comm comm_odd;
{ 
  MPI_Group group_world,group_work;
  MPI_Comm_group(MPI_COMM_WORLD,&group_world);
  int even[] = {0,2};
  MPI_Group_excl( group_world,2,even,&group_work );
  MPI_Comm_create(MPI_COMM_WORLD,group_work,&comm_odd );
  MPI_Group_free( &group_world ); MPI_Group_free( &group_work );
}

if(comm_even == MPI_COMM_NULL)
    {
        // I am not part of the new communicator, I can't participate to that broadcast.
       printf("Process %d did not take part to the even communicator broadcast.\n", myrank);
                     }
                         else
                             {
        // I am part of the new communicator, I can participate to that broadcast.
       MPI_Bcast(&data_e, 1, MPI_FLOAT, 0, comm_even);
       printf("Process %d took part to the even communicator broadcast.\n", myrank);
   }

if(comm_odd == MPI_COMM_NULL)
    {
    // I am not part of the new communicator, I can't participate to that broadcast.
  printf("Process %d did not take part to the odd communicator broadcast.\n", myrank);
    } else
    {
    // I am part of the new communicator, I can participate to that broadcast.                
    MPI_Bcast(&data_o, 1, MPI_FLOAT, 0, comm_odd);                             
    printf("Process %d took part to the odd communicator broadcast.\n", myrank);
}

printf("Rank %d has data values %f %f \n",myrank,data_e,data_o);

MPI_Finalize();

}
