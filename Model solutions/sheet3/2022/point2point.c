#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

int main(int argc, char** argv){
/*
  Takes two or three command line arguments:
  First: length of communicated data array
  Second: number of repetitions of poin-to-point communications
  Third: communication mode: integer from 0 to 6, see declarations below.
         If left out, mode is 'BLOCK'.
*/
    int rank, size, repeats, i;
    double time, time2, start, start_root, time_offset, end;
    int len,buflen;
    void *buf;
    float *data;

    MPI_Request request;
    MPI_Status status;

    enum types{BLOCK,NBLOCK,BLOCKBUF,NBLOCKBUF,BLOCKSYNC,NBLOCKSYNC,SENDRECV} type;
    char* names[]={"blocking","nonblocking","blocking buffered","nonblocking buffered",
                   "blocking synchronous","nonblocking synchronous","sendrecv"};
  
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (size!=2) {
      if (rank==0) {printf("Number of processes != 2! Exit.\n");}
      exit(0);
    }

    if (argc<3) {
      if (rank==0) {printf("Not enough parameters given! Exit.\n");}
      exit(0);
    }
    else {
      len=atoi(argv[1]);
      repeats=atoi(argv[2]);
      if (argc==3) {
        type=BLOCK;
        if (rank==0) {printf("No type given! Assuming 'BLOCK'.\n");}
      }
      else type=atoi(argv[3]);
    }
    buflen=len;
    data=(float*)malloc(sizeof(float)*len);

    // initialize data
    for (i=0;i<len;i++) data[i]=(float)i;

    // as OpenMPI is not synchronizing time: determine time offset between the two ranks
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    if (rank == 0){
      MPI_Send(&start,1,MPI_DOUBLE,1,1, MPI_COMM_WORLD);
    }else{
      MPI_Recv(&start_root,1,MPI_DOUBLE,0,1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      time_offset=start-start_root;
      //printf("Time offset = %12.5e %12.12e %12.12e \n",time_offset,start_root,start);
    }
    
    double time_per_repeat[repeats];
    time=0.;
    for (i=0; i<repeats; i++){

      MPI_Barrier(MPI_COMM_WORLD);

      if (rank == 0){

        start = MPI_Wtime();
        switch (type){
        case BLOCK:{
          MPI_Send(data, len, MPI_FLOAT, 1, 1, MPI_COMM_WORLD);
          break;
          }
        case NBLOCK:{
          MPI_Isend(data, len, MPI_FLOAT, 1, 1, MPI_COMM_WORLD, &request);
          MPI_Wait(&request, &status);
          break;
          }
        case BLOCKBUF:{
          int buffer_attached_size=buflen*sizeof(float) + MPI_BSEND_OVERHEAD;
          //printf("Buffer size = %d \n",buffer_attached_size);
          buf = (void*)malloc(buffer_attached_size);
          MPI_Buffer_attach(buf, buffer_attached_size);
          MPI_Bsend(data, len, MPI_FLOAT, 1, 1, MPI_COMM_WORLD);
          MPI_Buffer_detach(buf, &buffer_attached_size);
          free(buf);
          break;
          }
        case NBLOCKBUF:{
          int buffer_attached_size=buflen*sizeof(float) + MPI_BSEND_OVERHEAD;
          //printf("Buffer size = %d \n",buffer_attached_size);
          buf = (void*)malloc(buffer_attached_size);
          MPI_Buffer_attach(buf, buffer_attached_size);
          MPI_Ibsend(data, len, MPI_FLOAT, 1, 1, MPI_COMM_WORLD, &request);
          MPI_Wait(&request, &status);
          MPI_Buffer_detach(buf, &buffer_attached_size);
          free(buf);
          break;
          }
        case BLOCKSYNC:{
          MPI_Ssend(data, len, MPI_FLOAT, 1, 1, MPI_COMM_WORLD);
          break;
          }
        case NBLOCKSYNC:{
          MPI_Issend(data, len, MPI_FLOAT, 1, 1, MPI_COMM_WORLD, &request);
          MPI_Wait(&request, &status);
          break;
          }
        case SENDRECV:
          MPI_Sendrecv(data, len, MPI_FLOAT, 1, 1, NULL, 0, MPI_FLOAT, MPI_PROC_NULL, 0,
                       MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          break;
        }
      }
      else if (rank == 1){
        switch (type){
        case BLOCK:
        case BLOCKBUF:
        case NBLOCKBUF:
        case BLOCKSYNC:
        case NBLOCKSYNC:{
          MPI_Recv(data, len, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          break;
          }
        case NBLOCK:{
          MPI_Irecv(data, len, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request);
          MPI_Wait(&request, &status);
          break;
          }
        case SENDRECV:
          MPI_Sendrecv(NULL, 0, MPI_FLOAT, MPI_PROC_NULL, 0, data, len, MPI_FLOAT, 0, 1,
                       MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          break;
        }
        end = MPI_Wtime();
      }
      MPI_Barrier(MPI_COMM_WORLD);

      // determine time for one communication from difference between start on rank 0 and end on rank 1
      if (rank==0)
        MPI_Send(&start, 1, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD);
      else {
        MPI_Recv(&start, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        time_per_repeat[i]=end-(start+time_offset);
        time+=time_per_repeat[i];
	time2+=end-start;
      }
    }
    if (rank==1) {
      double avtime=time/repeats, devtime=0.;
      for (int i=0; i<repeats; i++){
         devtime+=pow(time_per_repeat[i]-avtime,2);
      }
      devtime=pow(devtime,0.5)/repeats;
      printf("Average execution time for mode '%s' with %d repeats: %12.5e seconds\n",names[type],repeats,avtime);
      printf("Standard deviation of average execution time for mode '%s' with %d repeats: %12.5e seconds\n",names[type],repeats,devtime);
      printf("Average execution time per data element for mode '%s' with %d repeats: %12.5e seconds\n",names[type],repeats,time/repeats/len);
      printf("Bandwidth %12.5e in Gbits/s:\n",8.*sizeof(double)/(time/repeats/len)/1e9);
      printf("Bandwidth %12.5e in GB/s\n:",sizeof(double)/(time/repeats/len)/1e9);
      //printf("Total execution times (sync) %12.5e (non-sync) %12.5e %12.12e %12.12e\n",time,time2,start,end);
    }

    MPI_Finalize();
    free(data);

    return 0;
}

