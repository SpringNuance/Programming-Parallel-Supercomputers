#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>

static const unsigned int stencil_order = 4, halo_width=stencil_order/2;
static unsigned int rank, nprocs;

float stencil(int i, float *data){

    const float coeff[]={1./16.,1./4.,3./8.,1./4.,1./16.};
    float ret = coeff[0]*data[i-2]+coeff[1]*data[i-1]+coeff[2]*data[i]+coeff[3]*data[i+1]+coeff[4]*data[i+2];
    //printf("data= %f %f %f %f %f %f \n", coeff[0]*data[i-2], coeff[1]*data[i-1], coeff[2]*data[i], coeff[3]*data[i+1], coeff[4]*data[i+2], ret);
    return ret;
}

void serial(int mx, float data[mx], int iterations)
{
    int nx, istart, istop, istopi, istarti;

    nx = mx-2*halo_width;
    istart = halo_width;
    istop  = nx+istart;    // exclusive
    istarti= istop-halo_width;

    float data_up[mx];
    for (unsigned int iter = 0; iter < iterations; ++iter)
    {
        // periodicity
        memcpy(data,              &(data[istarti]),sizeof(float)*halo_width);
        memcpy(&(data[istart+nx]),&(data[istart] ),sizeof(float)*halo_width);

        // Compute stencils in all points that are not affected by halo points.
        for (int i = istart; i < istop; ++i)
        {
            data_up[i] = stencil(i, data);
        }
        memcpy(data,data_up,sizeof(float)*mx);
    }
}

float maxvaldiff(int n,float data0[n], float data1[n])
{
    float max;
    max = abs(data0[0] - data1[0]);
    for (int i=1; i<n; i++){
        if (abs(data0[i]-data1[i]) > max) max = abs(data0[i]-data1[i]);
    }
    return max;
}

// main program: 1st arg = size of domain, 2nd = no of iterations, 3rd = blocking(1) or non-blocking(0)
int main(int argc, char** argv)
{
    enum side {LEFT,RIGHT};
    int blocking=1;
    if (argc>=4)
    {
        blocking = atoi(argv[3]);
        if (!(blocking==1 || blocking==0))
        {
          if (rank==0) {printf("blocking in argv[3] needs to be either 0 or 1!");}
          return 1;
        }
    }

    float sendbuf[2][halo_width], recvbuf[2][halo_width];    // buffers for send/recv to left and right
    unsigned int left, right, istart, istop, istarti, istopi;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    right = (rank+1)%nprocs;                                 // left and right neighbors
    left  = (nprocs+rank-1)%nprocs;

    /* alternatively
    int period=1;
    int nbors[2];
    MPI_Comm cartcomm;
    MPI_Cart_create(MPI_COMM_WORLD, 1, &nprocs, &period, 1, &cartcomm);
    MPI_Cart_shift(cartcomm, 0, 1, nbors, nbors + 1);
    printf("cart neighbors %d %d %d\n",rank, nbors[0],nbors[1]);
    */

    unsigned int domain_nx = atoi(argv[1]), domain_mx=domain_nx+2*halo_width,
                 subdomain_nx = domain_nx/nprocs,           // subdomain size w/o halos
                 subdomain_mx = subdomain_nx+2*halo_width;  //                with halos

    if (domain_nx%nprocs != 0)
    {
        if (rank==0) {printf("Domain size not divisible by processor number! Aborting.\n");}
        return 1;
    }
    if (subdomain_nx<stencil_order)
    {
        if (rank==0) {printf("Subdomain size < stencil_order! Aborting.\n");}
        return 1;
    }
    unsigned int iterations = atoi(argv[2]);

    istart = halo_width;
    istop  = subdomain_nx+istart;    // exclusive
    istopi = istart+halo_width;      // exclsuive
    istarti= istop-halo_width;
//printf("istart,istop,istarti,istopi= %d %d %d %d %d \n",istart,istop,istarti,istopi,iterations);
    float data[subdomain_mx],data_up[subdomain_mx];

    MPI_Request send_req[2], recv_req[2];

    // arbitrary initialization of data
    for (size_t i = istart; i < subdomain_nx+istart; ++i)
    {
        data[i] = (float)((rank+1)*i);
    }
#ifdef VERIFY
    // For verification. Activate when compiling.
    // GLobalize initial condition.
    float refdata[domain_mx];
    MPI_Allgather(&(data[istart]),subdomain_nx,MPI_FLOAT,&(refdata[istart]),subdomain_nx,MPI_FLOAT,MPI_COMM_WORLD);
#endif
    size_t datasz=sizeof(float);
    double start = MPI_Wtime();

    for (unsigned int iter = 0; iter < iterations; ++iter)
    {
        memcpy(&(sendbuf[LEFT ][0]),&(data[istart ]),halo_width*datasz);
        memcpy(&(sendbuf[RIGHT][0]),&(data[istarti]),halo_width*datasz);

        if (blocking)
        {
            MPI_Sendrecv(&(sendbuf[LEFT ][0]), halo_width, MPI_FLOAT, left , rank ,
                         &(recvbuf[RIGHT][0]), halo_width, MPI_FLOAT, right, right, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            MPI_Sendrecv(&(sendbuf[RIGHT][0]), halo_width, MPI_FLOAT, right, rank ,
                         &(recvbuf[LEFT ][0]), halo_width, MPI_FLOAT, left , left , MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            memcpy(&(data[0]    ),&(recvbuf[LEFT ][0]),halo_width*datasz);   // copy data into left halo
            memcpy(&(data[istop]),&(recvbuf[RIGHT][0]),halo_width*datasz);   //                right

            // Compute stencils in all points.
            for (int i = istart; i < istop; ++i)
            {
                data_up[i] = stencil(i, data);
            }
        }
        else
        {
            MPI_Isend(&(sendbuf[LEFT ][0]), halo_width, MPI_FLOAT, left ,rank , MPI_COMM_WORLD, &(send_req[LEFT] ));
            MPI_Isend(&(sendbuf[RIGHT][0]), halo_width, MPI_FLOAT, right,rank , MPI_COMM_WORLD, &(send_req[RIGHT]));
            MPI_Irecv(&(recvbuf[LEFT ][0]), halo_width, MPI_FLOAT, left ,left , MPI_COMM_WORLD, &(recv_req[LEFT] ));
            MPI_Irecv(&(recvbuf[RIGHT][0]), halo_width, MPI_FLOAT, right,right, MPI_COMM_WORLD, &(recv_req[RIGHT]));

            // Compute stencils in all points that are not affected by halo points.
            for (int i = istopi; i < istarti; ++i)
            {
                data_up[i] = stencil(i, data);
            }

            //MPI_Wait(&(send_req[LEFT ]), MPI_STATUSES_IGNORE);
            //MPI_Wait(&(send_req[RIGHT]), MPI_STATUSES_IGNORE);
            MPI_Wait(&(recv_req[LEFT ]), MPI_STATUSES_IGNORE);
            MPI_Wait(&(recv_req[RIGHT]), MPI_STATUSES_IGNORE);
            memcpy(&(data[0]    ),&(recvbuf[LEFT ][0]),halo_width*datasz);
            memcpy(&(data[istop]),&(recvbuf[RIGHT][0]),halo_width*datasz);

            // Compute Stencils in all points that are affected by halo points.
            for (int i = istart; i < istopi; ++i)
            {
                data_up[i] = stencil(i, data);
            }
            for (int i = istarti; i < istop; ++i)
            {
                data_up[i] = stencil(i, data);
            }
        }
        // Update original data
        for (int i = istart; i < istop; ++i)
        {
            data[i] = data_up[i];      // for PDEs typically data[i] += fac*data_up[i];
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double end = MPI_Wtime(), time=end-start, avtime;
    //printf("Elapsed timr for rank %u: %f\n", rank, time);

    MPI_Reduce(&time,&avtime,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    //if (rank==0) printf("Averaged elapsed time: %f\n", avtime/nprocs);
    if (rank==0) printf("Averaged time per iteration for %d gridpoints (%sblocking): %11.5e\n", domain_nx, (blocking ? "" : "non"), avtime/nprocs/iterations);
    if (rank==0) printf("Averaged time per grid point and iteration (%sblocking): %11.5e\n", (blocking ? "" : "non"), avtime/nprocs/iterations/domain_nx);

#ifdef VERIFY
    serial(domain_mx, refdata, iterations);
//if (rank==0) {for (size_t n=0; n<domain_mx; ++n) printf("%f ",refdata[n]); printf("\n");
    float maxdiff=maxvaldiff(subdomain_nx,&(data[halo_width]),&(refdata[rank*subdomain_nx+halo_width]));
    if (maxdiff>0) printf("maxdiff on rank %d = %f \n", rank, maxdiff);
#endif
    MPI_Finalize();
    return 0;
}
