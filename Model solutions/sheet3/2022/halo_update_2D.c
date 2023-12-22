#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>

static const unsigned int stencil_order = 4, halo_width=stencil_order/2;

float stencil(int i, int j, int ny, float data[][ny]){

    const float coeff[]={1./32.,1./8.,3./16.,1./8.,1./32.};
    float sum=0.;
    int k;

    for (k=-halo_width; k<=halo_width; k++){
        sum += coeff[k+halo_width]*data[i+k][j];
    }

    for (k=-halo_width; k<=halo_width; k++){
        sum += coeff[k+halo_width]*data[i][j+k];
    }
    return sum;
}

int find_proc(int ipx, int ipy, int nprocx, int nprocy)
{
   int ipyl = ipy>=0 ? ipy%nprocy : (nprocy+ipy)%nprocy,
       ipxl = ipx>=0 ? ipx%nprocx : (nprocx+ipx)%nprocx;

   return ipyl*nprocx + ipxl;
}

int* find_proc_coords(int rank, int nprocx, int nprocy)
{
   static int ret[2];

   ret[0]=rank%nprocx;
   ret[1]=(rank/nprocx)%nprocy;
//printf("0rank,ret= %d %d %d\n", rank,ret[0],ret[1]);
   return ret;
}

void update_data(const int xrange[2], const int yrange[2], int ny, float data[][ny])
{
    int ix,iy;

    for (ix = xrange[0]; ix < xrange[1]; ++ix)
        for (iy = yrange[0]; iy < yrange[1]; ++iy)
        {
            data[ix][iy] = stencil(ix, iy, ny, data);
        }
}
int main(int argc, char** argv)
{
    enum sidelr {LEFT,RIGHT};
    enum sidebt {BOT,TOP};

    const int blocking=0;

    unsigned int rank, nprocs, nprocx, nprocy, ipx, ipy, left, right, top, bot,
                 ixstart, ixstop, ixstarti, ixstopi, iystart, iystop, iystarti, iystopi;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    nprocx = atoi(argv[1]); nprocy = atoi(argv[2]);

    int *proc_coords = find_proc_coords(rank,nprocx,nprocy);
    ipx=proc_coords[0]; ipy=proc_coords[1];

    left  = find_proc(ipx-1,ipy,nprocx,nprocy);
    right = find_proc(ipx+1,ipy,nprocx,nprocy);

    bot = find_proc(ipx,ipy-1,nprocx,nprocy);
    top = find_proc(ipx,ipy+1,nprocx,nprocy);

    //printf("rank,ipx,ipy,left,right,bot,top= %d %d %d %d %d %d %d\n", rank,ipx,ipy,left,right,bot,top);

    unsigned int domain_nx = atoi(argv[3]),
                 subdomain_nx = domain_nx/nprocx,           // subdomain size w/o halos
                 subdomain_mx = subdomain_nx+2*halo_width;  //                with halos
    unsigned int domain_ny = atoi(argv[4]),
                 subdomain_ny = domain_ny/nprocy,           // subdomain size w/o halos
                 subdomain_my = subdomain_ny+2*halo_width;  //                with halos

    if (domain_nx%nprocx != 0)
    {
        if (rank==0) {printf("Domain x size not divisible by x processor number! Aborting.\n");}
        return 1;
    }
    if (domain_ny%nprocy != 0)
    {
        if (rank==0) {printf("Domain y size not divisible by y processor number! Aborting.\n");}
        return 1;
    }
    if (subdomain_nx<stencil_order || subdomain_ny<stencil_order)
    {
        if (rank==0) {printf("Subdomain size < stencil_order! Aborting.\n");}
        return 1;
    }

    float sendbufx[2][halo_width][subdomain_my], recvbufx[2][halo_width][subdomain_my];    // buffers for send/recv to left and right
    float sendbufy[2][subdomain_nx][halo_width], recvbufy[2][subdomain_nx][halo_width];    // buffers for send/recv to bottom and top
    int bufx_size=halo_width*subdomain_my, bufy_size=subdomain_nx*halo_width;

    ixstart = halo_width;
    ixstop  = subdomain_nx+ixstart;       // exclusive
    ixstopi = ixstart+halo_width;         // exclsuive
    ixstarti= ixstop-halo_width;

    iystart = halo_width;
    iystop  = subdomain_ny+iystart;       // exclusive
    iystopi = iystart+halo_width;         // exclsuive
    iystarti= iystop-halo_width;
//printf("ixstart,ixstop,ixstarti,ixstopi= %d %d %d %d %d \n",ixstart,ixstop,ixstarti,ixstopi,iterations);

    MPI_Request sendx_req[2], recvx_req[2], sendy_req[2], recvy_req[2];
    float data[subdomain_mx][subdomain_my];

    // Arbitrary initialisation of data.

    float *ptr=&(data[0][0]);
    for (size_t i = 0; i < subdomain_mx*subdomain_my; ++i)
    {
        *ptr = (float)(rank*i);
        ptr++;
    }

    size_t datasz=sizeof(float);
    int ix, iy, xrange[2], yrange[2];

    unsigned int iterations = atoi(argv[5]);

    double start = MPI_Wtime();

    for (unsigned int iter = 0; iter < iterations; ++iter)
    {
        memcpy(&(sendbufx[LEFT ][0][0]),&(data[ixstart ][0]),bufy_size*datasz);
        memcpy(&(sendbufx[RIGHT][0][0]),&(data[ixstarti][0]),bufy_size*datasz);

        for (ix = ixstart; ix < ixstop; ++ix)
        {
            for (iy = iystart; iy < iystopi; ++iy)
            {
                sendbufy[BOT][ix-ixstart][iy-iystart] = data[ix][iy];
            }
            for (iy = iystarti; iy < iystop; ++iy)
            {
                sendbufy[TOP][ix-ixstart][iy-iystarti] = data[ix][iy];
            }
        }

        if (blocking)
        {
            MPI_Sendrecv(&(sendbufx[LEFT ][0][0]), bufx_size, MPI_FLOAT, left , rank ,
                         &(recvbufx[RIGHT][0][0]), bufx_size, MPI_FLOAT, right, right, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            MPI_Sendrecv(&(sendbufx[RIGHT][0][0]), bufx_size, MPI_FLOAT, right, rank ,
                         &(recvbufx[LEFT ][0][0]), bufx_size, MPI_FLOAT, left , left , MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

            MPI_Sendrecv(&(sendbufy[BOT][0][0]), bufy_size, MPI_FLOAT, bot, rank,
                         &(recvbufy[TOP][0][0]), bufy_size, MPI_FLOAT, top, top , MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            MPI_Sendrecv(&(sendbufy[TOP][0][0]), bufy_size, MPI_FLOAT, top, rank,
                         &(recvbufy[BOT][0][0]), bufy_size, MPI_FLOAT, bot, bot , MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

            memcpy(&(data[0]     [0]),&(recvbufx[LEFT ][0][0]),bufx_size*datasz);
            memcpy(&(data[ixstop][0]),&(recvbufx[RIGHT][0][0]),bufx_size*datasz);

            for (ix = ixstart; ix < ixstop; ++ix)
            {
                for (iy = 0; iy < iystart; ++iy)
                {
                data[ix][iy] = recvbufy[BOT][ix-ixstart][iy];
                }
                for (iy = iystop; iy < subdomain_my; ++iy)
                {
                data[ix][iy] = recvbufy[TOP][ix-ixstart][iy-iystop];
                }
            }

        }
        else
        {
            MPI_Isend(&(sendbufx[LEFT ][0][0]),bufx_size, MPI_FLOAT, left ,rank , MPI_COMM_WORLD, &(sendx_req[LEFT] ));
            MPI_Isend(&(sendbufx[RIGHT][0][0]),bufx_size, MPI_FLOAT, right,rank , MPI_COMM_WORLD, &(sendx_req[RIGHT]));
            MPI_Irecv(&(recvbufx[LEFT ][0][0]),bufx_size, MPI_FLOAT, left ,left , MPI_COMM_WORLD, &(recvx_req[LEFT] ));
            MPI_Irecv(&(recvbufx[RIGHT][0][0]),bufx_size, MPI_FLOAT, right,right, MPI_COMM_WORLD, &(recvx_req[RIGHT]));

            MPI_Isend(&(sendbufy[BOT][0][0]),bufy_size, MPI_FLOAT, bot, rank, MPI_COMM_WORLD, &(sendy_req[BOT]));
            MPI_Isend(&(sendbufy[TOP][0][0]),bufy_size, MPI_FLOAT, top, rank, MPI_COMM_WORLD, &(sendy_req[TOP]));
            MPI_Irecv(&(recvbufy[BOT][0][0]),bufy_size, MPI_FLOAT, bot, bot , MPI_COMM_WORLD, &(recvy_req[BOT]));
            MPI_Irecv(&(recvbufy[TOP][0][0]),bufy_size, MPI_FLOAT, top, top , MPI_COMM_WORLD, &(recvy_req[TOP]));
        }

        // Compute stencils in all points that are not affected by halo points.
        xrange[0]=ixstopi; xrange[1]=ixstarti;
        yrange[0]=iystopi; yrange[1]=iystarti;
        update_data(xrange, yrange, subdomain_my, data);
        /*for (ix = ixstopi; ix < ixstarti; ++ix)
            for (iy = iystopi; iy < iystarti; ++iy)
            {
                data[ix][iy] = stencil(ix, iy, subdomain_my, data);
            }
        */
        if (!blocking)
        {
            //MPI_Wait(&(send_req[LEFT ]), MPI_STATUSES_IGNORE);
            //MPI_Wait(&(send_req[RIGHT]), MPI_STATUSES_IGNORE);
            MPI_Wait(&(recvx_req[LEFT ]), MPI_STATUSES_IGNORE);
            MPI_Wait(&(recvx_req[RIGHT]), MPI_STATUSES_IGNORE);

            MPI_Wait(&(recvy_req[BOT]), MPI_STATUSES_IGNORE);
            MPI_Wait(&(recvy_req[TOP]), MPI_STATUSES_IGNORE);

            memcpy(&(data[0]     [0]),&(recvbufx[LEFT ][0][0]),bufx_size*datasz);
            memcpy(&(data[ixstop][0]),&(recvbufx[RIGHT][0][0]),bufx_size*datasz);

            for (ix = ixstart; ix < ixstop; ++ix)
            {
                for (iy = 0; iy < iystart; ++iy)
                {
                data[ix][iy] = recvbufy[BOT][ix-ixstart][iy];
                }
                for (iy = iystop; iy < subdomain_my; ++iy)
                {
                data[ix][iy] = recvbufy[TOP][ix-ixstart][iy-iystop];
                }
            }

        }

        // Compute stencils in all points that *are* affected by halo points.

        xrange[0]=ixstart; xrange[1]=ixstopi;
        yrange[0]=iystart; yrange[1]=iystop;
        update_data(xrange, yrange, subdomain_my, data);

        xrange[0]=ixstarti; xrange[1]=ixstop;
        yrange[0]=iystart; yrange[1]=iystop;
        update_data(xrange, yrange, subdomain_my, data);

        xrange[0]=ixstopi; xrange[1]=ixstarti;
        yrange[0]=iystart; yrange[1]=iystopi;
        update_data(xrange, yrange, subdomain_my, data);

        xrange[0]=ixstopi; xrange[1]=ixstarti;
        yrange[0]=iystarti; yrange[1]=iystop;
        update_data(xrange, yrange, subdomain_my, data);
        /*
        for (ix = ixstart; ix < ixstopi; ++ix)
            for (iy = iystart; iy < iystop; ++iy)
            {
                data[ix][iy] = stencil(ix, iy, subdomain_my, data);
            }

        for (ix = ixstarti; ix < ixstop; ++ix)
            for (iy = iystart; iy < iystop; ++iy)
            {
                data[ix][iy] = stencil(ix, iy, subdomain_my, data);
            }

        for (ix = ixstopi; ix < ixstarti; ++ix)
            for (iy = iystart; iy < iystopi; ++iy)
            {
                data[ix][iy] = stencil(ix, iy, subdomain_my, data);
            }

        for (ix = ixstopi; ix < ixstarti; ++ix)
            for (iy = iystarti; iy < iystop; ++iy)
            {
                data[ix][iy] = stencil(ix, iy, subdomain_my, data);
            }
        */
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double end = MPI_Wtime();
    printf("proc %u time elapsed: %f\n", rank, end  - start);

    return 0;
}

~                                                                 
