#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <math.h>
#include <string.h>

static const int halo_width=2;
static int rank;

const float pi=3.14159;
const float u_x=1., u_y=0., c_amp=1., cdt=.3;
static float dx, dy;

float ugrad_upw(int i, int j, int ny, float data[][ny]){

    const float coeff[]={-3./2.,4./2.,-1./2.};
    float sum_x=0., sum_y=0.;
    int inc,k;

    if (u_x != 0.) {
        inc = -copysign(1.0, u_x);
        for (k=0; k<=halo_width; k++){
            sum_x += coeff[k]*data[i+inc*k][j];
        }
        sum_x *= abs(u_x)/dx;
    }

    if (u_y != 0.) {
        inc = -copysign(1.0, u_y);
        for (k=0; k<=halo_width; k++){
            sum_y += coeff[k]*data[i][j+inc*k];
        }
        sum_y *= abs(u_y)/dy;
    }

    return sum_x + sum_y;
}

int find_proc(int ipx, int ipy, int nprocx, int nprocy)
{
   int ipyl = ipy%nprocy, ipxl = ipx%nprocx;
   if (ipxl < 0) ipxl +=nprocx;
   if (ipyl < 0) ipyl +=nprocy;
   return ipyl*nprocx + ipxl;
}

int* find_proc_coords(int rank, int nprocx, int nprocy)
{
   static int ret[2];

   ret[0]=rank%nprocx;
   ret[1]=rank/nprocx;
   //printf("0rank,ret= %d %d %d\n", rank,ret[0],ret[1]);
   return ret;
}

void rhs(const int xrange[2], const int yrange[2], int ny, float data[][ny], float d_data[][ny])
{
    int ix,iy;

    for (ix = xrange[0]; ix < xrange[1]; ++ix)
        for (iy = yrange[0]; iy < yrange[1]; ++iy)
        {
            d_data[ix][iy] = ugrad_upw(ix, iy, ny, data);
        }
}

int main(int argc, char** argv)
{
    int nprocs, nprocx, nprocy, ipx, ipy, left_neigh, right_neigh, top_neigh, bot_neigh,
        ixstart, ixstop, ixstarti, ixstopi, iystart, iystop, iystarti, iystopi;

    if (argc<5) {
      if (rank==0) {printf("Not enough parameters given -- abort!.\n");}
      MPI_Finalize();
      exit(0);
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    nprocx = atoi(argv[1]); nprocy = atoi(argv[2]);

    if (nprocx*nprocy != nprocs){
        if (rank==0) printf("nprocx*nprocy != nprocs -- abort!");
        MPI_Finalize();
        exit(0);
    }
    int *proc_coords = find_proc_coords(rank,nprocx,nprocy);
    ipx=proc_coords[0]; ipy=proc_coords[1];

    left_neigh  = find_proc(ipx-1,ipy,nprocx,nprocy);
    right_neigh = find_proc(ipx+1,ipy,nprocx,nprocy);

    bot_neigh = find_proc(ipx,ipy-1,nprocx,nprocy);
    top_neigh = find_proc(ipx,ipy+1,nprocx,nprocy);

    //printf("rank,ipx,ipy,left,right,bot,top= %d %d %d %d %d %d %d\n", rank,ipx,ipy,left_neigh,right_neigh,bot_neigh,top_neigh);

    int domain_nx = atoi(argv[3]),
        subdomain_nx = domain_nx/nprocx,           // subdomain x-size w/o halos
        subdomain_mx = subdomain_nx+2*halo_width;  //                  with halos

    int domain_ny = atoi(argv[4]),
        subdomain_ny = domain_ny/nprocy,           // subdomain y-size w/o halos
        subdomain_my = subdomain_ny+2*halo_width;  //                  with halos

    if (domain_nx%nprocx != 0)
    {
        if (rank==0) {printf("Domain x-size not divisible by x processor number! Aborting.\n");}
        return 1;
    }
    if (domain_ny%nprocy != 0)
    {
        if (rank==0) {printf("Domain y-size not divisible by y processor number! Aborting.\n");}
        return 1;
    }
    if (subdomain_nx<2*halo_width || subdomain_ny<2*halo_width)
    {
        if (rank==0) {printf("Subdomain size < stencil_order! Aborting.\n");}
        return 1;
    }

    float data[subdomain_mx][subdomain_my], d_data[subdomain_mx][subdomain_my];

    float xextent=2.*pi, yextent=2.*pi;
    dx=xextent/domain_nx, dy=yextent/domain_ny;
    float x[subdomain_mx], y[subdomain_my];
    int ix, iy;

    for (ix=0;ix<subdomain_mx; ix++) x[ix] = (ipx*subdomain_nx - halo_width + ix + 0.5)*dx;
    for (iy=0;iy<subdomain_my; iy++) y[iy] = (ipy*subdomain_ny - halo_width + iy + 0.5)*dy;
    
    // Initialisation of data.

    for (ix = halo_width; ix < halo_width+subdomain_nx; ++ix)
    {
        for (iy = halo_width; iy < halo_width+subdomain_ny; ++iy)
        {
            data[ix][iy] = c_amp*sin((double) x[ix]);
            //data[ix][iy] = c_amp*sin((double) y[iy]);
            //data[ix][iy] = c_amp*sin((double) x[ix])*sin((double) y[iy]);
        }
    }
/*if (rank==0) {
  int ii, jj;
  for (ii=0; ii<subdomain_mx; ii++){
    for (jj=0; jj<subdomain_my; jj++) printf(" %f",data[ii][jj]);
    printf("\n");
  }
}*/
    MPI_Datatype left_get, right_get, bot_get, top_get;
    MPI_Datatype left_store, right_store, bot_store, top_store;

    int dims_data[2] = {subdomain_mx,subdomain_my};

    int dims_leftright[2] = {halo_width,subdomain_ny};
    int dims_topbot[2]    = {subdomain_nx,halo_width};

    int start_bot_get[2]   = {halo_width,halo_width};
    int start_bot_store[2] = {halo_width,0};
    int start_top_get[2]   = {halo_width,subdomain_ny};
    int start_top_store[2] = {halo_width,subdomain_ny+halo_width};
//printf("rank, start_bot_get: %d %d %d \n",rank,start_bot_get[0],start_bot_get[1]);
//printf("rank, start_top_get: %d %d %d \n",rank,start_top_get[0],start_top_get[1]);
    int start_left_get[2]    = {halo_width,halo_width};
    int start_left_store[2]  = {0,halo_width};
    int start_right_get[2]   = {subdomain_nx,halo_width};
    int start_right_store[2] = {subdomain_nx+halo_width,halo_width};
//printf("rank, start_right_get: %d %d %d \n",rank,start_right_get[0],start_right_get[1]);
//printf("rank, start_left_get: %d %d %d \n",rank,start_left_get[0],start_left_get[1]);

    MPI_Type_create_subarray(2, dims_data, dims_leftright, start_left_get, MPI_ORDER_C, MPI_FLOAT, &left_get);
    MPI_Type_commit(&left_get);

    MPI_Type_create_subarray(2, dims_data, dims_leftright, start_right_get, MPI_ORDER_C, MPI_FLOAT, &right_get);
    MPI_Type_commit(&right_get);

    MPI_Type_create_subarray(2, dims_data, dims_leftright, start_left_store, MPI_ORDER_C, MPI_FLOAT, &left_store);
    MPI_Type_commit(&left_store);

    MPI_Type_create_subarray(2, dims_data, dims_leftright, start_right_store, MPI_ORDER_C, MPI_FLOAT, &right_store);
    MPI_Type_commit(&right_store);

    MPI_Type_create_subarray(2, dims_data, dims_topbot, start_bot_get, MPI_ORDER_C, MPI_FLOAT, &bot_get);
    MPI_Type_commit(&bot_get);

    MPI_Type_create_subarray(2, dims_data, dims_topbot, start_top_get, MPI_ORDER_C, MPI_FLOAT, &top_get);
    MPI_Type_commit(&top_get);

    MPI_Type_create_subarray(2, dims_data, dims_topbot, start_bot_store, MPI_ORDER_C, MPI_FLOAT, &bot_store);
    MPI_Type_commit(&bot_store);

    MPI_Type_create_subarray(2, dims_data, dims_topbot, start_top_store, MPI_ORDER_C, MPI_FLOAT, &top_store);
    MPI_Type_commit(&top_store);

    MPI_Win win;
    MPI_Win_create(data, subdomain_mx*subdomain_my*sizeof(float), 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    
    // "Dry" test of data fetching
    MPI_Win_fence(0, win);
    MPI_Get(data, 1, left_store, left_neigh, 0, 1, right_get, win);
    MPI_Get(data, 1, right_store, right_neigh, 0, 1, left_get, win);
    MPI_Get(data, 1, bot_store, bot_neigh, 0, 1, top_get, win);
    MPI_Get(data, 1, top_store, top_neigh, 0, 1, bot_get, win);
    MPI_Win_fence(0, win);

    MPI_Request lreq, rreq, breq, treq;

    ixstart = halo_width;
    ixstop  = subdomain_nx+ixstart;       // exclusive
    ixstopi = ixstart+halo_width;         // exclusive
    ixstarti= ixstop-halo_width;

    iystart = halo_width;
    iystop  = subdomain_ny+iystart;       // exclusive
    iystopi = iystart+halo_width;         // exclusive
    iystarti= iystop-halo_width;

    unsigned int iterations = atoi(argv[5]);

    int xrange[2], yrange[2];

    if (u_x==0 && u_y==0) {
      if (rank==0) printf("velocity=0: no meaningful simulation -- abort!");
      MPI_Finalize();
      exit(1);
    }
    int lcompute = 1, lcommunicate = 1, lrget=0;
    float dt = cdt*(u_x==0. ? (u_y==0. ? 0. : dy/abs(u_y)) : (u_y==0 ? dx/abs(u_x) : fmin(dx/abs(u_x),dy/abs(u_y))));
    float t=0.;
    
    // Construct file name for data chunk of process.
    char str1[15]="field_chunk",str2[4],str3[]=".dat";
    sprintf(str2, "%d", rank);
    strcat(str1,str2);
    strcat(str1,str3);
    printf("file: %s\n", str1);
    FILE *fptr = fopen(str1,"w");

    double start = MPI_Wtime();

    for (unsigned int iter = 0; iter < iterations; ++iter)
    {
        for (int ix=ixstart; ix < ixstop; ++ix) fprintf(fptr,"%d",data[ix][0]);
        fprintf(fptr,"\n");

        // Get the data (non-blocking!)
         if (lcommunicate) {
             if (lrget){
                 MPI_Win_lock_all(0, win);
                 if (u_x>0) MPI_Rget(data, 1, left_store , left_neigh , 0, 1, right_get, win, &lreq);
                 if (u_x<0) MPI_Rget(data, 1, right_store, right_neigh, 0, 1, left_get , win, &rreq);
                 if (u_y>0) MPI_Rget(data, 1, bot_store  , bot_neigh  , 0, 1, top_get  , win, &breq);
                 if (u_y<0) MPI_Rget(data, 1, top_store  , top_neigh  , 0, 1, bot_get  , win, &treq);
             } else {
                 MPI_Win_fence(0, win);
                 if (u_x>0) MPI_Get(data, 1, left_store , left_neigh , 0, 1, right_get, win);
                 if (u_x<0) MPI_Get(data, 1, right_store, right_neigh, 0, 1, left_get , win);
                 if (u_y>0) MPI_Get(data, 1, bot_store  , bot_neigh  , 0, 1, top_get  , win);
                 if (u_y<0) MPI_Get(data, 1, top_store  , top_neigh  , 0, 1, bot_get  , win);
            }
        }
        // Compute stencils in all points that are not affected by halo points (can be further optimised).

        if (lcompute) {
            xrange[0]=ixstopi; xrange[1]=ixstarti;
            yrange[0]=iystopi; yrange[1]=iystarti;
            rhs(xrange, yrange, subdomain_my, data, d_data);
        }
//if (rank==0) printf("ddata: %e \n", d_data[subdomain_mx/2][subdomain_my/2]);

        if (lcommunicate) {
            if (lrget){
                if (u_x>0) MPI_Wait(&lreq,MPI_STATUS_IGNORE);
                if (u_x<0) MPI_Wait(&rreq,MPI_STATUS_IGNORE);
                if (u_y>0) MPI_Wait(&breq,MPI_STATUS_IGNORE);
                if (u_y<0) MPI_Wait(&treq,MPI_STATUS_IGNORE);
                MPI_Win_unlock_all(win);
            } else {
                MPI_Win_fence(0, win);
            }
        }

        // Data arrived -> compute stencils in all points that *are* affected by halo points (can be further optimised).

        if (lcompute) {
            xrange[0]=ixstart; xrange[1]=ixstopi;
            yrange[0]=iystart; yrange[1]=iystop;
            rhs(xrange, yrange, subdomain_my, data, d_data);

            xrange[0]=ixstarti; xrange[1]=ixstop;
            yrange[0]=iystart;  yrange[1]=iystop;
            rhs(xrange, yrange, subdomain_my, data, d_data);

            xrange[0]=ixstopi; xrange[1]=ixstarti;
            yrange[0]=iystart; yrange[1]=iystopi;
            rhs(xrange, yrange, subdomain_my, data, d_data);

            xrange[0]=ixstopi;  xrange[1]=ixstarti;
            yrange[0]=iystarti; yrange[1]=iystop;
            rhs(xrange, yrange, subdomain_my, data, d_data);

            // update concentration field
            for (ix = ixstart; ix < ixstop; ++ix)
                for (iy = iystart; iy < iystop; ++iy)
                {
                    data[ix][iy] += dt*d_data[ix][iy];
                }
            t = t+dt;
        }
    }
    fclose(fptr);

    double end = MPI_Wtime(), difftime, avtime;
    if (rank==0) printf("time step= %e \n", dt);
    difftime = end  - start;
    printf("proc %u time elapsed: %f\n", rank, difftime);
    MPI_Reduce(&difftime, &avtime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank==0) printf("mean elapsed time per iteration: %f\n",avtime/nprocs/iterations);

    MPI_Win_free(&win);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
