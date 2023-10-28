#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <math.h>

static const int halo_width= ;    // fill in halo_width
static int rank;

const float pi=3.14159;
const float u_x= , u_y= , c_amp=  // choose velocity components and amplitude of initial condition
const float cdt=.3;               // safety factor for timestep (experiment!)
static float dx, dy;              // grid spacings

float ugrad_upw(int i, int j, int ny, float data[][ny]){

// u.grad operator with upwinding acting on field in data at point i,j.

    const float coeff[]={-3./2.,4./2.,-1./2.};
    float sum_x=0., sum_y=0.;
    int k;

    int inc = -copysign(1.0, u_x);
    for (k=0; k<=halo_width; k++){
        sum_x += coeff[k]*data[i+inc*k][j];
    }
    sum_x *= abs(u_x)/dx;

    inc = -copysign(1.0, u_y);
    for (k=0; k<=halo_width; k++){
        sum_y += coeff[k]*data[i][j+inc*k];
    }
    sum_y *= abs(u_y)/dy;

    return sum_x + sum_y;
}

int find_proc(int ipx, int ipy, /*...*/)
{
// Implement finding process rank from coordinates ipx, ipy in process grid!
}

int* find_proc_coords(int rank, /*...*/)
{
// Implement finding process coordinates ipx, ipy in process grid from process rank!
}

void rhs(const int xrange[2], const int yrange[2], int ny, float data[][ny], float d_data[][ny])
{
//Right-hand side d_data of pde for field in data for a subdomain defined by xrange, yrange:
    int ix,iy;

    for (ix = xrange[0]; ix < xrange[1]; ++ix)
        for (iy = yrange[0]; iy < yrange[1]; ++iy)
        {
            d_data[ix][iy] = ugrad_upw(ix, iy, ny, data);
        }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    nprocx = atoi(argv[1]); nprocy = atoi(argv[2]);  // process numbers in x and y directions

    int *proc_coords = find_proc_coords(rank,nprocx,nprocy);
    ipx=proc_coords[0]; ipy=proc_coords[1];

    // Find neighboring processes!

    int domain_nx = atoi(argv[3]),                 // number of gridpoints in x direction
        subdomain_nx =                             // subdomain x-size w/o halos
        subdomain_mx =                             //                  with halos

    int domain_ny = atoi(argv[4]),                 // number of gridpoints in y direction
        subdomain_ny =                             // subdomain y-size w/o halos
        subdomain_my =                             //                  with halos

    // Check compatibility of argv parameters!

    float data[subdomain_mx][subdomain_my], d_data[subdomain_mx][subdomain_my];

    float xextent=2.*pi, yextent=2.*pi;            // domain has extents 2 pi x 2 pi

    // Set grid spacings dx, dy:
    dx=xextent/domain_nx, dy=yextent/domain_ny;

    float x[subdomain_mx], y[subdomain_my];
    int ix, iy;

    // Populate grid coordinate arrays x,y (equidistant): 
    for (ix=0;ix<subdomain_mx; ix++) x[ix] = (ipx*subdomain_nx - halo_width + ix + 0.5)*dx;
    for (iy=0;iy<subdomain_my; iy++) y[iy] = (ipy*subdomain_ny - halo_width + iy + 0.5)*dy;

    float x[subdomain_mx], y[subdomain_my];
    
    // Initialisation of field in data: harmonic function in x (can be modified to a harmonic in y or x and y):
    for (ix = halo_width; ix < halo_width+subdomain_nx; ++ix)
    {
        for (iy = halo_width; iy < halo_width+subdomain_ny; ++iy)
        {
            data[ix][iy] = c_amp*sin((double) x[ix]);
        }
    }
    // Think about convenient data types to access non-contiguous portions of array data!

    MPI_Win win;
    MPI_Win_create(/*...*/)

    unsigned int iterations = atoi(argv[5]);       // number of iterations=timesteps

    if (u_x==0 && u_y==0) {
      if (rank==0) printf("velocity=0 - no meaningful simulation!");
      exit(1);
    }

    // CFL condition for timestep:
    float dt = cdt*(u_x==0 ? (u_y==0 ? 0 : dy/u_y) : (u_y==0 ? dx/u_x : fmin(dx/u_x,dy/u_y)));

    if (rank==0) printf("dt= %f \n",dt);

    float t=0.;

    // Consider proper synchronization measures!

    // Initialize timing!

    for (unsigned int iter = 0; iter < iterations; ++iter)
    {
        // Get the data from neighbors!

        // Compute rhs. Think about concurrency of computation and data fetching by MPI_Get!

        // Update field in data using rhs in d_data (Euler's method):
        for (ix = ixstart; ix < ixstop; ++ix)
            for (iy = iystart; iy < iystop; ++iy)
            {
                data[ix][iy] += dt*d_data[ix][iy];
            }
        t = t+dt;

        // Output solution for checking/visualisation with choosable cadence!
    }

    // Finalize timing!

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
