#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <math.h>

// static const int halo_width=??? ;    // fill in halo_width
static const int halo_width=1; // For first order finite difference
// static const int halo_width=2; // For second order finite difference
static int rank;
static int nprocs, nprocx, nprocy;
const float pi = 3.14159;
int print_debug = 0;

// choose velocity components and amplitude of initial condition

// const float u_x = ??? ; 
// const float u_y = ??? ;
// const float c_amp = ??? ;  

const float u_x = 1.0; // choose velocity components of initial condition
const float u_y = 1.0; // choose velocity components ofinitial condition
const float c_amp = 1.0;  // choose amplitude of initial condition
const float cdt = 0.3;  // safety factor for timestep (experiment!)
static float dx, dy; // grid spacings

float ugrad_upw(int i, int j, int ny, float data[][ny]){

// u.grad operator with upwinding acting on field in data at point i,j.
    const float coeff[]={-3./2.,4./2.,-1./2.};
    float sum_x=0.0, sum_y=0.0;
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

// Implement finding process rank from coordinates ipx, ipy in process grid!
int find_proc(int ipx, int ipy, int nprocx, int nprocy) {
    int px = (ipx + nprocx) % nprocx;
    int py = (ipy + nprocy) % nprocy;
    return py * nprocx + px;
}

// Implement finding process coordinates ipx, ipy in process grid from process rank!
int* find_proc_coords(int rank, int nprocx, int nprocy) {
    static int coords[2];
    coords[0] = rank % nprocx; // x-coordinate
    coords[1] = rank / nprocx; // y-coordinate
    return coords;
}

void rhs(const int xrange[2], const int yrange[2], int ny, float data[][ny], float d_data[][ny]) {
//Right-hand side d_data of pde for field in data for a subdomain defined by xrange, yrange:
    for (int ix = xrange[0]; ix < xrange[1]; ++ix){
        for (int iy = yrange[0]; iy < yrange[1]; ++iy) {
            d_data[ix][iy] = ugrad_upw(ix, iy, ny, data);
        }
    }

}


// Function to print a matrix
void print_matrix(const char* name, float *matrix, int rows, int cols) {
    printf("Matrix: %s\n", name);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%8.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}



int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // nprocs = nprocx * nprocy = num-tasks
    // SUppose nprocs = 4, nprocx = 2, nprocy = 2

    nprocx = atoi(argv[1]); // process numbers in x directions = px subinterval in the README
    nprocy = atoi(argv[2]);  // process numbers in y directions = py subinterval in the README

    if (rank == 0) printf("Your configurations\n\n");
    if (rank == 0) printf("nprocx = %d, nprocy = %d\n", nprocx, nprocy);
    
    // Return ipx and ipy. Suppose a process grid of 3x3 processes, then
    // rank 0 => ipx = 0, ipy = 0  rank 1 => ipx = 1, ipy = 0  rank 2 => ipx = 2, ipy = 0
    // rank 3 => ipx = 0, ipy = 1  rank 4 => ipx = 1, ipy = 1  rank 5 => ipx = 2, ipy = 1
    // rank 6 => ipx = 0, ipy = 2  rank 7 => ipx = 1, ipy = 2  rank 8 => ipx = 2, ipy = 2

    // Now we stick to our example of nprocs = 4, nprocx = 2, nprocy = 2
    // rank 0 => ipx = 0, ipy = 0  rank 1 => ipx = 1, ipy = 0
    // rank 2 => ipx = 0, ipy = 1  rank 3 => ipx = 1, ipy = 1

    int *proc_coords = find_proc_coords(rank,nprocx,nprocy);
    int ipx=proc_coords[0]; int ipy=proc_coords[1];

    printf("Rank %d, ipx = %d, ipy = %d\n", rank, ipx, ipy);


    // Find neighboring processes!
    // Suppose domain_nx = 100, domain_ny = 100

    int domain_nx = atoi(argv[3]);  // Number of gridpoints in the x direction
    int domain_ny = atoi(argv[4]);  // Number of gridpoints in the y direction

    if (rank == 0) printf("domain_nx = %d, domain_ny = %d\n", domain_nx, domain_ny);

    int subdomain_nx = domain_nx / nprocx;   // subdomain x-size without halos
    int subdomain_ny = domain_ny / nprocy;   // subdomain y-size without halos

    if (rank == 0) printf("subdomain_nx = %d, subdomain_ny = %d\n", subdomain_nx, subdomain_ny);

    // subdomain_nx = domain_nx / nprocx = 100 / 2 = 50
    // subdomain_ny = domain_ny / nprocy = 100 / 2 = 50

    int subdomain_mx = subdomain_nx + 2 * halo_width; // subdomain x-size with halos
    int subdomain_my = subdomain_ny + 2 * halo_width; // subdomain y-size with halos

    if (rank == 0) printf("subdomain_mx = %d, subdomain_my = %d\n", subdomain_mx, subdomain_my);

    // Terminology of halo regions: halo_width=1 => 1-wide halo region on each side of subdomain
    // It is like the padding size in convolutional neural networks

    // subdomain_mx = subdomain_nx + 2 * halo_width = 50 + 2 * 1 = 52
    // subdomain_my = subdomain_ny + 2 * halo_width = 50 + 2 * 1 = 52

    // Check compatibility of argv parameters!

    float data[subdomain_mx][subdomain_my];
    float d_data[subdomain_mx][subdomain_my];
    float data_true[subdomain_mx][subdomain_my];

    // data is the field in the subdomain, d_data is the right-hand side of the pde for the subdomain
    // data[52][52], d_data[52][52]

    // domain has extents 2π x 2π
    float xextent = 2 * pi;
    float yextent = 2 * pi;            

    // Set grid spacings dx, dy. Each grid point has a spacing of dx and dy
    
    dx=xextent / domain_nx; // dx = 2π / 100 = π/50

    dy=yextent / domain_ny; // dy = 2π / 100 = π/50

    if (rank == 0) printf("dx = %f, dy = %f\n", dx, dy);

    // These two arrays holding the physical x and y coordinates of the grid points within the subdomain with halos
    float x[subdomain_mx]; // x[52]
    float y[subdomain_my]; // y[52]
    
    // Populate grid coordinate arrays x,y (equidistant): 
    for (int ix = 0; ix < subdomain_mx; ix++) {
        x[ix] = (ipx * subdomain_nx - halo_width + ix + 0.5) * dx;
        // example: ix = 20 (ix in range [0, 51])
        // ipx = 1 (this changes for each process along x direction)
        // subdomain_nx = 50 (fixed for all processes)
        // halo_width = 1 (fixed for all processes)
        // dx = π/50 (fixed for all processes)
        // The +0.5 means that we want the center of the grid point as the coordinate for that grid point
        // x[20] = (1 * 50 - 1 + 20 + 0.5) * π/50 = (50 - 1 + 20 + 0.5) * π/50 = 69.5 * π/50
    }

    for (int iy = 0; iy < subdomain_my; iy++) {
        y[iy] = (ipy*subdomain_ny - halo_width + iy + 0.5) * dy;
        // example: iy = 20 (iy in range [0, 51])
        // ipy = 1 (this changes for each process along y direction)
        // subdomain_ny = 50 (fixed for all processes)
        // halo_width = 1 (fixed for all processes)
        // dy = π/50 (fixed for all processes)
        // The +0.5 means that we want the center of the grid point as the coordinate for that grid point
        // y[20] = (1 * 50 - 1 + 20 + 0.5) * π/50 = (50 - 1 + 20 + 0.5) * π/50 = 69.5 * π/50
    }

    // Initialize data and d_data arrays
    for (int i = 0; i < subdomain_mx; ++i) {
        for (int j = 0; j < subdomain_my; ++j) {
            data[i][j] = 0.0; // Initialize with 0.0 or another appropriate value
            d_data[i][j] = 0.0;
            data_true[i][j] = 0.0;
        }
    }

    // Initialisation of field in data: harmonic function in x (can be modified to a harmonic in y or x and y):
    for (int ix = halo_width; ix < halo_width + subdomain_nx; ++ix) {
        for (int iy = halo_width; iy < halo_width + subdomain_ny; ++iy) {
            // ix in range [1, 50]
            // iy in range [1, 50]
            // periodicity in x and y
            // data[20][20] = c_amp * sin(69.5 * x[20]) = 1 * 69.5 * π/50 = -0.94
            data[ix][iy] = c_amp * sin((double) x[ix]);
            data_true[ix][iy] = c_amp * sin((double) x[ix]) * sin((double) y[iy]);
        }
    }

    // Consider proper synchronization measures!

    // Think about convenient data types to access non-contiguous portions of array data!
    // The answer is MPI_Type_vector for fetching a column 
    // and MPI_Type_contiguous for fetching a row

    MPI_Win win; // MPI window object

    MPI_Aint size = sizeof(float) * subdomain_mx * subdomain_my; // size of the MPI window = 4 * 52 * 52 = 10816 bytes
    int disp_unit = sizeof(float);  // displacement unit = 4 bytes

    // Create the MPI window
    MPI_Win_create(data, size, disp_unit, MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    
    // MPI_Win_create(void *base, MPI_Aint size, int disp_unit,
    //                MPI_Info info, MPI_Comm comm, MPI_Win *win)

    // base: Initial address of window (choice).
    // size: Size of window in bytes (nonnegative integer).
    // disp_unit: Local unit size for displacements, in bytes (positive integer).
    // info: Info argument (handle).
    // comm: Communicator (handle).
    // win: Window object returned by the call (handle).


    unsigned int iterations = atoi(argv[5]);       // number of iterations=timesteps

    if (u_x==0 && u_y==0) {
      if (rank==0) printf("velocity=0 - no meaningful simulation!");
      exit(1);
    }

    // CFL condition for timestep:
    float dt = cdt*(u_x==0 ? (u_y==0 ? 0 : dy/abs(u_y)) : (u_y==0 ? dx/abs(u_x) : fmin(dx/abs(u_x),dy/abs(u_y))));

    if (rank==0) printf("dt= %f \n\n",dt);

    float t=0.;


    // Create Cartesian communicator
    int dims[2] = {nprocx, nprocy}; // dimensions of the grid
    int periods[2] = {1, 1}; // periodic boundary conditions in both dimensions

    // Variables for neighbor ranks
    int north_neighbor_rank, south_neighbor_rank, east_neighbor_rank, west_neighbor_rank;

    for (unsigned int iter = 0; iter < iterations; ++iter) {

        if (rank == 0) printf("\nIteration %u\n", iter);

        double start_comm, end_comm, start_comp, end_comp, total_comm, total_comp;

        // Call the print function for both matrices
        if (print_debug) {
            //if (rank == 0) { // Optionally, print only for a specific rank
                printf("\nIteration %u\n", iter);
                print_matrix("data", &data[0][0], subdomain_mx, subdomain_my);
                printf("\n");
                print_matrix("data_true", &data_true[0][0], subdomain_mx, subdomain_my);
                printf("\n");
                print_matrix("d_data", &d_data[0][0], subdomain_mx, subdomain_my);
                printf("\n\n");
            //}
        }


        // Get the data from neighbors!

        // Define MPI datatypes for a column
        MPI_Datatype column_type;

        // Define MPI datatypes for a row
        MPI_Datatype row_type;

        // In C, an MPI datatype is of type MPI_Datatype. 
        // When sending a message in MPI, the message length is expressed as a number of elements and not a number of bytes. 
        // Example: sending an array that contains 4 ints is expressed as a buffer containing 4 MPI_INT, not 8 or 16 bytes.
        // MPI_Datatype is a handle (like a pointer) that refers to a predefined type or a user-defined type.

        // MPI vector type for fetching a column
        MPI_Type_vector(subdomain_ny, 1, subdomain_mx, MPI_FLOAT, &column_type);
        // int MPI_Type_vector(int count, int blocklength, int stride, MPI_Datatype oldtype, MPI_Datatype * newtype)
    
        // count: number of blocks (nonnegative integer)
        // when we fetch a column from a 2D array, count would be the number of rows you want to include in the column.
        
        // blocklength: number of elements in each block (nonnegative integer)
        // In the case of a column datatype, since we are interested in only one element from each row 
        // (a single column), blocklength is set to 1.

        // stride: number of elements between start of each block (integer)
        // This means that to get to the next element in the column, the MPI routine would skip subdomain_mx=52 elements 
        // (the width of a row), reaching the next row's corresponding column element.
        
        // oldtype: old datatype (handle)
        // newtype: new datatype (handle) (output parameter)

        MPI_Type_commit(&column_type);
        // MPI_Type_commit must be called on user-defined datatypes before they may be used in communications.
        // int MPI_Type_commit(MPI_Datatype* datatype);

        // datatype: A pointer on the MPI_Datatype to commit.

        // MPI contiguous type for fetching a row
        MPI_Type_contiguous(subdomain_nx, MPI_FLOAT, &row_type);

        // MPI_Type_contiguous creates an MPI datatype by replicating an existing one a certain number of times. 
        // These replications are created into contiguous locations, resulting in a contiguous data type created. 
        // The datatype created must be committed with MPI_Type_commit before it can be used in communications. 

        // int MPI_Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype * newtype)

        // count: The number of replications of the existing MPI datatype in the new one. (nonnegative integer)
        // oldtype: old datatype (handle)
        // newtype: new datatype (handle) (output parameter)

        // C uses row-major order for multi-dimensional arrays, meaning that the elements of a row are laid out consecutively in memory.
        
        MPI_Type_commit(&row_type);
        
        // Buffers to receive halo data
        // We know that the process coordinate is ipx, ipy  
        float recv_north_halo[subdomain_nx]; // This halo fetches the southmost row of the north neighbor (ipx, ipy + 1) 
        float recv_south_halo[subdomain_nx]; // This halo fetches the northrow of the south neighbor (ipx, ipy - 1)
        float recv_east_halo[subdomain_ny]; // This halo fetches the westmost column of the east neighbor (ipx + 1, ipy)
        float recv_west_halo[subdomain_ny]; // This halo fetches the eastmost column of the west neighbor (ipx - 1, ipy)

        // Displacements for each neighbor's halo data
        int displacement_north = subdomain_mx * (subdomain_my - halo_width - 1); // Start of southmost row of the north neighbor
        int displacement_south = 0; // First row of the south neighbor

        // For the east and west neighbors, we need to consider column-wise displacements
        int displacement_east = subdomain_nx; // Last column of the east neighbor
        int displacement_west = 0; // First column of the west neighbor

        // MPI_Get(void *origin_addr, int origin_count, MPI_Datatype origin_datatype, 
        //          int target_rank, MPI_Aint target_disp,
        //          int target_count, MPI_Datatype target_datatype, MPI_Win win)

        // Create a 2D Cartesian grid
        int dims[2] = {nprocx, nprocy};
        int periods[2] = {1, 1};  // Periodic boundaries
        // The periods array specifies whether the grid is periodic (wraps around) in each dimension. If the value is 1 (true), 
        // it indicates periodicity in that dimension, creating a torus-like wrap-around connection. If the value is 0 (false), 
        // it indicates no periodicity.

        MPI_Comm cart_comm; // Cartesian communicator

        // int MPI_Cart_create(MPI_Comm old_communicator, int dimension_number,
        //            const int* dimensions, const int* periods,
        //            int reorder, MPI_Comm* new_communicator);
        

        MPI_Cart_create(MPI_COMM_WORLD, 2, 
                        dims, periods, 
                        1, &cart_comm);

        // old_communicator: Communicator with which to associate the new Cartesian communicator (handle).
        // dimension_number: Number of dimensions of Cartesian grid (integer).
        // dimensions: Integer array of size ndims specifying the number of processes in each dimension.
        // periods: Logical array of size ndims specifying whether the grid is periodic (true) or not (false) in each dimension.
        // reorder: Ranking may be reordered (true) or not (false) (logical).
        // new_communicator: Communicator with new Cartesian topology (handle).

        // MPI_Cart_create creates a communicator from the cartesian topology information passed.

        // Variables to store the Cartesian topology information
        int coords[2];

        // Get the Cartesian topology information for the current process
        MPI_Cart_get(cart_comm, 2, dims, periods, coords);

        // Now, coords[0] and coords[1] hold the x and y coordinates of the current process in the grid
        int my_x_coord = coords[0];
        int my_y_coord = coords[1];


        // Determine the ranks of the neighboring processes
        int north_neighbor_rank, south_neighbor_rank, east_neighbor_rank, west_neighbor_rank;

        // MPI_Cart_shift virtually moves the cartesian topology of a communicator (created with MPI_Cart_create) 
        // in the dimension specified. It permits to find the two processes that would respectively reach, 
        // and be reached by, the calling process with that shift. Shifting a cartesian topology by 1 unit 
        // (the displacement) in a dimension therefore allows a process to find its neighbours in that dimension. 
        // In case no such neighbour exists, virtually located outside the boundaries of a non periodic dimension 
        // for instance, MPI_PROC_NULL is given instead.

        MPI_Cart_shift(cart_comm, 0, 1, 
                        &south_neighbor_rank, &north_neighbor_rank);

        MPI_Cart_shift(cart_comm, 1, 1, 
                        &west_neighbor_rank, &east_neighbor_rank);



        // int MPI_Cart_shift(MPI_Comm communicator, int direction, int displacement,
        //                    int* source, int* destination);
        
        // Parameters
        // communicator: The communicator concerned.
        // direction: the index of the dimension in which the shift will be made.
        // displacement: The number of units by which virtually move the topology.
        // source: The variable in which store the rank of the process that, given this shift, would reach the calling process.
        // destination: The rank of the process that, given this shift, would be reached by the calling process.


        // Fetch the halo data from each neighbor using MPI_Get
        // MPI_Get(void *origin_addr, int origin_count, MPI_Datatype
        //          origin_datatype, int target_rank, MPI_Aint target_disp,
        //          int target_count, MPI_Datatype target_datatype, MPI_Win win)

        // origin_addr: Initial address of origin buffer (choice).
        // origin_count: Number of entries in origin buffer (nonnegative integer).
        // origin_datatype: Data type of each entry in origin buffer (handle).
        // target_rank: Rank of target (nonnegative integer).
        
        // target_disp: Displacement from window start to the beginning of the target buffer (nonnegative integer).
        // For example, if you create a window exposing an array of double values, and you set disp_unit to sizeof(double), 
        // then a displacement value of 1 in an RMA operation refers to the next double in the array. If you set disp_unit to 1 (byte), 
        // a displacement of 1 would refer to the next byte in the memory region. 

        // Therefore, the target displacement in Get, Put and Accumulate operations is in integer, 
        // which refers to the number of disp_unit defined by MPI_Win_create
        
        // target_count: Number of entries in target buffer (nonnegative integer).
        // target_datatype: datatype of each entry in target buffer (handle)
        // win: window object used for communication (handle)

        // Each process only need to update their own subdomain in subdomain_nx and ny.

        // However, we need the halos to update the grid points at the edge/corner of the subdomain for each process for PDE. 
        // Because the data of the halo is the edge/corner of another subdomain, we need to use MPI_Get to obtain the data of the halo from another process 

        // Start timing for communication
        start_comm = MPI_Wtime();
        // Start the RMA epoch
        // Perform an MPI fence synchronization on a MPI window
        MPI_Win_fence(0, win);

        // int MPI_Win_fence(int assert, MPI_Win win)

        // assert: program assertion (integer)
        // win: window object (handle)

        // Start computation that does not depend on the received halo data
        // For example, compute RHS for the interior points of the subdomain
        int interior_xrange[2] = {halo_width + 1, halo_width + subdomain_nx - 1};
        int interior_yrange[2] = {halo_width + 1, halo_width + subdomain_ny - 1};

        float start_inner_comp = MPI_Wtime();
        rhs(interior_xrange, interior_yrange, subdomain_my, data, d_data);
        float end_inner_comp = MPI_Wtime();
        float total_inner_comp = end_inner_comp - start_inner_comp;


        MPI_Get(recv_north_halo, 1, row_type, 
                north_neighbor_rank, displacement_north, 
                1, row_type, win);
                
        MPI_Get(recv_south_halo, 1, row_type, 
                south_neighbor_rank, displacement_south, 
                1, row_type, win);

        MPI_Get(recv_east_halo, 1, column_type, 
                east_neighbor_rank, displacement_east, 
                1, column_type, win);

        MPI_Get(recv_west_halo, 1, column_type, 
                west_neighbor_rank, displacement_west, 
                1, column_type, win);

        // End the RMA epoch
        MPI_Win_fence(0, win);

        // Communication code here
        end_comm = MPI_Wtime();

        // Data is being fetched from the memory window (win) exposed by the process with rank west_neighbor_rank. 
        // This rank corresponds to the MPI process located to the west (left) of the current process in 
        // the Cartesian grid.

        //  The fetched data is stored in the local buffer recv_west_halo on the process executing this MPI_Get call. 
        // The datatype column_type is used for both the origin (local buffer) (recv_west_halo) and the target (remote buffer)

        // The target buffer is the location in the target process's memory from which data is fetched. 
        //The location within this buffer is specified by displacement_west.

        // Now that halo data is available, compute RHS for the boundary points of the subdomain
        int boundary_xrange[2] = {halo_width, halo_width + subdomain_nx};
        int boundary_yrange[2] = {halo_width, halo_width + subdomain_ny};
        rhs(boundary_xrange, boundary_yrange, subdomain_my, data, d_data);

        if (iter == 0 && print_debug){
            printf("Rank %d north_neighbor_rank: %d\n", rank, north_neighbor_rank);
            printf("Rank %d south_neighbor_rank: %d\n", rank, south_neighbor_rank);
            printf("Rank %d east_neighbor_rank: %d\n", rank, east_neighbor_rank);
            printf("Rank %d west_neighbor_rank: %d\n", rank, west_neighbor_rank);
        }

        if (print_debug){

            
            // Print the contents of halos
            printf("Rank %d recv_north_halo: ", rank);
            for (int i = 0; i < subdomain_nx; i++) {
                printf("%f ", recv_north_halo[i]);
            }
            printf("\n");

            printf("Rank %d recv_south_halo: ", rank);
            for (int i = 0; i < subdomain_nx; i++) {
                printf("%f ", recv_south_halo[i]);
            }
            printf("\n");

            printf("Rank %d recv_east_halo: ", rank);
            for (int i = 0; i < subdomain_ny; i++) {
                printf("%f ", recv_east_halo[i]);
            }

            printf("\n");

            printf("Rank %d recv_west_halo: ", rank);
            for (int i = 0; i < subdomain_ny; i++) {
                printf("%f ", recv_west_halo[i]);
            }
        }

        // Compute rhs. Think about concurrency of computation and data fetching by MPI_Get!

        // // Update the top (north) and bottom (south) halo regions
        // for (int ix = halo_width; ix < subdomain_nx + halo_width; ++ix) {
        //     data[ix][0] = recv_north_halo[ix - 1]; // Top halo updated with data from north neighbor
        //     data[ix][subdomain_my - 1] = recv_south_halo[ix]; // Bottom halo updated with data from south neighbor
        //     printf("Rank %d, north, data[%d][0] = %f\n", rank, ix + 1, recv_north_halo[ix]);
        //     printf("Rank %d, south, data[%d][%d] = %f\n", rank, ix + 1, subdomain_my - 1, recv_south_halo[ix]);

        // }

        // // Update the left (west) and right (east) halo regions
        // for (int iy = 0; iy < subdomain_ny; ++iy) {
        //     data[0][iy + halo_width] = recv_west_halo[iy]; // Left halo updated with data from west neighbor
        //     printf("Rank %d, west, data[0][%d] = %f\n", rank, iy + 1, recv_west_halo[iy]);
        //     data[subdomain_mx - halo_width][iy + halo_width] = recv_east_halo[iy]; // Right halo updated with data from east neighbor
        //     printf("Rank %d, east, data[%d][%d] = %f\n", rank, subdomain_mx - 1, iy + 1, recv_east_halo[iy]);
        // }

        for (int ix = halo_width; ix < subdomain_mx - halo_width; ++ix) {
            data[ix][0] = recv_north_halo[ix - halo_width];  // Update north halo
            data[ix][subdomain_my - 1] = recv_south_halo[ix - halo_width];  // Update south halo
        }

        // Update left and right halos
        for (int iy = halo_width; iy < subdomain_my - halo_width; ++iy) {
            data[0][iy] = recv_west_halo[iy - halo_width];  // Update west halo
            data[subdomain_mx - 1][iy] = recv_east_halo[iy - halo_width];  // Update east halo
        }




        // Compute the RHS for the subdomain
        int xrange[2] = {halo_width, halo_width + subdomain_nx};
        int yrange[2] = {halo_width, halo_width + subdomain_ny};
        
        // Start timing for computation
        float start_outer_comp = MPI_Wtime();

        // This function updates the d_data array with the right-hand side of the PDE
        rhs(xrange, yrange, subdomain_my, data, d_data);

        // End timing for computation
        float end_outer_comp = MPI_Wtime();

        float total_outer_comp = end_outer_comp - start_outer_comp;

        // Update field in data using rhs in d_data (Euler's method):

        // Calculate the start and stop indices for the interior points (excluding halos)

        // Update field in data using rhs in d_data (Euler's method)

        int ixstart = halo_width;
        int ixstop = halo_width + subdomain_nx;
        int iystart = halo_width;
        int iystop = halo_width + subdomain_ny;

        float local_total_error = 0.0;
        for (int ix = ixstart; ix < ixstop; ++ix) {
            for (int iy = iystart; iy < iystop; ++iy) {
                data[ix][iy] += dt * d_data[ix][iy];

                // Adjust x and y for periodic boundary conditions and advection
                float adjusted_x = fmod(x[ix] - u_x * iter + xextent, xextent);
                float adjusted_y = fmod(y[iy] - u_y * iter + yextent, yextent);

                // Update the analytical solution at each grid point
                data_true[ix][iy] = c_amp * sin(adjusted_x) * sin(adjusted_y);

                // Compute the error at this grid point
                float error = fabs(data[ix][iy] - data_true[ix][iy]);
                local_total_error += error;
            }
        }

        // Average error for this iteration (if needed)
        local_total_error /= ((ixstop - ixstart) * (iystop - iystart));

        float global_total_error = 0.0;
        MPI_Reduce(&local_total_error, &global_total_error, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);


        if (rank == 0) {
            printf("Global total error: %f\n", global_total_error);
        }

        t += dt;


        total_comm = end_comm - start_comm;
        total_comp = total_inner_comp + total_outer_comp;


        // Gather and print the results
        double global_comm, global_comp;
        MPI_Reduce(&total_comm, &global_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&total_comp, &global_comp, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Total communication time: %f seconds\n", global_comm);
            printf("Total computation time: %f seconds\n", global_comp);
        }


        // Clean up custom MPI data types
        MPI_Type_free(&column_type);
        MPI_Type_free(&row_type);

        // Output solution for checking/visualisation with choosable cadence!
    }

    // Finalize timing!
    MPI_Win_free(&win); // Clean up the MPI Window
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}


    //    // Populate the top (north) and bottom (south) halo regions from MPI_Get data
    //     for (int ix = 0; ix < subdomain_mx; ++ix) {
    //         data[ix][0] = data[ix][subdomain_my - 2 * halo_width]; // Top halo from bottom of the domain
    //         data[ix][subdomain_my - 1] = data[ix][halo_width]; // Bottom halo from top of the domain
    //     }

    //     // Populate the left (west) and right (east) halo regions
    //     for (int iy = 0; iy < subdomain_my; ++iy) {
    //         data[0][iy] = data[subdomain_mx - 2 * halo_width][iy]; // Left halo from right of the domain
    //         data[subdomain_mx - 1][iy] = data[halo_width][iy]; // Right halo from left of the domain
    //     }