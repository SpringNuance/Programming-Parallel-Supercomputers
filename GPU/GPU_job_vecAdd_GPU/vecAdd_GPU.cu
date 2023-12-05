#include <stdio.h>
#include <stdlib.h>

//Program that adds two n-vectors together, and reverts the indices
//of the resulting array on one GPU device.
//Demonstrates the very basic CUDA functionalities, e.g., how to
//launch kernels, transfer memory in between the host and device
//on the default stream, how to use dynamic shared mem.
__global__
void vecAddKernel(float* A, float* B, float* C, int n) { 

int i = blockDim.x*blockIdx.x + threadIdx.x; 

if(i<n) C[i] = A[i] + B[i]; 

} 

__global__
void
gpuswap(float *sarr, int count)
{	

extern __shared__ float smemarr [];

int i = threadIdx.x;
int rev = count - 1 - i;

smemarr[i]=sarr[i];
__syncthreads();
sarr[i]=smemarr[rev];
	
}
	

void vecAdd(float* h_A, float* h_B, float* h_C, float* hrev_C, int n) // h refers to host
     {

	int size = n * sizeof(float);

	float *d_A, *d_B, *d_C; //Pointers to device mem, hence start with d_ 

	cudaMalloc((void **) &d_A, size); // Allocating device mem	
	cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice); //Copying data over to device mem

	cudaMalloc((void **) &d_B, size); // Same stuff for B
	cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);

	cudaMalloc((void **) &d_C, size); // Allocation C ll hold the result
	
	vecAddKernel<<<256,256>>>(d_A,d_B,d_C,n);

	cudaMemcpy(h_C,d_C,size,cudaMemcpyDeviceToHost); //Copying result to host
        gpuswap<<<1,n,size>>>(d_C,n);

        cudaMemcpy(hrev_C,d_C,size,cudaMemcpyDeviceToHost); // Copying reversed

	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

}

int
main(void)
{
  // Check n:o GPUs
  int num_devices;
  cudaGetDeviceCount(&num_devices);
  printf("Num devices: %d\n", num_devices);

  // Try different lengths of the vectors. The code stops to work
  // at some point; why?
  int n=1000;
  float h_A[n],h_B[n],h_C[n],hrev_C[n];

  int maxn=100, minn=10;

  for (int i = 0; i < n ;i++) {
    h_A[i] = 1.0*(rand() % (maxn + 1 - minn) + minn);
    h_B[i] = 1.0*(rand() % (maxn + 1 - minn) + minn);
  }

  vecAdd(h_A,h_B,h_C,hrev_C,n);

  printf("Computed vector sum of two %d-vectors\n",n);  
  printf("Then I reverted indices in the vector...\n just for fun, checking first and last\n  %f %f %f %f\n",h_C[0],hrev_C[0],h_C[n-1],hrev_C[n-1]);  
}

