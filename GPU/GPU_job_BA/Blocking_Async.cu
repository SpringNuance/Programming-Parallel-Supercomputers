// Overlapping data transfers and communications example.

#include <stdio.h>

__global__ 
void compute_kernel(float *d_A, int offset)
{
  int i = offset + threadIdx.x + blockIdx.x*blockDim.x;
  float x = (float)i;
  float y = (float)gridDim.x;
  d_A[i]=1.-2.*(x/y)*(x/y) + (x/y)*(x/y)*(x/y); 
}

int main(int argc, char **argv)
{
  const int blockSize = 256, nStreams = 4;
  const int n = 4 * 1024 * blockSize * nStreams;
  const int streamSize = n / nStreams;
  const int streamBytes = streamSize * sizeof(float);
  const int bytes = n * sizeof(float);
   
  //Host memory; reserved in a "normal way".
  //If one does the mem alloc for host this way, the blocking scheme is faster
  //float *h_A = (float*)malloc(bytes);
  //Using pinned memory streams outperform in transfer speed; read more from
  //https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
  float *h_A;
  cudaMallocHost((void**)&h_A, bytes) ;
  // End pinned
  float *d_A;
  cudaMalloc((void**)&d_A, bytes) ; // device

  float timems; // elapsed time in milliseconds
  
  // create events and streams
  cudaEvent_t startEvent, stopEvent, dummyEvent;
  cudaStream_t stream[nStreams];
  cudaEventCreate(&startEvent) ;
  cudaEventCreate(&stopEvent) ;
  cudaEventCreate(&dummyEvent) ;
  for (int i = 0; i < nStreams; ++i)
    cudaStreamCreate(&stream[i]);
  
  // On one stream transfers should be blocking, and sequentialise the code.
  memset(h_A, 0, bytes);
  cudaEventRecord(startEvent,0);
  cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice) ;
  compute_kernel<<<n/blockSize, blockSize>>>(d_A, 0);
  cudaMemcpy(h_A, d_A, bytes, cudaMemcpyDeviceToHost) ;
  cudaEventRecord(stopEvent, 0) ;
  cudaEventSynchronize(stopEvent) ;
  cudaEventElapsedTime(&timems, startEvent, stopEvent) ;
  printf("Time for blocking transfers and computation (ms): %f\n", timems);

  // Let us try the same on streams, when transfers and computation should become concurrent and therefore more efficient
  memset(h_A, 0, bytes);
  cudaEventRecord(startEvent,0);
  for (int i = 0; i < nStreams; ++i) {
    int offset = i * streamSize;
    cudaMemcpyAsync(&d_A[offset], &h_A[offset], 
                               streamBytes, cudaMemcpyHostToDevice, 
                               stream[i]);
    compute_kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(d_A, offset);
    cudaMemcpyAsync(&h_A[offset], &d_A[offset], 
                               streamBytes, cudaMemcpyDeviceToHost,
                               stream[i]);
  }
  cudaEventRecord(stopEvent, 0);
  cudaEventSynchronize(stopEvent);
  cudaEventElapsedTime(&timems, startEvent, stopEvent);
  printf("Time for transfer and execution on %d streams (ms): %f\n", nStreams,timems);

  // Deallocate
  cudaEventDestroy(startEvent) ;
  cudaEventDestroy(stopEvent) ;
  cudaEventDestroy(dummyEvent) ;
  for (int i = 0; i < nStreams; ++i)
    cudaStreamDestroy(stream[i]) ;
  cudaFree(d_A);
  cudaFreeHost(h_A);

  return 0;
}
