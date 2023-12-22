#include <stdio.h>
#include <cuda_runtime_api.h>
#define NUM_BANKS 32 
#define LOG_NUM_BANKS 5 
#define CONFLICT_FREE_OFFSET(n)  ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS)) 

constexpr int threads_per_block = 256;

static int* ltbuffer;
static int* eqbuffer;
static int* gtbuffer;

static int* lt_outbuffer;
static int* eq_outbuffer;
static int* gt_outbuffer;

static int* sumbuffer;
static int* sum_outbuffer;

static float* data_outbuffer;





__global__ void assign(float* data_out, float* data_in, int* lt, int* eq, int* gt, int lt_sum, int eq_sum, int gt_sum, float pivot, int n) { 
    //Assigns elems from data_in to data_out to perform the partition

    int thid = threadIdx.x + blockDim.x*blockIdx.x; 
    if(thid >= n) return;

    //branchless programming technique where 
    //if the output of a branch is a scalar value can 
    //instead do a similar sum expression of different conditions
    //used for CPUs to avoid branch misses, but especially useful with GPUs to avoid warp divergence!

    int out_index = (data_in[thid]<pivot)*lt[thid]
                  + (data_in[thid] == pivot)*(eq[thid] + lt_sum)
                  + (data_in[thid] > pivot)*(gt[thid] + lt_sum+eq_sum);
    data_out[out_index] = data_in[thid];
}
__global__ void populate_tables(float* data, int* lt, int* eq, int* gt, float pivot, int n) { 
    int thid = threadIdx.x + blockDim.x*blockIdx.x; 
    if(thid >= n) return;
    lt[thid] = data[thid] < pivot;
    eq[thid] = data[thid] == pivot;
    gt[thid] = data[thid] > pivot;
}
__global__ void add_sums(int* data, int* sums, int n){
    //Add the sum of the blocks previous to this one to get the block global prefix sums
    int thid = threadIdx.x;
    int start = 2*blockDim.x*blockIdx.x;
    if(start+2*thid<n) data[start+2*thid] += sums[blockIdx.x];
    if(start+2*thid+1<n) data[start+2*thid+1] += sums[blockIdx.x];
}
__global__ void prescan(int *g_odata, int *g_idata, int size, int* sums) { 
    //Calculates the local prefix sum and sum of a single block
    //The block length has to be a power of two
   
    extern __shared__ int temp[];  
    int thid = threadIdx.x; 
    //Each block takes care of 2*blockDim.x elems
    //Is more efficient compared to handling blockDim.x elems since
    //then half of the threads would be idle after loading data to shared memory

    //This could be taken further like each block handling k*blockDim.x elems
    //This would be more effient due to Brent's theorem as stated in
    //https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda

    int n = 2*blockDim.x;
    
    //Each block starts with an offset to the array
    int start = n*blockIdx.x;

    //Load inputs into shared memory
    //Is free of bank-conflicts since each warp loads continuos range of elements
    int ai = thid;
    int bi = thid + (n/2); 
    if(start+ai >= size){
        //If we are passed the array simply load a padded value of zero
        temp[ai] = 0;
    } else{
        temp[ai] = g_idata[start+ai];
    }
    if(start+bi >= size){
        //If we are passed the array simply load a padded value of zero
        temp[bi] = 0;
    } else{
        temp[bi] = g_idata[start+bi]; 
    }
    __syncthreads();

    //Do the upsweep phase of Blelloch's algorihm
    //https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
    int offset = 1; 
    for (int d = n>>1; d > 0; d >>= 1)
    { 
        __syncthreads();
        if(thid < d)    { 
            int ai = offset*(2*thid+1)-1;     
            int bi = offset*(2*thid+2)-1;
            //Add padding to indexes to remove bank conflicts
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            temp[bi] += temp[ai];    
        }    
        offset *= 2;
    } 
    __syncthreads();

    //Do the downsweep phase of Blelloch's algorihm
    //https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
    if (thid == 0) { temp[n - 1 + CONFLICT_FREE_OFFSET(n-1)] = 0; } // clear the last element
    for (int d = 1; d < n; d *= 2) // traverse down tree & build scan 
    {      
        offset >>= 1;      
        __syncthreads();      
        if (thid < d){ 
             int ai = offset*(2*thid+1)-1;     
             int bi = offset*(2*thid+2)-1; 
            //Add padding to indexes to remove bank conflicts
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
              int t = temp[ai]; 
              temp[ai] = temp[bi]; 
              temp[bi] += t;       
        } 
    }  
    __syncthreads(); 

    //If not past the array write the local prefix sum out
    if(start+ai<size) g_odata[start+ai] = temp[ai]; // write results to device memory      
    if(start+bi<size) g_odata[start+bi] = temp[bi]; 
     
     //last thread writes the sum out
     if(thid == blockDim.x-1){
        if(start+n-1<size) sums[blockIdx.x] = temp[n-1] + g_idata[start+n-1];
        //If this is the last block have to read from the end of the array
        else sums[blockIdx.x] = temp[n-1] + g_idata[size-1];
     }
}
void
prescan_host(int* data, int* data_out, int n){
    //Each block handles 2*blockDim.x elems
    int elems_in_block = 2*threads_per_block;
    int num_blocks = (n+elems_in_block-1)/elems_in_block;

    int* sums = sumbuffer;
    cudaDeviceSynchronize();
    //Calculate block locals prefix sums
    prescan<<<num_blocks, threads_per_block, elems_in_block*sizeof(float), 0>>>(data_out,data, n, sums);
    cudaDeviceSynchronize();
    //If there were more than one block have to add block prefix sums to get global prefix sums
    if(num_blocks>1){
        int* sums_out = sum_outbuffer;
        cudaDeviceSynchronize();
        //Getting the block prefix sums is a prefix sum operation so we recurse
        prescan_host(sums, sums_out, num_blocks);
        cudaDeviceSynchronize();
        add_sums<<<num_blocks, threads_per_block, 0, 0>>>(data_out, sums_out, n);
        cudaDeviceSynchronize();
    }

}

void 
partition(float* &data, int start, int end, int &belows_end, int &above_start, float& lower_pivot, float& upper_pivot, float pivot)
{
    //Partitions the range data[start-end] in a stable way into upper part
    //such that for elem>pivot for all elem in the upper part
    //and into a lower part such that elem<pivot

    int size = end-start;
    int n = size;

    //Result buffers for tables
    int* lt_out;
    int* eq_out;
    int* gt_out;
    float* data_out = &data_outbuffer[start];

    int* lt = ltbuffer;
    int* eq = eqbuffer;
    int* gt = gtbuffer;

    lt_out  = lt_outbuffer;
    eq_out  = eq_outbuffer;
    gt_out  = gt_outbuffer;



    //Store for each elem is it less, equal or greater than the pivot
    populate_tables<<<(size+threads_per_block-1)/threads_per_block,threads_per_block,0,0>>>(&data[start],lt,eq,gt,pivot,size);
    cudaDeviceSynchronize();

    //Calculate prefix sums needed to perform the partition
    prescan_host(lt,lt_out, n);
    cudaDeviceSynchronize();
    prescan_host(eq,eq_out, n);
    cudaDeviceSynchronize();
    prescan_host(gt,gt_out, n);
    cudaDeviceSynchronize();

    //Get the global sums non-prefix sums
    int lt_last;
    int eq_last;
    int gt_last;

    float last_elem;

    cudaMemcpy(&lt_last,&lt_out[size-1],1*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(&eq_last,&eq_out[size-1],1*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(&gt_last,&gt_out[size-1],1*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_elem,&data[end-1],1*sizeof(float),cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    int lt_sum = lt_last + (last_elem < pivot);
    int eq_sum = eq_last + (last_elem == pivot);
    int gt_sum = gt_last + (last_elem > pivot);

    //Partitions elements in a similar manner to the singlethreaded version
    assign<<<(size+threads_per_block-1)/threads_per_block,threads_per_block,0,0>>>(data_out,&data[start],lt_out,eq_out,gt_out,lt_sum,eq_sum,gt_sum,pivot,size);
    cudaDeviceSynchronize();


    //Recurse for range [start-index of last less than elem]
    belows_end = start+lt_sum;
    cudaMemcpy(&lower_pivot,&data_out[lt_sum-1],1*sizeof(float),cudaMemcpyDeviceToHost);

    //Recurse for range [first greater than elem - end]
    above_start= start+lt_sum+eq_sum;
    cudaMemcpy(&upper_pivot,&data_out[lt_sum+eq_sum],1*sizeof(float),cudaMemcpyDeviceToHost);

    //Update the range of the original array with the partitioned elems
    cudaMemcpy(&data[start],data_out,n*sizeof(float),cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
}
void quicksort_recursive(float pivot, int start, int end, float* &data)
{
    //Base case only a single element can return 
    if(end-start<2) return;

    //get new pivots and ranges from the partition
    float lower_pivot;
    float upper_pivot;
    int belows_end;
    int above_start;
    partition(data,start,end,belows_end,above_start,lower_pivot,upper_pivot,pivot);
    
    //recurse
    quicksort_recursive(lower_pivot,start,belows_end, data);
    quicksort_recursive(upper_pivot,above_start,end, data);

}
void quicksort(float pivot, int start, int end, float* &data)
{
        //Separate non recursive call to do initialization
        int size = end-start;

        //We allocate helper buffers that are used during the quicksort
        //This is more efficient than constantly allocating and freeing new memory
        cudaMalloc((void**)&ltbuffer,size*sizeof(int));
        cudaMalloc((void**)&eqbuffer,size*sizeof(int));
        cudaMalloc((void**)&gtbuffer,size*sizeof(int));

        cudaMalloc((void**)&lt_outbuffer,size*sizeof(int));
        cudaMalloc((void**)&eq_outbuffer,size*sizeof(int));
        cudaMalloc((void**)&gt_outbuffer,size*sizeof(int));
        cudaMalloc((void**)&sumbuffer, sizeof(int) * (size+threads_per_block-1)/threads_per_block);
        cudaMalloc((void**)&sum_outbuffer, sizeof(int) * (size+threads_per_block-1)/threads_per_block);
        cudaMalloc((void**)&data_outbuffer,size*sizeof(float));
        cudaDeviceSynchronize();

        //Start the sorting
        quicksort_recursive(pivot,start,end,data);
        cudaDeviceSynchronize();


        //Free allocated memory from the GPU
        cudaFree(ltbuffer);
        cudaFree(eqbuffer);
        cudaFree(gtbuffer);

        cudaFree(lt_outbuffer);
        cudaFree(eq_outbuffer);
        cudaFree(gt_outbuffer);

        cudaFree(sumbuffer);
        cudaFree(sum_outbuffer);
}
