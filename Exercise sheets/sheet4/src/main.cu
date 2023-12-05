#include <stdlib.h>
#include <stdio.h>
#include <algorithm>
#include <cuda_runtime_api.h>
#if DISTRIBUTED
    #include "quicksort_distributed.cuh"
    #include <mpi.h>
#else
    #include "quicksort.cuh"
#endif
bool is_sorted(float* data, int n){
    for(int i=1;i<n;i++){
        if(data[i]<data[i-1]){
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]){
    int rank;
    #if DISTRIBUTED
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    #else
        rank = 0;
    #endif
    constexpr int size = 20000;
    bool multi = false;
    float data[size];
    float* c_data = (float*)malloc(size*sizeof(float));
    float* g_data;
    float* result;
    float data_gpu[size];

    srand(12345678);
    for(int i=0;i<size;i++){
        data[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
    #if GPU
        cudaMalloc((void**)&g_data,size*sizeof(float));
        cudaDeviceSynchronize();
       
        cudaMemcpy(g_data,data,size*sizeof(float),cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        result = data_gpu;
        #if DISTRIBUTED 
            quicksort_distributed(data[0],0,size,g_data,MPI_COMM_WORLD);
            //quicksort(data[0],0,size,g_data);
        #else
            quicksort(data[0],0,size,g_data);
        #endif
        cudaDeviceSynchronize();
        cudaMemcpy(data_gpu,g_data,size*sizeof(float),cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        cudaFree(g_data);
    #else
        for(int i=0;i<size;i++) c_data[i] = data[i];
        #if DISTRIBUTED
            quicksort_distributed(data[0],0,size,c_data,MPI_COMM_WORLD);
        #else
            quicksort(data[0],0,size,c_data);
        #endif
        result = c_data;
    #endif
    
    std::stable_sort(data,data+size);
    bool sorted = is_sorted(result,size);
    if(sorted){
        printf("Is sorted at rank %d\n", rank);
    }else{
        printf("Not sorted at rank %d!!\n", rank);
    }
    bool is_correct = true;
    for(int i=0;i<size;i++) is_correct &= (result[i] == data[i]);
    if(is_correct){
        printf("Correct at rank: %d\n",rank);
    }else{
        printf("Incorrect at rank %d!!\n",rank);
    }
    #if DISTRIBUTED
	MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
    #endif
    return 0;

}

