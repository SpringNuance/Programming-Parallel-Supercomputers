#include <mpi.h>
#include <iostream>
#include <cstdlib>

void swap(float* arr, int i, int j) {
    float t = arr[i];
    arr[i] = arr[j];
    arr[j] = t;
}

void quicksort(float* arr, int start, int end) {
    if (end <= start + 1)
        return;
    
    float pivot = arr[start + (end - start) / 2];
    swap(arr, start, start + (end - start) / 2);
    int index = start;

    for (int i = start + 1; i < end; ++i) {
        if (arr[i] < pivot) {
            index++;
            swap(arr, i, index);
        }
    }
    swap(arr, start, index);
    quicksort(arr, start, index);
    quicksort(arr, index + 1, end);
}

void merge(float* arr1, int n1, float* arr2, int n2, float* result) {
    int i = 0, j = 0, k = 0;
    while (i < n1 && j < n2) {
        if (arr1[i] < arr2[j]) {
            result[k++] = arr1[i++];
        } else {
            result[k++] = arr2[j++];
        }
    }
    while (i < n1) {
        result[k++] = arr1[i++];
    }
    while (j < n2) {
        result[k++] = arr2[j++];
    }
}

void quicksort_distributed(float pivot, int start, int end, float* &data, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int n = end - start;
    int local_n = n / size;
    float* local_data = new float[local_n];
    float* temp_data = new float[n];

    MPI_Scatter(data, local_n, MPI_FLOAT, local_data, local_n, MPI_FLOAT, 0, comm);

    quicksort(local_data, 0, local_n);

    MPI_Gather(local_data, local_n, MPI_FLOAT, temp_data, local_n, MPI_FLOAT, 0, comm);

if (rank == 0) {
    // Merge the subarrays in temp_data
    for (int i = 1; i < size; ++i) {
        float* merged_data = new float[(i + 1) * local_n];
        merge(temp_data, i * local_n, temp_data + i * local_n, local_n, merged_data);
        for (int j = 0; j < (i + 1) * local_n; ++j) {
            temp_data[j] = merged_data[j];
        }
        delete[] merged_data;
    }
    for (int i = 0; i < n; ++i) {
        data[start + i] = temp_data[i];
    }
}

// Broadcast the sorted data from the root process to all other processes
MPI_Bcast(data, n, MPI_FLOAT, 0, comm);

delete[] local_data;
delete[] temp_data;

}
