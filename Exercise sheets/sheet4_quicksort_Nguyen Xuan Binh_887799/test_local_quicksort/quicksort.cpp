// #include <iostream>
// #include <cstdlib>

// // Utility function to swap two elements
// void swap(float &a, float &b) {
//     float temp = a;
//     a = b;
//     b = temp;
// }

// // Utility function to swap two elements in the index array
// void swapIndex(int &a, int &b) {
//     int temp = a;
//     a = b;
//     b = temp;
// }

// // Function to partition the array and return the pivot index
// int partition(float* data, int* index, int start, int end, float pivot) {
//     int i = start, j = end;
//     while (i <= j) {
//         while (i <= j && (data[i] < pivot || (data[i] == pivot && index[i] < index[start]))) {
//             i++;
//         }
//         while (i <= j && (data[j] > pivot || (data[j] == pivot && index[j] > index[start]))) {
//             j--;
//         }
//         if (i < j) {
//             swap(data[i], data[j]);
//             swapIndex(index[i], index[j]);
//             i++;
//             j--;
//         }
//     }
//     return i; // Return the partition point
// }

// // The quicksort function
// void quicksort(float pivot, int start, int end, float* &data) {
//     static int* index = nullptr;
    
//     // Initialize index array on the first call
//     if (index == nullptr && start == 0) {
//         index = new int[end + 1];
//         for (int i = 0; i <= end; i++) {
//             index[i] = i;
//         }
//     }

//     if (start < end) {
//         int pivotIndex = partition(data, index, start, end, pivot);
//         quicksort(data[start], start, pivotIndex - 1, data);
//         quicksort(data[pivotIndex], pivotIndex, end, data);
//     }

//     // Clean up index array after the top-level call
//     if (start == 0) {
//         delete[] index;
//         index = nullptr;
//     }
// }

// // Test the quicksort implementation
// int main() {
//     const int size = 10;  // Adjust size as needed
//     float* data = new float[size];

//     // Initialize data with random values
//     for (int i = 0; i < size; ++i) {
//         data[i] = static_cast<float>(rand()) / RAND_MAX;
//     }

//     // Call quicksort
//     quicksort(data[0], 0, size - 1, data);

//     // Verify and print sorted data
//     for (int i = 1; i < size; ++i) {
//         if (data[i - 1] > data[i]) {
//             std::cout << "Sort error at position " << i << std::endl;
//             break;
//         }
//     }

//     delete[] data;
//     return 0;
// }



#include <iostream>
int partition(int* arr, int start, int end)
{   
      // assuming last element as pivotElement
    int index = 0, pivotElement = arr[end], pivotIndex;
    int* temp = new int[end - start + 1]; // making an array whose size is equal to current partition range...
    for (int i = start; i <= end; i++) // pushing all the elements in temp which are smaller than pivotElement
    {
        if(arr[i] < pivotElement)
        {
            temp[index] = arr[i];
            index++;
        }
    }
 
    temp[index] = pivotElement; // pushing pivotElement in temp
    index++;
 
    for (int i = start; i < end; i++) // pushing all the elements in temp which are greater than pivotElement
    {
        if(arr[i] > pivotElement)
        {
            temp[index] = arr[i];
            index++;
        }
    }
  // all the elements now in temp array are order : 
  // leftmost elements are lesser than pivotElement and rightmost elements are greater than pivotElement
               
     
     
    index = 0;
    for (int i = start; i <= end; i++) // copying all the elements to original array i.e arr
    {   
        if(arr[i] == pivotElement)
        {
              // for getting pivot index in the original array.
              // we need the pivotIndex value in the original and not in the temp array
            pivotIndex = i;
        }
        arr[i] = temp[index];
        index++;
    }
    return pivotIndex; // returning pivotIndex
}
 
void quickSort(int* arr, int start, int end)
{  
    if(start < end)
    {   
        int partitionIndex = partition(arr, start, end); // for getting partition
        quickSort(arr, start, partitionIndex - 1); // sorting left side array
        quickSort(arr, partitionIndex + 1, end); // sorting right side array
    }
    return;
}

// Test the quicksort implementation
int main() {
    const int size = 200000;  // Adjust size as needed
    float* data = new float[size];

    // Initialize data with random values
    for (int i = 0; i < size; ++i) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Call quicksort
    quicksort(data[0], 0, size, data);

    // Verify and print sorted data
    for (int i = 1; i < size; ++i) {
        if (data[i - 1] > data[i]) {
            std::cout << "Sort error at position " << i << std::endl;
            break;
        }
    }

    std::cout << "Success!" << std::endl;

    // // Print sorted data
    // for (int i = 0; i < size; ++i) {
    //     std::cout << "Data[" << i << "]: " << data[i] << std::endl;
    // }

    delete[] data;
    return 0;
}
