#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <algorithm>

void quicksort(float pivot, int start, int end, float* &data) {
    // Adjust end to be the last valid index
    end--;

    if (start >= end) {
        return;
    }

    // Partitioning the array
    int left = start;
    int right = end;
    while (left <= right) {
        // Find a value greater than or equal to the pivot from the left
        while (left <= right && data[left] < pivot) {
            left++;
        }
        // Find a value smaller than the pivot from the right
        while (left <= right && data[right] > pivot) {
            right--;
        }
        // Swap if needed
        if (left <= right) {
            std::swap(data[left], data[right]);
            left++;
            right--;
        }
    }

    // Adjust the pivot for the next recursive calls
    if (start < right) {
        float newPivotLeft = data[start];
        quicksort(newPivotLeft, start, right + 1, data);
    }
    if (left < end) {
        float newPivotRight = data[left];
        quicksort(newPivotRight, left, end + 1, data);
    }
}



// Assume quicksort function is defined above this point

int main() {
    const int size = 20000;
    float* data = new float[size];
    float* data_model = new float[size]; // Duplicate array for comparison

    // Initialize the array with random floats
    srand(static_cast<unsigned int>(time(nullptr)));
    for (int i = 0; i < size; i++) {
        data[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX/100000.0));
        data_model[i] = data[i]; // Copy the same values to data_model
    }
    
    // Sort using quicksort
    quicksort(data[0], 0, size, data);

    // // print the sorted array
    // for (int i = 0; i < size; i++) {
    //     std::cout << data[i] << " ";
    // }

    std::cout << std::endl;
    
    // Sort using std::stable_sort for comparison
    std::stable_sort(data_model, data_model + size);

    // // print the sorted model array
    // for (int i = 0; i < size; i++) {
    //     std::cout << data_model[i] << " ";
    // }

    // Check for correctness by comparing the two arrays
    bool sorted = true;
    for (int i = 0; i < size; i++) {
        if (data[i] != data_model[i]) {
            sorted = false;
            break;
        }
    }

    std::cout << std::endl;

    if (sorted) {
        std::cout << "Array is sorted correctly." << std::endl;
    } else {
        std::cout << "Array is not sorted correctly." << std::endl;
    }

    delete[] data;
    delete[] data_model;

    return 0;
}




        
