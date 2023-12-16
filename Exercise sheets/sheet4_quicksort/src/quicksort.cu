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