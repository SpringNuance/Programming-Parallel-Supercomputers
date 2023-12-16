#include <iostream>
#include <omp.h>

void prefix_sum_rec(float *x, float *s, int n) {
    if (n == 1) {
        s[0] = x[0];
    } else {
        float *y = new float[n/2];
        #pragma omp parallel for
        for (int i = 0; i < n/2; ++i) {
            y[i] = x[2*i] + x[2*i + 1];
        }
        float *z = new float[n/2];
        prefix_sum_rec(y, z, n/2);
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            if (i == 0) {
                s[0] = x[0];
            } else if (i % 2 == 1) {
                s[i] = z[i/2];
            } else {
                s[i] = z[(i-1)/2] + x[i];
            }
        }
        delete[] y;
        delete[] z;
    }
}

void prefix_sum(float *x, int n) {
    float *s = new float[n];
    prefix_sum_rec(x, s, n);
    for (int i = 0; i < n; ++i) {
        x[i] = s[i];
    }
    delete[] s;
}

int main() {
    const int n = 8;
    float arr[n] = {1, 2, 3, 4, 5, 6, 7, 8};

    std::cout << "Original array: ";
    for (int i = 0; i < n; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;

    prefix_sum(arr, n);

    std::cout << "Prefix sum array: ";
    for (int i = 0; i < n; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}




        
