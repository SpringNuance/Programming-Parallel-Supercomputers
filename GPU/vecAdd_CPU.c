#include <stdio.h>
#include <stdlib.h>

// CPU-program that adds two n-vectors together, and reverts the indices
// of the resulting array.

void vecAdd(float* h_A, float* h_B, float* h_C, int n) 
{
  for (int i = 0; i < n; i++)    
    h_C[i] = h_A[i] + h_B[i]; 
} 


void swap(float* arr, float* sarr, const int count)
{
  for (int i = 0; i < count; i++) {
    sarr[i]=arr[count-1-i];
  }
}

int main() { 

  int n=1000;
  float h_A[n], h_B[n], h_C[n], sh_C[n];
  int maxn=100, minn=10;

  for (int i = 0; i < n ;i++) {
    h_A[i] = 1.0*(rand() % (maxn + 1 - minn) + minn);
    h_B[i] = 1.0*(rand() % (maxn + 1 - minn) + minn);
  }

  vecAdd(h_A, h_B, h_C, n); 
  swap(h_C,sh_C,n);
  

  printf("Computed vector sum of two %d-vectors\n",n);  
  printf("Then I reverted indices in the vector...\n just for fun, checking first and last\n  %f %f %f %f",h_C[0],sh_C[0],h_C[n-1],sh_C[n-1]);  
}

