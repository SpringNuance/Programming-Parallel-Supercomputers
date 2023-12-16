void quicksort(float pivot, int start, int end, float* &data)
{
/**
        Exercise 4: Your code here
        Input:
                pivot: a pivot value based on which to split the array in to less and greater elems
                start: starting index of the range to be sorted
                end: exclusive ending index of the range to be sorted
                data: array of floats allocated on the GPU to sort in range start till end
        Return:
                upon return the array range should be sorted
        Task:
                to sort the array using the idea of quicksort in a stable manner
                a sort is stable if it maintains the relative order of elements with equal values
		during the sorting try to keep the data movement of the CPU and GPU as small as possible
		and ideally only move scalars between them
	Hint:
		prefix sum is a beneficial primitive in this case that you are recommended to use:
		https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
**/

}
