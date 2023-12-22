void 
partition(float* &data, int start, int end, int& belows_end, int &above_start, float& lower_pivot, float& upper_pivot, float pivot)
{
    //Partitions the range data[start-end] in a stable way into upper part
    //such that for elem>pivot for all elem in the upper part
    //and into a lower part such that elem<pivot

    int size = end-start;
    float* local_data = &data[start];

    //calculate the prefix sum of arrays that measure are the elems less, equal or greater than the pivo
    int lt[size];
    int eq[size];
    int gt[size];
    int lt_sum = 0;
    int eq_sum = 0;
    int gt_sum = 0;
    for(int i=0;i<size;i++){
        lt[i] = lt_sum;
        eq[i] = eq_sum;
        gt[i] = gt_sum;

        if(local_data[i]<pivot) lt_sum++;
        if(local_data[i]==pivot) eq_sum++;
        if(local_data[i]>pivot) gt_sum++;
    }

    //Based on the prefix sums we can partition the array in a stable manner.
    //Elements that are less come first, then that are equal and then those that are greater than the pivot
    //Since the prefix sums are monotonic this is a stable way to do the partition.
    float tmp[size];
    for(int i=0;i<size;i++){
        if(local_data[i]<pivot) tmp[lt[i]] = local_data[i];
        if(local_data[i]==pivot) tmp[eq[i] + lt_sum] = local_data[i];
        if(local_data[i]>pivot) tmp[gt[i] + lt_sum+eq_sum] = local_data[i];
    }
    memcpy(local_data,tmp,size*sizeof(float));

    //Recurse for range [start-index of last less than elem]
    belows_end= start+lt_sum;
    lower_pivot = local_data[lt_sum-1];
    
    //Recurse for range [first greater than elem - end]
    above_start = start+lt_sum+eq_sum;
    upper_pivot = local_data[lt_sum+eq_sum];
}
void quicksort(float pivot, int start, int end, float* &data)
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
    quicksort(lower_pivot,start,belows_end, data);
    quicksort(upper_pivot,above_start,end, data);

}
