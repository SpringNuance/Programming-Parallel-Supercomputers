#include <mpi.h>
//Copied from quicksort.cu //
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


// /////////
void quicksort_distributed(float pivot, int start, int end, float* &data,  MPI_Comm comm)
{
    if (end - start < 2) return;   // if only one element in range, no further sorting
    int rank;
    int nprocs;     // number of processes in local communicator comm
    MPI_Comm_rank(comm, &rank); 
    MPI_Comm_size(comm, &nprocs);

    // split about pivot at start
    int belows_end;
    int aboves_start;

    float lower_pivot;
    float upper_pivot;
    partition(data, start, end, belows_end, aboves_start, lower_pivot, upper_pivot, pivot);
    unsigned int ndata = belows_end == end ? aboves_start-start : belows_end-start;
    if (nprocs == 1)         // there is only one process for this range, sort locally -> recursion stops.
    {
        quicksort(lower_pivot,start,belows_end, data);
        quicksort(upper_pivot,aboves_start,end, data);
    }
    else
    {
        // more than one process available for sorting this range

        unsigned int belows_assigned_to = 0; // the process coordinating sorting of values below pivot
        unsigned int aboves_assigned_to = 1; // the process coordinating sorting of values above pivot

        // Split the group of processes into ~two halves and assign one half to
        // each split of the data, then recurse.
        unsigned int follow_aboves = rank % 2;

        MPI_Comm new_comm;
        MPI_Comm_split(comm, !follow_aboves, rank, &new_comm);


        if (!follow_aboves)         // executed by even ranks for belows subrange 
        {
            quicksort_distributed(lower_pivot, start, belows_end, data,  new_comm);
	     // Coordinating process of belows sends sorted belows to coordinating process of aboves
            // and receives sorted aboves.
            if (rank == belows_assigned_to) {
                MPI_Sendrecv(&data[start], ndata, MPI_FLOAT, aboves_assigned_to, rank,
                             &data[aboves_start], (end-aboves_start), MPI_FLOAT, aboves_assigned_to, aboves_assigned_to,
                             comm, MPI_STATUSES_IGNORE);
	    }
            // Finally, the coordinating rank broadcasts the sorted aboves
            // (just obtained by sendrecv) to its group
	    // -> sorted array now in all processes of group
	    MPI_Bcast(&data[aboves_start], end - aboves_start, MPI_FLOAT, 0, new_comm);  // 0 is root rank in each of the two communicators!

        }
        else
        {
            quicksort_distributed(upper_pivot, aboves_start, end, data, new_comm);
	     if (rank == aboves_assigned_to){
                MPI_Sendrecv(&data[aboves_start], (end-aboves_start), MPI_FLOAT, belows_assigned_to, rank,
                             &data[start], ndata, MPI_FLOAT, belows_assigned_to, belows_assigned_to,
                             comm, MPI_STATUSES_IGNORE);
	    }
	    // see if nest; nothing to broadcast if there was nothing to sort
	    if (ndata>0) MPI_Bcast(&data[start], ndata, MPI_FLOAT, 0, new_comm);    // 0 is root rank in each of the two communicators!


        }

        MPI_Comm_free(&new_comm);
    }
}
