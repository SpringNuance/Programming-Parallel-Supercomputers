#include <stdio.h>
#include <stdlib.h>

#if USE_MPI
#include <mpi.h>
#endif

#include "errchk.h"
#include "reduce.cuh"

#define NUM_SAMPLES (512)
#define MAX_COUNT ((1 * 1024 * 1024) / sizeof(int))
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))

static int
modelreduce(const int* arr, const size_t count)
{
  ERRCHK(count > 0);
  int max = arr[0];

  for (int i = 1; i < count; ++i)
    max = arr[i] > max ? arr[i] : max;

  return max;
}

static void
write_to_file(const char* path, const char* str)
{
  FILE* fp = fopen(path, "w");
  ERRCHK(fp);

  fprintf(fp, "%s\n", str);

  fclose(fp);
}

int
main(int argc, char* argv[])
{
  char* outfile = NULL;
  if (argc > 1)
    outfile = argv[1];

#if USE_MPI
  MPI_Init(NULL, NULL);
  int nprocs, pid;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
#else
  int num_devices;
  ERRCHK_CUDA(cudaGetDeviceCount(&num_devices));
  const int nprocs = num_devices;
  const int pid    = 0;
#endif

  srand(982357885u);
  for (size_t i = 0; i < NUM_SAMPLES; ++i) {
    // Set count a multiple of nprocs/devices for simplicity
    const size_t count = nprocs * (1 + (rand() % (MAX_COUNT / nprocs)));
    int* arr           = (int*)malloc(count * sizeof(arr[0]));
    for (size_t i = 0; i < count; ++i)
      arr[i] = rand();

    const int model = modelreduce(arr, count);

#if USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    const int candidate = reduce(arr, count);
#if USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    int failed = 0;
    if (pid == 0) {
      printf("Model: %d, Candidate: %d, Correct? %s\n", model, candidate,
             model == candidate ? "Yes" : "No");
      fflush(stdout);

      if (model != candidate) {
        fprintf(stderr,
                "Failure:  Model: %d, Candidate: %d, "
                "Correct? %s\n",
                model, candidate, model == candidate ? "Yes" : "No");
        if (outfile)
          write_to_file(outfile, "FAILURE");

        failed = 1;
      }
    }
    free(arr);

#if USE_MPI
    MPI_Bcast(&failed, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
    if (failed) {
      if (pid == 0 && outfile)
        write_to_file(outfile, "FAILURE");

#if USE_MPI
      MPI_Finalize();
#endif
      printf("pid %d exit failure\n", pid);
      return EXIT_FAILURE;
    }
  }

#if USE_MPI
  MPI_Finalize();
#endif

  if (outfile)
    write_to_file(outfile, "OK");

  printf("pid %d exit success\n", pid);
  return EXIT_SUCCESS;
}