#pragma once
#include <cuda_runtime_api.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ERROR(str)                                                             \
  {                                                                            \
    time_t terr;                                                               \
    time(&terr);                                                               \
    fprintf(stderr, "%s", ctime(&terr));                                       \
    fprintf(stderr, "\tError in file %s line %d: %s\n", __FILE__, __LINE__,    \
            str);                                                              \
    fflush(stderr);                                                            \
    exit(EXIT_FAILURE);                                                        \
    abort();                                                                   \
  }

#define ERRCHK(retval)                                                         \
  {                                                                            \
    if (!(retval))                                                             \
      ERROR(#retval " was false");                                             \
  }

static inline void
cuda_assert(cudaError_t code, const char* file, int line, bool abort)
{
  if (code != cudaSuccess) {
    time_t terr;
    time(&terr);
    fprintf(stderr, "%s", ctime(&terr));
    fprintf(stderr, "\tCUDA error in file %s line %d: %s\n", file, line,
            cudaGetErrorString(code));
    fflush(stderr);

    if (abort)
      exit(code);
  }
}

#define ERRCHK_CUDA(params)                                                    \
  {                                                                            \
    cuda_assert((params), __FILE__, __LINE__, true);                           \
  }

#define ERRCHK_CUDA_KERNEL()                                                   \
  {                                                                            \
    ERRCHK_CUDA(cudaPeekAtLastError());                                        \
    ERRCHK_CUDA(cudaDeviceSynchronize());                                      \
  }
