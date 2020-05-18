#ifndef CHECKER_H
#define CHECKER_H
#include <cuda.h>
#include <cuda_runtime.h>

static void check_cuda_error(const cudaError_t e, const char* file,
                             const int line) {
  if (e != cudaSuccess) {
    fprintf(stderr, "%s:%d: %s (%d)\n", file, line, cudaGetErrorString(e), e);
    exit(1);
  }
}
#define check_cuda(x) check_cuda_error(x, __FILE__, __LINE__)

#endif
