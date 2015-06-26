/* -*- mode: c++ -*- */
#include <cuda.h>
#include <stdio.h>
#include "hpr_cuda.h"
#include "gg.h"

__global__ void test_cuda_too(void) {
  printf("hello from the GPU!\n");
}

void test_cuda(void) {
  CSRGraphTy x;

  printf("hello from cuda!\n");
  test_cuda_too<<<1,1>>>();
  check_cuda(cudaDeviceSynchronize());

  printf("%d\n", x.nnodes);
}
