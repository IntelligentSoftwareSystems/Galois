/* -*- mode: c++ -*- */
#include <cuda.h>
#include <stdio.h>
#include "hpr_cuda.h"
#include "gg.h"
#include "cuda_mtypes.h"

__global__ void test_cuda_too(void) {
  printf("hello from the GPU!\n");
}

void load_graph_CUDA(MarshalGraph &g) {
  CSRGraphTy x;
  x.nnodes = g.nnodes;

  printf("load_graph_GPU: %d\n", x.nnodes);
  
}

void test_cuda(void) {
  printf("hello from cuda!\n");
  test_cuda_too<<<1,1>>>();
  check_cuda(cudaDeviceSynchronize());
}
