/* -*- mode: c++ -*- */
#include <cuda.h>
#include "gg.h"

static struct ggc_rt_dev_info dinfo = {-1, -1};

void ggc_init_dev_info() {
  int dev;
  struct cudaDeviceProp p;

  check_cuda(cudaGetDevice(&dev));
  dinfo.dev = dev;
  
  check_cuda(cudaGetDeviceProperties(&p, dev));
  dinfo.nSM = p.multiProcessorCount;
}

void ggc_set_gpu_device(int dev) {
  check_cuda(cudaSetDevice(dev));
  ggc_init_dev_info();
}

int ggc_get_nSM() {
  if(dinfo.dev == -1)
    ggc_init_dev_info();

  return dinfo.nSM;
}
