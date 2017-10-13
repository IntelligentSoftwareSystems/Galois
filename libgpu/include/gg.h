/*
   gg.h

   Implements the main GG header file. Part of the GGC source code. 

   Copyright (C) 2014--2016, The University of Texas at Austin

   See LICENSE.TXT for copyright license.

   Author: Sreepathi Pai <sreepai@ices.utexas.edu> 

   TODO: RTLICENSE
*/

#ifndef GALOIS_GPU
#define GALOIS_GPU

#include <fstream>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cassert>

#ifndef GGDEBUG
#define GGDEBUG 0
#endif 

#define dprintf	if (debug) printf
unsigned const debug = GGDEBUG;

#include "Timer.h"

static void check_cuda_error(const cudaError_t e, const char *file, const int line)
{
  if (e != cudaSuccess) {
    fprintf(stderr, "%s:%d: %s (%d)\n", file, line, cudaGetErrorString(e), e);
    exit(1);
  }
}

template <typename T>
static void check_retval(const T retval, const T expected, const char *file, const int line) {
  if(retval != expected) {
    fprintf(stderr, "%s:%d: Got %d, expected %d\n", file, line, retval, expected);
    exit(1);
  }
}

inline static __device__ __host__ int roundup(int a, int r) {
  return ((a + r - 1) / r) * r;
}

inline static __device__ __host__ int GG_MIN(int x, int y) {
  if(x > y) return y; else return x;
}

#define check_cuda(x) check_cuda_error(x, __FILE__, __LINE__)
#define check_rv(r, x) check_retval(r, x, __FILE__, __LINE__)

#include "bmk2.h"
#include "csr_graph.h"
#include "sharedptr.h"
#include "worklist.h"
#include "aolist.h"
#include "lockarray.h"
#include "abitset.h"
#include "gbar.cuh"
#include "cuda_launch_config.hpp"
#include "pipe.h"
#include "exclusive.h"
#include "internal.h"
#include "rv.h"
#include "failfast.h"
#include "ggc_rt.h"
#include "instr.h"

#include <util/mgpucontext.h>

extern mgpu::ContextPtr mgc;
#endif 
