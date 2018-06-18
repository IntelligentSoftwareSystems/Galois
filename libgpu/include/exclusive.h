/*
   exclusive.h

   Runtime implementation for Exclusive. Part of the GGC source code.

   Copyright (C) 2014--2016, The University of Texas at Austin

   See LICENSE.TXT for copyright license.

   Author: Sreepathi Pai <sreepai@ices.utexas.edu>
*/

#include "sharedptr.h"
#include <cassert>
#include <cub/cub.cuh>

#ifndef LOADCV
#define LOADCV(x) cub::ThreadLoad<cub::LOAD_CV>((x))
#endif

#ifndef LOADCG
#define LOADCG(x) cub::ThreadLoad<cub::LOAD_CG>((x))
#endif

#ifndef STORECG
#define STORECG(x, y) cub::ThreadStore<cub::STORE_CG>((x), (y))
#endif

class ExclusiveLocks {
public:
  Shared<int> locks; // need not be shared, GPU-only is fine.
  int* lk;
  int nitems;

  ExclusiveLocks() { nitems = 0; }

  ExclusiveLocks(size_t nitems) {
    this->nitems = nitems;
    locks.alloc(nitems);
    locks.cpu_wr_ptr();
    lk = locks.gpu_wr_ptr();
  }

  void alloc(size_t nitems) {
    // to be called once if default constructor was used
    assert(this->nitems == 0);
    locks.alloc(nitems);
    locks.cpu_wr_ptr();
    lk = locks.gpu_wr_ptr();
  }

  __device__ void mark_p1(int n, int* a, int id) {
    // try to claim ownership
    for (int i = 0; i < n; i++)
      STORECG(lk + a[i], id);
  }

  __device__ void mark_p1_iterator(int start, int n, int step, int* a, int id) {
    // try to claim ownership
    for (int i = start; i < n; i += step)
      STORECG(lk + a[i], id);
  }

  __device__ void mark_p2(int n, int* a, int id) {
    for (int i = 0; i < n; i++)
      if (LOADCG(lk + a[i]) != id)
        atomicMin(lk + a[i], id);
  }

  __device__ void mark_p2_iterator(int start, int n, int step, int* a, int id) {
    for (int i = start; i < n; i += step)
      if (LOADCG(lk + a[i]) != id)
        atomicMin(lk + a[i], id);
  }

  __device__ bool owns(int n, int* a, int id) {
    for (int i = 0; i < n; i++)
      if (LOADCG(lk + a[i]) != id)
        return false;

    return true;
  }

  __device__ bool owns_iterator(int start, int n, int step, int* a, int id) {
    for (int i = start; i < n; i += step)
      if (LOADCG(lk + a[i]) != id)
        return false;

    return true;
  }
};
