/*
   lockarray.h

   Implements LockArray*. Part of the GGC source code.

   Copyright (C) 2014--2016, The University of Texas at Austin

   See LICENSE.TXT for copyright license.

   Author: Sreepathi Pai <sreepai@ices.utexas.edu>
*/

#include "sharedptr.h"
#include <cassert>

#define UNLOCKED 0
#define LOCKED 1

class LockArraySimple {
public:
  Shared<int> locks;
  int* glocks;
  int nlocks;

  LockArraySimple(size_t nlocks) {
    locks        = Shared<int>(nlocks);
    this->nlocks = nlocks;
    glocks       = locks.zero_gpu();
  }

  // do not use this
  __device__ bool acquire(int ndx) {
    assert(ndx >= 0 && ndx < nlocks);
    while (atomicCAS(glocks + ndx, UNLOCKED, LOCKED) == LOCKED) {
      __threadfence();
    }
    return glocks[ndx] == LOCKED;
  }

  __device__ bool acquire_or_fail(int ndx) {
    assert(ndx >= 0 && ndx < nlocks);
    return atomicCAS(glocks + ndx, UNLOCKED, LOCKED) == UNLOCKED;
  }

  __device__ bool is_locked(int ndx) {
    // TODO: atomic reads?
    assert(ndx >= 0 && ndx < nlocks);
    return glocks[ndx] == LOCKED;
  }

  __device__ void release(int ndx) {
    __threadfence();
    bool was_locked = atomicCAS(glocks + ndx, LOCKED, UNLOCKED) == LOCKED;
    assert(was_locked);
  }
};

class LockArrayTicket : public LockArraySimple {
public:
  Shared<int> tickets;

  int* gtickets;

  LockArrayTicket(size_t nlocks) : LockArraySimple(nlocks) {
    tickets  = Shared<int>(nlocks);
    gtickets = tickets.gpu_wr_ptr();
    assert(cudaMemset(gtickets, 0, nlocks * sizeof(int)) == cudaSuccess);
  }

  __device__ int reserve(int ndx) {
    assert(ndx >= 0 && ndx < nlocks);
    return atomicAdd(gtickets + ndx, 1);
  }

  __device__ bool acquire_or_fail(int ndx, int ticket) {
    assert(ndx >= 0 && ndx < nlocks);
    return glocks[ndx] == ticket;
  }

  __device__ bool is_locked(int ndx) {
    assert(ndx >= 0 && ndx < nlocks);
    return glocks[ndx] < gtickets[ndx];
  }

  __device__ void release(int ndx) {
    __threadfence();
    bool was_locked = glocks[ndx]++ < gtickets[ndx];
    assert(was_locked);
  }
};

typedef LockArraySimple LockArray;
