/*
   rv.h

   Implements Reduce on the GPU. Adapted from the GGC source code.

   Copyright (C) 2014--2016, The University of Texas at Austin

   See LICENSE.TXT for copyright license.

   Author: Sreepathi Pai <sreepai@ices.utexas.edu>
   Author: Roshan Dathathri <roshan@cs.utexas.edu>
*/

#pragma once
#include "cub/cub.cuh"
#include "atomic_helpers.h"

template <typename Type>
class HGReducible {
public:
  Type* rv; // allocated by the user

  __device__ void thread_entry() {}

  template <typename T>
  __device__ void thread_exit(typename T::TempStorage& temp_storage) {}

  __device__ void reduce(Type value) {}
};

template <typename Type>
class HGAccumulator : public HGReducible<Type> {
  Type local;

public:
  __device__ void thread_entry() { local = 0; }

  template <typename T>
  __device__ void thread_exit(typename T::TempStorage& temp_storage) {
    local = T(temp_storage).Sum(local);

    if (threadIdx.x == 0 && local) {
      atomicTestAdd((Type*)HGReducible<Type>::rv, local);
    }
  }

  __device__ void reduce(Type value) {
    if (value)
      local += value;
  }
};

template <typename Type>
class HGReduceMax : public HGReducible<Type> {
  Type local;

  struct MaxOp {
    __device__ Type operator()(const Type& a, const Type& b) {
      return (a > b) ? a : b;
    }
  };
  MaxOp maxOp;

public:
  __device__ void thread_entry() { local = 0; } // assumes positive numbers

  template <typename T>
  __device__ void thread_exit(typename T::TempStorage& temp_storage) {
    local = T(temp_storage).Reduce(local, maxOp);

    if (threadIdx.x == 0 && local) {
      atomicTestMax((Type*)HGReducible<Type>::rv, local);
    }
  }

  __device__ void reduce(Type value) {
    if (local < value)
      local = value;
  }
};

template <typename Type>
class HGReduceMin : public HGReducible<Type> {
  Type local;

  struct MinOp {
    __device__ Type operator()(const Type& a, const Type& b) {
      return (a < b) ? a : b;
    }
  };
  MinOp minOp;

public:
  __device__ void thread_entry() {
    local = 1073741823;
  } // assumes Type can hold this number

  template <typename T>
  __device__ void thread_exit(typename T::TempStorage& temp_storage) {
    local = T(temp_storage).Reduce(local, minOp);

    if (threadIdx.x == 0 && (local != 1073741823)) {
      atomicTestMin((Type*)HGReducible<Type>::rv, local);
    }
  }

  __device__ void reduce(Type value) {
    if (local > value)
      local = value;
  }
};
