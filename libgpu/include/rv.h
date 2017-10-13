/*
   rv.h

   Implements RetVal. Part of the GGC source code. 

   Copyright (C) 2014--2016, The University of Texas at Austin

   See LICENSE.TXT for copyright license.

   Author: Sreepathi Pai <sreepai@ices.utexas.edu> 
*/

#pragma once
#include "cub/cub.cuh"

class RV {
 public:
  int *rv;

  __device__ void thread_entry() {};
  __device__ void thread_exit() {};
  __device__ void do_return(int value) {};
  __device__ void return_(int value) {};
};

class Any: public RV {
 public:
  int local;
  __device__ void thread_entry() { local = 0;}

  template <typename T>
  __device__ void thread_exit(typename T::TempStorage &temp_storage) { 
    local = T(temp_storage).Sum(local);

    if(threadIdx.x == 0 && local)  {
      *rv = local;
    }
  }

  __device__ void do_return(int value) { if(value) local = value;}
  __device__ void return_(int value) { if(value) *rv = value;}
};

class All: public RV {
 public:
  __device__ void thread_entry() {};
  __device__ void thread_exit() {};
  __device__ void do_return(int value) {};
  __device__ void return_(int value) { if(value) atomicAdd((int *) rv, 1);}
};

class Sum: public RV {
 public:
  int local;
  __device__ void thread_entry() { local = 0; }

  template <typename T>
  __device__ void thread_exit(typename T::TempStorage &temp_storage) { 
    local = T(temp_storage).Sum(local);

    if(threadIdx.x == 0 && local)  {
      atomicAdd((int *) rv, local);
    }
  }

  __device__ void do_return(int value) { if(value) local += value; }
  __device__ void return_(int value) { if(value) atomicAdd((int *) rv, value); }
};
