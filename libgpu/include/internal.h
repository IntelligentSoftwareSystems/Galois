/*
   internal.h

   Implements internal runtime routines. Part of the GGC source code.

   Copyright (C) 2014--2016, The University of Texas at Austin

   See LICENSE.TXT for copyright license.

   Author: Sreepathi Pai <sreepai@ices.utexas.edu>
*/

#pragma once

typedef int cuda_size_t;

// TODO: specialize this
const int MAX_TB_SIZE     = 1024;
const int LOG_MAX_TB_SIZE = 10;

/* container to perform multiple independent sums (scans) */
template <int items, typename T>
struct multiple_sum {
  T el[items];

  // https://nvlabs.github.io/cub/classcub_1_1_block_scan.html#a6ed3f77795e582df31d3d6d9d950615e
  // "This operation assumes the value of obtained by the T's default constructor (or by zero-initialization if no user-defined default constructor exists) is suitable as the identity value zero for addition."
  __device__ __host__ multiple_sum() : multiple_sum(T()) { }

  __device__ __host__ multiple_sum(const T e) {
    for (int i = 0; i < items; i++)
      el[i] = e;
  }

  __device__ __host__ multiple_sum& operator=(const T rhs) {
    for (int i = 0; i < items; i++)
      el[i] = rhs;

    return *this;
  }

  __device__ __host__ multiple_sum& operator+=(const multiple_sum& rhs) {
    for (int i = 0; i < items; i++)
      el[i] += rhs.el[i];

    return *this;
  }

  __device__ __host__ friend multiple_sum operator+(multiple_sum lhs,
                                                    const multiple_sum& rhs) {
    return lhs += rhs;
  }
};

/* for two scans */
struct pair {
  int x, y, z;

  __device__ __host__ pair& operator+=(const pair& rhs) {
    x += rhs.x;
    y += rhs.y;
    z += rhs.z;

    return *this;
  }

  __device__ __host__ friend pair operator+(pair lhs, const pair& rhs) {
    return lhs += rhs;
  }
};

template <const int WARPS_PER_TB>
struct warp_np {
  volatile index_type owner[WARPS_PER_TB];
  volatile index_type start[WARPS_PER_TB];
  volatile index_type size[WARPS_PER_TB];
  volatile index_type offset[WARPS_PER_TB]; // task offset
  volatile index_type src[WARPS_PER_TB];
};

struct tb_np {
  index_type owner;
  index_type start;
  index_type size;
  index_type offset;
  index_type src;
};

template <const int ITSIZE>
struct fg_np {
  index_type itvalue[ITSIZE];
  index_type src[ITSIZE];
};

struct empty_np {};

template <typename ts_type, typename index_type, typename TTB, typename TWP,
          typename TFG>
union np_shared {
  // for scans
  ts_type temp_storage;

  // for tb-level np
  TTB tb;

  // for warp-level np
  TWP warp;

  TFG fg;
};

struct NPInspector1 {
  cuda_size_t total;   // total work across all threads
  cuda_size_t done;    // total work done across all threads
  cuda_size_t size;    // size of this thread's work
  cuda_size_t start;   // this thread's iteration start value
  cuda_size_t offset;  // offset within flattened iteration space
  cuda_size_t my_done; // items completed within this thread's space

  // inspect should be inspect_begin, inspect_end, inspect_update really?
  // especially for custom closures...

  template <typename T>
  __device__ __host__ cuda_size_t inspect(T* itvalue,
                                          const cuda_size_t ITSIZE) {
    cuda_size_t _np_i;
    for (_np_i = 0;
         (my_done + _np_i) < size && (offset - done + _np_i) < ITSIZE;
         _np_i++) {
      itvalue[offset - done + _np_i] = start + my_done + _np_i;
    }

    my_done += _np_i;
    offset += _np_i;

    return _np_i;
  }

  template <typename T>
  __device__ __host__ cuda_size_t inspect2(T* itvalue, T* source,
                                           const cuda_size_t ITSIZE,
                                           const cuda_size_t src) {
    cuda_size_t _np_i;
    for (_np_i = 0;
         (my_done + _np_i) < size && (offset - done + _np_i) < ITSIZE;
         _np_i++) {
      itvalue[offset - done + _np_i] = start + my_done + _np_i;
      source[offset - done + _np_i]  = src;
    }

    my_done += _np_i;
    offset += _np_i;

    return _np_i;
  }

  __device__ __host__ bool work() const { return total > 0; }

  __device__ __host__ bool valid(const cuda_size_t ltid) const {
    return ltid < total; // remember total decreases every round
  }

  __device__ __host__ void execute_round_done(const cuda_size_t ITSIZE) {
    total -= ITSIZE;
    done += ITSIZE;
  }
};
