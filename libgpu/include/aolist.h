#pragma once
/*
   aolist.h

   Implements AppendOnlyList. Part of the GGC source code.

   Copyright (C) 2014--2016, The University of Texas at Austin

   See LICENSE.TXT for copyright license.

   Author: Sreepathi Pai <sreepai@ices.utexas.edu>
*/

#include "cub/cub.cuh"
#include "cutil_subset.h"
#include "bmk2.h"
#include <kernels/mergesort.cuh>

struct AppendOnlyList {
  int* dl;
  int *dindex, index;
  int size;
  bool f_will_write;

  Shared<int> list;

  AppendOnlyList() { size = 0; }

  AppendOnlyList(size_t nsize) {
    size = nsize;

    if (nsize == 0) {
      dl    = NULL;
      index = 0;
    } else {
      list.alloc(nsize);
      dl = list.gpu_wr_ptr();
      CUDA_SAFE_CALL(cudaMalloc(&dindex, 1 * sizeof(int)));
      CUDA_SAFE_CALL(cudaMemcpy((void*)dindex, &zero, 1 * sizeof(zero),
                                cudaMemcpyHostToDevice));
      index = 0;
    }
  }

  void sort() {
    MergesortKeys(list.gpu_wr_ptr(), nitems(), mgpu::less<int>(), *mgc);
  }

  void update_cpu() { list.cpu_rd_ptr(); }

  void display_items() {
    int nsize = nitems();
    int* l    = list.cpu_rd_ptr();

    printf("LIST: ");
    for (int i = 0; i < nsize; i++)
      printf("%d %d, ", i, l[i]);

    printf("\n");
    return;
  }

  void reset() {
    CUDA_SAFE_CALL(cudaMemcpy((void*)dindex, &zero, 1 * sizeof(zero),
                              cudaMemcpyHostToDevice));
  }

  __device__ __host__ int nitems() {
#ifdef __CUDA_ARCH__
    return *dindex;
#else
    CUDA_SAFE_CALL(cudaMemcpy(&index, (void*)dindex, 1 * sizeof(index),
                              cudaMemcpyDeviceToHost));
    return index;
#endif
  }

  __device__ int push(int item) {
    int lindex = atomicAdd((int*)dindex, 1);
    assert(lindex <= size);

    dl[lindex] = item;
    return 1;
  }

  __device__ int pop_id(int id, int& item) {
    if (id < *dindex) {
      item = cub::ThreadLoad<cub::LOAD_CG>(dl + id);
      // item = dwl[id];
      return 1;
    }

    return 0;
  }

  __device__ int pop(int& item) {
    int lindex = atomicSub((int*)dindex, 1);
    if (lindex <= 0) {
      *dindex = 0;
      return 0;
    }

    item = dl[lindex - 1];
    return 1;
  }

  __device__ int setup_push_warp_one() {
    int first, total, offset, lindex = 0;

    warp_active_count(first, offset, total);

    if (offset == 0) {
      lindex = atomicAdd((int*)dindex, total);
      assert(lindex <= size);
    }

    lindex = cub::ShuffleIndex(lindex, first);
    // lindex = cub::ShuffleIndex(lindex, first); // CUB > 1.3.1
    return lindex + offset;
  }

  __device__ int setup_push_thread(int nitems) {
    int lindex = atomicAdd((int*)dindex, nitems);
    assert(lindex <= size);

    return lindex;
  }

  __device__ int do_push(int start, int id, int item) {
    assert(id <= size);
    dl[start + id] = item;
    return 1;
  }

  template <typename T>
  __device__ __forceinline__ int push_1item(int nitem, int item,
                                            int threads_per_block) {
    __shared__ typename T::TempStorage temp_storage;
    __shared__ int queue_index;
    int total_items = 0;
    int thread_data = nitem;

    T(temp_storage).ExclusiveSum(thread_data, thread_data, total_items);

    if (threadIdx.x == 0) {
      if (debug)
        printf("t: %d\n", total_items);
      queue_index = atomicAdd((int*)dindex, total_items);
      // printf("queueindex: %d %d %d %d %d\n", blockIdx.x, threadIdx.x,
      // queue_index, thread_data + n_items, total_items);
    }

    __syncthreads();

    if (nitem == 1) {
      if (queue_index + thread_data >= size) {
        printf("GPU: exceeded length: %d %d %d\n", queue_index, thread_data,
               size);
        return 0;
      }

      // dwl[queue_index + thread_data] = item;
      cub::ThreadStore<cub::STORE_CG>(dl + queue_index + thread_data, item);
    }

    return total_items;
  }

  void save(const char* f, const unsigned iteration) {
    char n[255];
    int ret;

    ret = snprintf(n, 255, "%s%s-%05d-%s.wl", instr_trace_dir(), f, iteration,
                   instr_uniqid());

    if (ret < 0 || ret >= 255) {
      fprintf(stderr, "Error creating filename for kernel '%s', iteration %d\n",
              f, iteration);
      exit(1);
    }

    int nsize = nitems();
    int* wl   = list.cpu_rd_ptr();

    TRACE of = trace_open(n, "w");
    instr_write_array(n, of, sizeof(int), nsize, wl);
    trace_close(of);
    bmk2_log_collect("ggc/wlcontents", n);
    return;
  }

  void load(const char* f, const unsigned iteration) {
    char n[255];
    int ret;

    ret = snprintf(n, 255, "%s%s-%05d-%s.wl", instr_trace_dir(), f, iteration,
                   instr_uniqid());

    if (ret < 0 || ret >= 255) {
      fprintf(stderr, "Error creating filename for kernel '%s', iteration %d\n",
              f, iteration);
      exit(1);
    }

    TRACE of = trace_open(n, "w");
    int nsize =
        instr_read_array(n, of, sizeof(int), size, list.cpu_wr_ptr(true));
    list.gpu_rd_ptr();
    check_cuda(cudaMemcpy((void*)dindex, &nsize, 1 * sizeof(nsize),
                          cudaMemcpyHostToDevice));
    trace_close(of);
    return;
  }
};
