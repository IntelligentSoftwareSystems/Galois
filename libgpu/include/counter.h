#pragma once
/*
   counter.h

   Implements instrumentation counters. Part of the GGC source code.

   Copyright (C) 2014--2016, The University of Texas at Austin

   See LICENSE.TXT for copyright license.

   Author: Sreepathi Pai <sreepai@ices.utexas.edu>
*/

#include <stdio.h>
#include <cassert>
#include <cub/util_device.cuh>

// timeblocks 2.0

const int MAGIC = 0x5a5e5a61; // random

// from http://forums.nvidia.com/index.php?showtopic=186669
static __device__ uint get_smid_reg(void) {
  uint ret;
  asm("mov.u32 %0, %smid;" : "=r"(ret));
  return ret;
}

class GPUCounter {
  unsigned count;
  unsigned value;

public:
  unsigned* tvalues;
  unsigned* tcounts;
  unsigned* smids; // make this a char?
  clock_t* start;

  __device__ void init() {
    count = 0;
    value = 0;
  }

  __device__ void record(int value) { this->value += value; }

  __device__ void count_iter() { count++; }

  __device__ void begin(unsigned tid) {
    this->start[tid] = clock64();
    if (threadIdx.x == 0)
      smids[blockIdx.x] = get_smid_reg();
  }

  __device__ void end(unsigned tid) {
    clock_t e = clock64();
    value     = (e - start[tid]);
  }

  __device__ void finish(unsigned tid) {
    tvalues[tid] = value;
    tcounts[tid] = count;
  }
};

class Counter {
public:
  GPUCounter gc;
  FILE* f;
  Shared<unsigned> tvalues;
  Shared<unsigned> tcounts;
  Shared<unsigned> smids;
  Shared<clock_t> start;

  int threads;
  int blocks, tpblock, dynsmem, residency;
  const void* function;

  int get_residency(int tpb, int dynsmem) {
    int res;

#if CUDA_VERSION < 6050
    assert(dynsmem == 0);
    cub::MaxSmOccupancy(res, function, tpb);
#else
    assert(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
               &res, function, tpb, dynsmem) == cudaSuccess);
#endif

    return res;
  }

  __host__ void init(const char* fname, const void* fn, int blocks, int tpb,
                     int dynsmem) {
    this->blocks  = blocks;
    this->tpblock = tpb;

    threads = blocks * tpb;

    function  = fn;
    residency = get_residency(tpb, dynsmem);
    assert(residency > 0);

    f = fopen(fname, "w");
    if (!f) {
      fprintf(stderr, "Could not open '%s'", fname);
      assert(false);
    }

    assert(fwrite(&MAGIC, sizeof(MAGIC), 1, f) == 1);
    assert(fwrite(&blocks, sizeof(blocks), 1, f) == 1);
    assert(fwrite(&tpb, sizeof(tpb), 1, f) == 1);

    tvalues.alloc(threads);
    tcounts.alloc(threads);
    smids.alloc(blocks);
    start.alloc(threads);

    gc.tvalues = tvalues.gpu_wr_ptr();
    gc.tcounts = tcounts.gpu_wr_ptr();
    gc.smids   = smids.gpu_wr_ptr();
    gc.start   = start.gpu_wr_ptr();
  }

  __host__ GPUCounter& get_gpu() {
    tvalues.gpu_wr_ptr();
    tcounts.gpu_wr_ptr();
    smids.gpu_wr_ptr();
    start.gpu_wr_ptr();

    return gc;
  }

  __host__ void write_data(int iteration, unsigned work, int iblocks = 0,
                           int ithreads = 0, int idynsmem = -1) {
    int zero     = 0;
    unsigned* tv = tvalues.cpu_rd_ptr();
    unsigned* tc = tcounts.cpu_rd_ptr();
    int res;

    if (iblocks == 0)
      iblocks = blocks;
    if (ithreads == 0)
      ithreads = tpblock;
    if (idynsmem == -1)
      idynsmem = dynsmem;

    if (ithreads != tpblock) {
      res = get_residency(ithreads, idynsmem);
      assert(res > 0);
    } else {
      res = residency;
    }

    assert(fwrite(&zero, sizeof(zero), 1, f) == 1);
    assert(fwrite(&iteration, sizeof(iteration), 1, f) == 1);
    assert(fwrite(&work, sizeof(work), 1, f) == 1);
    assert(fwrite(&iblocks, sizeof(iblocks), 1, f) == 1);
    assert(fwrite(&ithreads, sizeof(ithreads), 1, f) == 1);
    assert(fwrite(&idynsmem, sizeof(idynsmem), 1, f) == 1);
    assert(fwrite(&res, sizeof(res), 1, f) == 1);

    // reserved for type identifiers
    assert(fwrite(&zero, sizeof(zero), 1, f) == 1);
    assert(fwrite(&zero, sizeof(zero), 1, f) == 1);

    assert(fwrite(smids.cpu_rd_ptr(), sizeof(unsigned), blocks, f) == blocks);
    assert(fwrite(start.cpu_rd_ptr(), sizeof(clock_t), threads, f) == threads);

    assert(fwrite(tv, sizeof(tv[0]), threads, f) == threads);
    assert(fwrite(tc, sizeof(tc[0]), threads, f) == threads);
  }
  __host__ void zero_gpu() {
    tvalues.zero_gpu();
    tcounts.zero_gpu();
    smids.zero_gpu();
    start.zero_gpu();
  }
};
