/*
   pipe.h

   Implements PipeContext*. Part of the GGC source code.

   Copyright (C) 2014--2016, The University of Texas at Austin

   See LICENSE.TXT for copyright license.

   Author: Sreepathi Pai <sreepai@ices.utexas.edu>
*/

#pragma once
#include <cuda.h>

class Worklist2Light;

struct oi_save {
  int in;
  int out;
  int re;
  // should be if-def'ed?
  int in_currslot;
  int out_currslot;
  int re_currslot;
};

template <class WLT>
struct PipeContextT {
  WLT wl[3];
  int in, out, re;
  struct oi_save* ois;

  PipeContextT() {}

  PipeContextT(size_t size) {
    wl[0] = WLT(size);
    wl[1] = WLT(size);
    wl[2] = WLT(size);
    in    = 0;
    out   = 1;
    re    = 2;
    ois   = 0;
  }

  __device__ __host__ WLT& in_wl() { return wl[in]; }

  __device__ __host__ WLT& out_wl() { return wl[out]; }

  __device__ __host__ WLT& re_wl() { return wl[re]; }

  __device__ __host__ void swap(int& x, int& y) {
    int t;
    t = x;
    x = y;
    y = t;
  }
  __device__ __host__ inline void advance() {
    wl[in].reset();
    swap(in, out);
  }

  __device__ __host__ inline void advance2() { swap(in, out); }

  __device__ __host__ inline void retry() {
    wl[in].reset();
    swap(re, in);
  }

  __device__ __host__ inline void retry2() { swap(re, in); }

  __host__ void prep() { check_cuda(cudaMalloc(&ois, sizeof(struct oi_save))); }

  __device__ void save() {
    ois->in  = in;
    ois->out = out;
    ois->re  = re;

    ois->in_currslot  = wl[in].currslot;
    ois->out_currslot = wl[out].currslot;
    ois->re_currslot  = wl[re].currslot;
  }

  __host__ void restore() {
    struct oi_save local;
    check_cuda(cudaMemcpy(&local, ois, sizeof(struct oi_save),
                          cudaMemcpyDeviceToHost));

    in  = local.in;
    out = local.out;
    re  = local.re;

    wl[in].set_slot(local.in_currslot);
    wl[out].set_slot(local.out_currslot);
    wl[re].set_slot(local.re_currslot);

    check_cuda(cudaFree(ois));
  }

  __host__ void free() {
    for (int i = 0; i < 3; i++) {
      wl[i].free();
    }
  }
};

struct PipeContextLight {
  Worklist2Light wl[2];
  int index;
  struct oi_save* ois;

  template <typename T>
  __device__ PipeContextLight(PipeContextT<T> pipe) {
    wl[0].fromWL2(pipe.in_wl());
    wl[1].fromWL2(pipe.out_wl());
    // wl[2].fromWL2(pipe.re_wl());   // not used
    index = 0;
    ois   = 0;
  }

  /* __device__ __host__ __forceinline__ */
  /* Worklist2Light &in_wl() { */
  /*   //assert(in != re && in != out && re != out); */
  /*   return wl[index]; */
  /* } */

  /* __device__ __host__ __forceinline__ */
  /* Worklist2Light &out_wl() { */
  /*   //assert(out != re && in != out  && re != in); */
  /*   return wl[index ^ 1]; */
  /* } */

  /* __device__ __host__ __forceinline__ */
  /* Worklist2Light &re_wl() { */
  /*   //assert(in != re && re != out  && in != out); */
  /*   //return wl[2]; */
  /* } */

  /* __device__ __host__  */
  /* void swap(int &x, int &y) { */
  /*   int t; */
  /*   t = x; */
  /*   x = y; */
  /*   y = t; */
  /* } */
  /* __device__ __host__ inline */
  /* void advance() { */
  /*   //wl[in].reset(); */
  /*   //swap(in, out); */
  /* } */

  /* __device__ __host__ inline */
  /* void advance2() { */
  /*   //swap(in, out); */
  /*   index ^= 1; */
  /* } */

  /* __device__ __host__ inline */
  /* void retry() { */
  /*   //wl[in].reset(); */
  /*   //swap(re, in); */
  /* } */

  /* __device__ __host__ inline */
  /* void retry2() { */
  /*   //swap(re, in); */
  /* } */

  /* __host__ void prep() { */
  /*   check_cuda(cudaMalloc(&ois, sizeof(struct oi_save))); */
  /* } */

  template <typename T>
  __device__ void save(PipeContextT<T>& pipe, int index) {
    pipe.ois->in  = index;
    pipe.ois->out = index ^ 1;
    pipe.ois->re  = 2;

    pipe.ois->in_currslot  = wl[index].currslot;
    pipe.ois->out_currslot = wl[index ^ 1].currslot;
    // pipe.ois->re_currslot = wl[2].currslot;
  }

  /* __host__ void restore() { */
  /*   struct oi_save local; */
  /*   check_cuda(cudaMemcpy(&local, ois, sizeof(struct oi_save),
   * cudaMemcpyDeviceToHost)); */

  /*   index = local.in; */
  /*   // = local.out; */
  /*   //re = local.re; */

  /*   wl[index].set_slot(local.in_currslot); */
  /*   wl[index ^ 1].set_slot(local.out_currslot); */
  /*   //wl[2].set_slot(local.re_currslot); */

  /*   check_cuda(cudaFree(ois)); */
  /* }     */

  /* __host__ void free() { */
  /*   for(int i = 0; i < 3; i++) { */
  /*     //wl[i].free(); */
  /*   } */
  /* } */
};

typedef PipeContextT<Worklist2> PipeContext;
typedef PipeContextT<WorklistT> PipeContextWT;
