#ifndef EXP_PARALLEL_H
#define EXP_PARALLEL_H

#include <cstdlib>

extern void DMP_Galois_init();

namespace Exp {

void initRuntime();

namespace {
// Dummy code to call static initializer 
struct Init {
  Init() {
#ifdef GALOIS_DMP
    DMP_Galois_init();
#endif
    initRuntime(); 
  }
};

static Init iii;
}

extern __thread unsigned TID;
extern unsigned nextID;

// NB(ddn): Not "DRF" for DMP but this is okay if we don't interpret the value
// itself, i.e., only use this as a identifier for thread-local data.
static inline unsigned getTID() {
  unsigned x = TID;
  if (x & 1)
    return x >> 1;
  x = __sync_fetch_and_add(&nextID, 1);
  TID = (x << 1) | 1;
  return x;
}

unsigned getNumThreads();
int getNumRounds();

}

#if defined(EXP_DOALL_CILK)
#include <cilk/cilk.h>
#define parallel_doall(type, index, begin, end) \
  cilk_for (type index = begin; index < end; ++index)
#define parallel_doall_1(type, index, begin, end) \
  cilk_for (type index = begin; index < end; ++index)
#define parallel_doall_end 

// openmp
#elif defined(EXP_DOALL_OPENMP)
#include <omp.h>
#define cilk_spawn
#define cilk_sync
#define parallel_doall(type, index, begin, end) \
  _Pragma("omp parallel for") for (type index = begin; index < end; ++index)
#define parallel_doall_1(type, index, begin, end) \
  _Pragma("omp parallel for schedule (static,1)") for (type index = begin; index < end; ++index)
#define parallel_doall_end 

// openmp
#elif defined(EXP_DOALL_OPENMP_RUNTIME)
#include <omp.h>
#define cilk_spawn
#define cilk_sync
#define parallel_doall(type, index, begin, end) \
  _Pragma("omp parallel for schedule (runtime)") for (type index = begin; index < end; ++index)
#define parallel_doall_1(type, index, begin, end) \
  _Pragma("omp parallel for schedule (static,1)") for (type index = begin; index < end; ++index)
#define parallel_doall_end 

// GALOIS
#elif defined(EXP_DOALL_GALOIS)
#include "Galois/Galois.h"
#include "boost/iterator/counting_iterator.hpp"
#define cilk_spawn
#define cilk_sync
#define parallel_doall(type, index, begin, end) \
  Galois::do_all(boost::counting_iterator<type>(begin), boost::counting_iterator<type>(end), [&](type index)
#define parallel_doall_1(type, index, begin, end) \
  Galois::do_all(boost::counting_iterator<type>(begin), boost::counting_iterator<type>(end), [&](type index)
#define parallel_doall_end );

// TBB
#elif defined(EXP_DOALL_TBB)
#include "boost/iterator/counting_iterator.hpp"
#include "tbb/parallel_for_each.h"
#define cilk_spawn
#define cilk_sync
#define parallel_doall(type, index, begin, end) \
  tbb::parallel_for_each(boost::counting_iterator<type>(begin), boost::counting_iterator<type>(end), [&](type index)
#define parallel_doall_1(type, index, begin, end) \
  tbb::parallel_for_each(boost::counting_iterator<type>(begin), boost::counting_iterator<type>(end), [&](type index)
#define parallel_doall_end );

// c++
#else
#define cilk_spawn
#define cilk_sync
#define parallel_doall_1(type, index, begin, end) \
  for (type index = begin; index < end; ++index)
#define parallel_doall(type, index, begin, end) \
  for(type index = begin; index < end; ++index)
#define parallel_doall_end 

#endif

#endif
