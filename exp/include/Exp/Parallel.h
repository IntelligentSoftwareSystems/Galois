#ifndef EXP_PARALLEL_H
#define EXP_PARALLEL_H

#include <cstdlib>
#ifdef EXP_DOALL_TBB
#include "tbb/task_scheduler_init.h"
#endif
#ifdef EXP_DOALL_GALOIS
#include "Galois/Galois.h"
#endif
extern void DMP_Galois_init();

namespace Exp {

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


#ifdef EXP_DOALL_GALOIS
struct Init {
  Init() {
#ifdef GALOIS_DMP
    DMP_Galois_init();
#endif
    Galois::setActiveThreads(Exp::getNumThreads()); 
  }
};
#endif

#ifdef EXP_DOALL_TBB
struct Init {
  tbb::task_scheduler_init* init;

  int get_threads() {
    char *p = getenv("TBB_NUM_THREADS");
    if (p) {
      int n = atoi(p);
      if (n > 0)
        return n;
    }
    return 1;
  }
  
  Init() {
    int n = get_threads();
    init = new tbb::task_scheduler_init(n);  
  }

  ~Init() {
    if (init)
      delete init;
  }
};
#endif

#if defined(EXP_DOALL_CILK) || defined(EXP_DOALL_CILKP) || defined(EXP_DOALL_OPENMP) || defined(EXP_DOALL_OPENMP_RUNTIME)
struct Init { };
#endif

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
