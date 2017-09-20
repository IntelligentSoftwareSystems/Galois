#ifndef EXP_PARALLEL_H
#define EXP_PARALLEL_H

#include <cstdlib>
#include <iterator>
#ifdef __GXX_EXPERIMENTAL_CXX0X__
#include <functional>
#else
#include <tr1/functional>
#endif

#ifdef EXP_DOALL_TBB
#include "tbb/task_scheduler_init.h"
#endif

#ifdef EXP_DOALL_GALOIS
#include "Galois/Galois.h"
#endif

#include <pthread.h>
#ifdef GALOIS_USE_DMP
#include "dmp.h"
#endif

//#define USE_SIMPLE_RUNTIME

namespace Exp {

namespace Config {
#ifdef __GXX_EXPERIMENTAL_CXX0X__
  using std::function;
  using std::ref;
#else
  using std::tr1::function;
  using std::tr1::ref;
#endif
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
void beginSampling();
void endSampling();

typedef Config::function<void (void)> RunCommand;

void do_all_init();
void do_all_finish();
void do_all_impl(RunCommand* begin, RunCommand* end);

class PthreadBarrier {
  pthread_barrier_t bar;
  void checkResults(int val);

public:
  PthreadBarrier();
  PthreadBarrier(unsigned int val);
  ~PthreadBarrier();

  void reinit(int val);
  void wait();
  void operator()(void) { wait(); }
};

template<typename IterTy,typename FunctionTy>
struct Work {
  IterTy begin, end;
  FunctionTy function;
  Work(const IterTy& b, const IterTy& e, const FunctionTy& f): begin(b), end(e), function(f) { }

  void operator()() {
    IterTy b(begin), e(end);
    unsigned int a = getNumThreads();
    unsigned int id = getTID();
    unsigned int dist = std::distance(b, e);
    unsigned int num = (dist + a - 1) / a; //round up
    unsigned int A = std::min(num * id, dist);
    unsigned int B = std::min(num * (id + 1), dist);
    e = b;
    std::advance(b, A);
    std::advance(e, B);
    for (; b != e; ++b)
      function(*b);
  }
};

template<typename IterTy,typename FunctionTy>
void do_all(IterTy begin, IterTy end, FunctionTy fn) {
  Work<IterTy,FunctionTy> W(begin, end, fn);
  PthreadBarrier bar(Exp::getNumThreads());

  RunCommand w[2] = { Config::ref(W), Config::ref(bar) }; 

#ifdef USE_SIMPLE_RUNTIME
  do_all_impl(&w[0], &w[1]);
#else
  do_all_impl(&w[0], &w[2]);
#endif
}

#ifdef EXP_DOALL_GALOIS
struct Init {
  Init() {
    galois::setActiveThreads(Exp::getNumThreads()); 
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

#ifdef EXP_DOALL_PTHREAD
struct Init {
  Init() {
    do_all_init();
  }
  ~Init() {
    do_all_finish();
  }
};
#endif

#if defined(EXP_DOALL_CILK) || defined(EXP_DOALL_CILKP) || defined(EXP_DOALL_OPENMP) || defined(EXP_DOALL_OPENMP_RUNTIME) || defined(EXP_DOALL_PTHREAD_OPENMP)
struct Init { };
#endif

}

#if defined(EXP_DOALL_CILK)
#include <cilk/cilk.h>
#define parallel_doall(type, index, begin, end) \
  cilk_for (type index = begin; index < end; ++index)
#define parallel_doall_1(type, index, begin, end) \
  cilk_for (type index = begin; index < end; ++index)
#define parallel_doall_obj(type, index, begin, end, obj) \
  cilk_for (type index = begin; index < end; ++index)
#define parallel_doall_obj_1(type, index, begin, end, obj) \
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
#define parallel_doall_obj(type, index, begin, end, obj) \
  _Pragma("omp parallel for") for (type index = begin; index < end; ++index)
#define parallel_doall_obj_1(type, index, begin, end, obj) \
  _Pragma("omp parallel for schedule (static,1)") for (type index = begin; index < end; ++index)
#define parallel_doall_end 

#elif defined(EXP_DOALL_PTHREAD_OPENMP)
#include <omp.h>
#define cilk_spawn
#define cilk_sync
#define parallel_doall(type, index, begin, end) \
  for(type index = begin; index < end; ++index)
#define parallel_doall_1(type, index, begin, end) \
  for(type index = begin; index < end; ++index)
#define parallel_doall_obj(type, index, begin, end, obj) \
  if (true) { _Pragma("omp parallel for") for (type index = begin; index < end; ++index) { obj(index); } } else for(type index = begin; index < end; ++index)
#define parallel_doall_obj_1(type, index, begin, end, obj) \
  if (true) { _Pragma("omp parallel for schedule (static,1)") for (type index = begin; index < end; ++index) { obj(index); } } else for(type index = begin; index < end; ++index)
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
#define parallel_doall_obj(type, index, begin, end, obj) \
  _Pragma("omp parallel for schedule (runtime)") for (type index = begin; index < end; ++index)
#define parallel_doall_obj_1(type, index, begin, end, obj) \
  _Pragma("omp parallel for schedule (runtime)") for (type index = begin; index < end; ++index)
#define parallel_doall_end 

// GALOIS
#elif defined(EXP_DOALL_GALOIS)
#include "Galois/Galois.h"
#include "boost/iterator/counting_iterator.hpp"
#define cilk_spawn
#define cilk_sync
#define parallel_doall(type, index, begin, end) \
  galois::do_all(boost::counting_iterator<type>(begin), boost::counting_iterator<type>(end), [&](type index)
#define parallel_doall_1(type, index, begin, end) \
  galois::do_all(boost::counting_iterator<type>(begin), boost::counting_iterator<type>(end), [&](type index)
#define parallel_doall_obj(type, index, begin, end, obj) \
  galois::do_all(boost::counting_iterator<type>(begin), boost::counting_iterator<type>(end), [&](type index)
#define parallel_doall_obj_1(type, index, begin, end, obj) \
  galois::do_all(boost::counting_iterator<type>(begin), boost::counting_iterator<type>(end), [&](type index)
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
#define parallel_doall_obj(type, index, begin, end, obj) \
  tbb::parallel_for_each(boost::counting_iterator<type>(begin), boost::counting_iterator<type>(end), [&](type index)
#define parallel_doall_obj_1(type, index, begin, end, obj) \
  tbb::parallel_for_each(boost::counting_iterator<type>(begin), boost::counting_iterator<type>(end), [&](type index)
#define parallel_doall_end );

#elif defined(EXP_DOALL_PTHREAD)
#include "boost/iterator/counting_iterator.hpp"
#define cilk_spawn
#define cilk_sync
#define parallel_doall(type, index, begin, end) \
  for(type index = begin; index < end; ++index)
#define parallel_doall_1(type, index, begin, end) \
  for (type index = begin; index < end; ++index)
#define parallel_doall_obj(type, index, begin, end, obj) \
  if (true) {Exp::do_all(boost::counting_iterator<type>(begin), boost::counting_iterator<type>(end), obj); } else for(type index = begin; index < end; ++index)
#define parallel_doall_obj_1(type, index, begin, end, obj) \
  if (true) {Exp::do_all(boost::counting_iterator<type>(begin), boost::counting_iterator<type>(end), obj); } else for(type index = begin; index < end; ++index)
#define parallel_doall_end 

// c++
#else
#define cilk_spawn
#define cilk_sync
#define parallel_doall(type, index, begin, end) \
  for(type index = begin; index < end; ++index)
#define parallel_doall_1(type, index, begin, end) \
  for (type index = begin; index < end; ++index)
#define parallel_doall_obj(type, index, begin, end, obj) \
  for(type index = begin; index < end; ++index)
#define parallel_doall_obj_1(type, index, begin, end, obj) \
  for(type index = begin; index < end; ++index)
#define parallel_doall_end 

#endif

#endif
