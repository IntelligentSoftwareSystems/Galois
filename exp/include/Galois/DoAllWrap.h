/** DoAll wrapper -*- C++ -*-
 * @file
 * This is the only file to include for basic Galois functionality.
 *
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */
#ifndef GALOIS_DOALL_WRAPPER_H
#define GALOIS_DOALL_WRAPPER_H

#include "Galois/Galois.h"
#include "Galois/Runtime/DoAllCoupled.h"
#include "Galois/Runtime/ll/EnvCheck.h"

#ifdef GALOIS_USE_TBB
#include "tbb/parallel_for_each.h"
#endif

#if defined(__INTEL_COMPILER)
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#endif

#include <unistd.h>

#include "llvm/Support/CommandLine.h"

namespace cll = llvm::cl;


namespace Galois {

enum DoAllTypes { 
  GALOIS, GALOIS_STEAL, COUPLED, CILK, OPENMP 
};

extern cll::opt<DoAllTypes> doAllKind;

template <DoAllTypes TYPE> 
struct DoAllImpl {
  template <typename I, typename F>
  static inline void go (I beg, I end, F func, const char* loopname) {
    std::abort ();
  }

  template <typename PW, typename F>
  static inline void go (PW& perThrdWL, F func, const char* loopname) {
    std::abort ();
  }
};

template <>
struct DoAllImpl<GALOIS> {
  template <typename I, typename F>
  static inline void go (I beg, I end, F func, const char* loopname) {
    Galois::Runtime::do_all_impl (Runtime::makeStandardRange (beg, end), func, loopname, false);
  }

  template <typename PW, typename F>
  static inline void go (PW& perThrdWL, F func, const char* loopname) {
    go (perThrdWL.begin_all (), perThrdWL.end_all (), func, loopname);
  }
};

template <>
struct DoAllImpl<GALOIS_STEAL> {
  template <typename I, typename F>
  static inline void go (I beg, I end, F func, const char* loopname) {
    Galois::Runtime::do_all_impl (Runtime::makeStandardRange (beg, end), func, loopname, true);
  }

  template <typename PW, typename F>
  static inline void go (PW& perThrdWL, F func, const char* loopname) {
    go (perThrdWL.begin_all (), perThrdWL.end_all (), func, loopname);
  }
};

template <>
struct DoAllImpl<COUPLED> {
  template <typename I, typename F>
  static inline void go (I beg, I end, F func, const char* loopname) {
    Galois::Runtime::do_all_coupled (beg, end, func, loopname);
  }

  template <typename PW, typename F>
  static inline void go (PW& perThrdWL, F func, const char* loopname) {
    Galois::Runtime::do_all_coupled (perThrdWL, func, loopname);
  }
};


#if defined(__INTEL_COMPILER)

template <>
struct DoAllImpl<CILK> {

  template <typename I, typename F>
  static inline void go (I beg, I end, F func, const char* loopname) {

    init ();

    cilk_for(I it = beg; it != end; ++it) {
      func (*it);
    }
  }

  template <typename PW, typename F>
  static inline void go (PW& perThrdWL, F func, const char* loopname) {
    go (perThrdWL.begin_all (), perThrdWL.end_all (), func, loopname);
  }

  struct BusyBarrier {
    volatile int entered;

    void check () const { assert (entered > 0); }

    BusyBarrier (unsigned val) : entered (val) 
    {
      check ();
    }

    void wait () {
      check ();
      __sync_fetch_and_sub (&entered, 1);
      while (entered > 0) {}
    }

    void reinit (unsigned val) {
      entered = val;
      check ();
    }
  };

  static bool initialized;


  static void initOne (BusyBarrier& busybarrier, unsigned tid) {
    Runtime::LL::initTID(tid % Runtime::getMaxThreads());
        Runtime::initPTS_cilk ();

        unsigned id = Runtime::LL::getTID ();
        pthread_t self = pthread_self ();

        std::printf ("CILK: Thread %ld assigned id=%d\n", self, id);

        if (id != 0 || !Runtime::LL::EnvCheck("GALOIS_DO_NOT_BIND_MAIN_THREAD")) {
          Runtime::LL::bindThreadToProcessor(id);
        }


        busybarrier.wait (); 
  }

  static void init () {

    if (initialized) { 
      return ;
    } else {

      initialized = true;

      unsigned numT = getActiveThreads ();

      unsigned tot = __cilkrts_get_total_workers ();
      std::printf ("CILK: total cilk workers = %d\n", tot);

      // char nw_str[128];
      // std::sprintf (nw_str, "%d", numT);
      // __cilkrts_set_param ("nworkers", nw_str);

      unsigned nw = __cilkrts_get_nworkers ();

      if (nw != numT) {
        std::printf ("CILK: cilk nworkers=%d != galois threads=%d\n", nw, numT); 
        unsigned tot = __cilkrts_get_total_workers ();
        std::printf ("CILK: total cilk workers = %d\n", tot);
        std::abort ();
      }

      BusyBarrier busybarrier (numT);

      for (unsigned i = 0; i < numT; ++i) {
        cilk_spawn initOne (busybarrier, i);
      } // end for
    }
  }

};

#endif

template <typename I, typename F> 
void do_all_choice (I beg, I end, F func, const char* loopname=0) {
  switch (doAllKind) {
    case GALOIS_STEAL:
      DoAllImpl<GALOIS_STEAL>::go (beg, end, func, loopname);
      break;
    case GALOIS:
      DoAllImpl<GALOIS>::go (beg, end, func, loopname);
      break;
    case COUPLED:
      DoAllImpl<COUPLED>::go (beg, end, func, loopname);
      break;
    case CILK:
      DoAllImpl<CILK>::go (beg, end, func, loopname);
      break;
    case OPENMP:
      DoAllImpl<OPENMP>::go (beg, end, func, loopname);
      break;
    default:
      abort ();
      break;
  }
}

template <typename PW, typename F> 
void do_all_choice (PW& perThrdWL, F func, const char* loopname=0) {
  switch (doAllKind) {
    case GALOIS_STEAL:
      DoAllImpl<GALOIS_STEAL>::go (perThrdWL, func, loopname);
      break;
    case GALOIS:
      DoAllImpl<GALOIS>::go (perThrdWL, func, loopname);
      break;
    case COUPLED:
      DoAllImpl<COUPLED>::go (perThrdWL, func, loopname);
      break;
    case CILK:
      DoAllImpl<CILK>::go (perThrdWL, func, loopname);
      break;
    case OPENMP:
      DoAllImpl<OPENMP>::go (perThrdWL, func, loopname);
      break;
    default:
      abort ();
      break;
  }
}

} // end namespace Galois

#endif //  GALOIS_DOALL_WRAPPER_H

