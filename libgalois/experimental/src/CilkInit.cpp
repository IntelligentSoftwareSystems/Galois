/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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

#include "galois/Threads.h"
#include "galois/CilkInit.h"
#include "galois/runtime/PerThreadStorage.h"
#include "galois/runtime/ThreadPool.h"
#include "galois/gIO.h"
#include "galois/substrate/HWTopo.h"
#include "galois/substrate/TID.h"
#include "galois/substrate/EnvCheck.h"

#ifdef HAVE_CILK
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#endif

#ifdef HAVE_CILK 
namespace internal {

static bool initialized = false;

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

static void initOne (BusyBarrier& busybarrier, unsigned tid) {
  galois::runtime::LL::initTID(tid % galois::runtime::LL::getMaxThreads());
  galois::runtime::initPTS_cilk ();

  unsigned id = galois::runtime::LL::getTID ();
  pthread_t self = pthread_self ();

  std::printf ("CILK: Thread %ld assigned id=%d\n", self, id);

  if (id != 0 || !galois::runtime::LL::EnvCheck("GALOIS_DO_NOT_BIND_MAIN_THREAD")) {
    galois::runtime::LL::bindThreadToProcessor(id);
  }


  busybarrier.wait (); 
}

} // end namespace internal

void galois::CilkInit (void) {

  if (internal::initialized) { 
    return ;
  } else {

    internal::initialized = true;

    unsigned numT = getActiveThreads ();

    // unsigned tot = __cilkrts_get_total_workers ();
    // std::printf ("CILK: total cilk workers = %d\n", tot);

    if (!galois::runtime::LL::EnvCheck("GALOIS_DO_NOT_BIND_MAIN_THREAD")) {
      GALOIS_DIE("Run program as: GALOIS_DO_NOT_BIND_MAIN_THREAD=1 prog args");
    }

    char nw_str[128];
    std::sprintf (nw_str, "%d", numT);

    std::printf ("CILK: Trying to set worker count to: %s\n", nw_str);
    if (0 != __cilkrts_set_param ("nworkers", nw_str)) {
      GALOIS_DIE("CILK: Failed to set Cilk worker count\n");
    } else {
      std::printf ("CILK: successfully set nworkers=%s\n", nw_str);
    }

    // if (nw != numT) {
      // std::printf ("CILK: cilk nworkers=%d != galois threads=%d\n", nw, numT); 
      // unsigned tot = __cilkrts_get_total_workers ();
      // std::printf ("CILK: total cilk workers = %d\n", tot);
      // std::abort ();
    // }

    internal::BusyBarrier busybarrier (numT);

    for (unsigned i = 0; i < numT; ++i) {
      cilk_spawn initOne (busybarrier, i);
    } // end for
  }
}
#else 
void galois::CilkInit (void) {
  GALOIS_DIE("Cilk not found\n");
}
#endif // HAVE_CILK
