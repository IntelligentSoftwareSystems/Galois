/**  -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 */

#include "galois/Threads.h"
#include "galois/CilkInit.h"
#include "galois/runtime/PerThreadStorage.h"
#include "galois/runtime/ThreadPool.h"
#include "galois/gIO.h"
#include "galois/substrate/HWTopo.h"
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
