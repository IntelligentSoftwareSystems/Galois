/** Thread ID -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS SOFTWARE
 * AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY
 * PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY
 * WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF TRADE.
 * NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO THE USE OF THE
 * SOFTWARE OR DOCUMENTATION. Under no circumstances shall University be liable
 * for incidental, special, indirect, direct or consequential damages or loss of
 * profits, interruption of business, or related expenses which may arise from use
 * of Software or Documentation, including but not limited to those resulting from
 * defects in Software and/or Documentation, or loss or inaccuracy of data of any
 * kind.
 *
 * @section Description
 *
 * This contains support for thread id.  See TID.h.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#include "Galois/Runtime/ll/TID.h"
#include "Galois/Runtime/ll/HWTopo.h"
#include <cassert>

#ifdef GALOIS_USE_DRF
#include <pthread.h>
#endif

#ifdef GALOIS_USE_DMP
#include "dmp.h"
#endif

__thread unsigned GaloisRuntime::LL::TID = 0;
static unsigned nextID = 0;

namespace {
struct AtomicNextId {
  unsigned next() {
    return __sync_fetch_and_add(&::nextID, 1);
  }
};
#ifdef GALOIS_USE_DRF
struct PthreadNextId {
  pthread_mutex_t lock;
  PthreadNextId() {
    pthread_mutex_init(&lock, NULL);
  }
  ~PthreadNextId() {
    pthread_mutex_destroy(&lock);
  }
  unsigned next() {
    unsigned val;
    pthread_mutex_lock(&lock);
    val = ::nextID;
    ++::nextID;
    pthread_mutex_unlock(&lock);
    return val;
  }
};

typedef PthreadNextId NextId;
#else
typedef AtomicNextId NextId;
#endif
}

static NextId next;

void GaloisRuntime::LL::initTID() {
  TID = next.next();
  assert(TID < getMaxThreads());
}
