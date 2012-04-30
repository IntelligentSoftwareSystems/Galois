/** Per Thread Storage -*- C++ -*-
 * @file
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
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#ifndef _GALOIS_RUNTIME_PERTHREADSTORAGE_H
#define _GALOIS_RUNTIME_PERTHREADSTORAGE_H

#include <cassert>
#include "ll/TID.h"
#include "ll/HWTopo.h"
#include "Threads.h"

namespace GaloisRuntime {

namespace HIDDEN {

extern __thread char* base;
unsigned allocOffset(unsigned size);
void* getRemote(unsigned thread, unsigned offset);

static inline void* getLocal(unsigned offset) {
  char* B = base;
  assert(B);
  return &B[offset];
}

}

void initPTS();

template<typename T>
class PerThreadStorage {
protected:
  unsigned offset;

public:
  PerThreadStorage() {
    //in case we make one of these before initializing the thread pool
    //This will call initPTS for each thread if it hasn't already
    GaloisRuntime::getSystemThreadPool();

    offset = HIDDEN::allocOffset(sizeof(T));
    for (unsigned n = 0; n < ThreadPool::getActiveThreads(); ++n)
      new (HIDDEN::getRemote(n, offset)) T();
  }

  ~PerThreadStorage() {
    for (unsigned n = 0; n < ThreadPool::getActiveThreads(); ++n)
      reinterpret_cast<T*>(HIDDEN::getRemote(n, offset))->~T();
  }

  T* getLocal() {
    void* ditem = HIDDEN::getLocal(offset);
    return reinterpret_cast<T*>(ditem);
  }

  T* getRemote(unsigned int thread) {
    void* ditem = HIDDEN::getRemote(thread, offset);
    return reinterpret_cast<T*>(ditem);
  }

  unsigned size() {
    return LL::getMaxThreads();
  }
};

}

#endif
