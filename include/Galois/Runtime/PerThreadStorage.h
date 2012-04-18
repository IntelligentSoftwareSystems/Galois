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

namespace GaloisRuntime {

namespace HIDDEN {

extern __thread void** base;
unsigned allocOffset();
void* getRemote(unsigned thread, unsigned offset);

static inline void*& getLocal(unsigned offset) {
  void** B = base;
  return B[offset];
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
    initPTS();

    offset = HIDDEN::allocOffset();
  }

  //FIXME: Figure out what deallocation means

  T* getLocal() {
    void*& ditem = HIDDEN::getLocal(offset);
    return reinterpret_cast<T*>(ditem);
  }

  void setLocal(T* v) {
    HIDDEN::getLocal(offset) = reinterpret_cast<void*>(v);
  }

  T* getRemote(unsigned int thread) {
    void* ditem = HIDDEN::getRemote(thread, offset);
    return reinterpret_cast<T*>(ditem);
  }
};

}

#endif
