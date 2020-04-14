/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

#ifndef GALOIS_SUBSTRATE_THREAD_RW_LOCK_H
#define GALOIS_SUBSTRATE_THREAD_RW_LOCK_H

#include "galois/config.h"
#include "galois/substrate/PaddedLock.h"
#include "galois/substrate/PerThreadStorage.h"

namespace galois {
namespace substrate {

class ThreadRWlock {

  typedef substrate::PaddedLock<true> Lock_ty;
  // typedef galois::runtime::LL::SimpleLock<true> Lock_ty;
  typedef substrate::PerThreadStorage<Lock_ty> PerThreadLock;

  PerThreadLock locks;

public:
  void readLock() { locks.getLocal()->lock(); }

  void readUnlock() { locks.getLocal()->unlock(); }

  void writeLock() {
    for (unsigned i = 0; i < locks.size(); ++i) {
      locks.getRemote(i)->lock();
    }
  }

  void writeUnlock() {
    for (unsigned i = 0; i < locks.size(); ++i) {
      locks.getRemote(i)->unlock();
    }
  }
};

//! readOrUpdate is a generic function to perform reads or writes using a
//! rwmutex \param rwmutex is a read/write lock that implements
//! readLock/readUnlock, writeLoack/writeUnlock \param readAndCheck is function
//! object to execute when reading. It returns true only if read was successful.
//! Should update state to store read result. Shouldn't use rwmutex internally
//! \param write is function object to perform the write. It should update state
//! to store result after writing. Shouldn't use rwmutex internally
template <typename L, typename R, typename W>
void readUpdateProtected(L& rwmutex, R& readAndCheck, W& write) {

  rwmutex.readLock();

  if (readAndCheck()) {

    rwmutex.readUnlock();
    return;

  } else {

    rwmutex.readUnlock();

    rwmutex.writeLock();
    {
      // check again in case another thread made the write
      if (!readAndCheck()) {
        write();
      }
    }
    rwmutex.writeUnlock();
  }
}

} // end namespace substrate
} // end namespace galois

#endif // GALOIS_SUBSTRATE_THREAD_RW_LOCK_H
