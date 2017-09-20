/** TODO -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
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
 * @section Description
 *
 * TODO 
 *
 * @author <ahassaan@ices.utexas.edu>
 */

#ifndef GALOIS_SUBSTRATE_THREAD_RW_LOCK_H
#define GALOIS_SUBSTRATE_THREAD_RW_LOCK_H

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


  void readLock () {
    locks.getLocal ()->lock ();
  }

  void readUnlock () {
    locks.getLocal ()->unlock ();
  }

  void writeLock () {
    for (unsigned i = 0; i < locks.size (); ++i) {
      locks.getRemote (i)->lock ();
    }
  }

  void writeUnlock () {
    for (unsigned i = 0; i < locks.size (); ++i) {
      locks.getRemote (i)->unlock ();
    }
  }


};

//! readOrUpdate is a generic function to perform reads or writes using a rwmutex
//! \param rwmutex is a read/write lock that implements readLock/readUnlock, writeLoack/writeUnlock
//! \param readAndCheck is function object to execute when reading. It returns true only if read was successful. 
//! Should update state to store read result. Shouldn't use rwmutex internally
//! \param write is function object to perform the write. It should update state to store result after writing. 
//! Shouldn't use rwmutex internally
template <typename L, typename R, typename W>
void readUpdateProtected(L& rwmutex, R& readAndCheck, W& write) {

  rwmutex.readLock();

  if (readAndCheck()) {

    rwmutex.readUnlock();
    return;

  } else {

    rwmutex.readUnlock();

    rwmutex.writeLock(); {
      // check again in case another thread made the write 
      if (!readAndCheck()) {
        write();
      }
    } rwmutex.writeUnlock();

  }
}


} // end namespace substrate
} // end namespace galois



#endif // GALOIS_SUBSTRATE_THREAD_RW_LOCK_H
