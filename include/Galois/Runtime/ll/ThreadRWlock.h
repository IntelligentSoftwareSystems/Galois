/** TODO -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
 * @section Description
 *
 * TODO 
 *
 * @author <ahassaan@ices.utexas.edu>
 */

#ifndef GALOIS_THREAD_RW_LOCK_H
#define GALOIS_THREAD_RW_LOCK_H

#include "Galois/Runtime/ll/PaddedLock.h"
#include "Galois/Runtime/PerThreadStorage.h"

namespace Galois {
namespace Runtime {
namespace LL {

//FIXME: nothing in LL should depend on Runtime

class ThreadRWlock {

  typedef PaddedLock<true> Lock_ty;
  // typedef Galois::Runtime::LL::SimpleLock<true> Lock_ty;
  typedef PerThreadStorage<Lock_ty> PerThreadLock;

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


} // end namespace LL
} // end namespace Runtime
} // end namespace Galois



#endif // GALOIS_THREAD_RW_LOCK_H
