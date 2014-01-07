/** Basic galois contention manager base classes -*- C++ -*-
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
 * @section Description
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#include "Galois/Runtime/Lockable.h"

#include <cassert>

using namespace Galois::Runtime;

LockManagerBase::AcquireStatus
LockManagerBase::tryAcquire(Galois::Runtime::Lockable* lockable) {
  assert(lockable);
  // XXX(ddn): Hand inlining this code makes a difference on 
  // delaunaytriangulation (GCC 4.7.2)
  if (lockable->owner.getValue() == this) {
    return ALREADY_OWNER;
  } else if (lockable->owner.try_lock()) {
    lockable->owner.setValue(this);
    locks.push_back(lockable);
    return NEW_OWNER;
  }
  return FAIL;
}

LockManagerBase* LockManagerBase::forceAcquire(Galois::Runtime::Lockable* lockable) {
  assert(lockable);
  do {
    auto r = tryAcquire(lockable);
    switch (r) {
    case ALREADY_OWNER:
      return this;
    case NEW_OWNER:
      locks.push_back(lockable);
      return nullptr;
    case FAIL: {
      LockManagerBase* retval = lockable->owner.getValue();
      if (lockable->owner.stealing_CAS(retval, this)) {
        locks.push_back(lockable);
        return retval;
      }
      break;
    }
    }
  } while (true);
}

unsigned LockManagerBase::releaseAll() {
  unsigned retval = 0;
  for (auto ii = locks.begin(), ee = locks.end(); ii != ee; ++ii) {
    assert((*ii)->owner.getValue() == this);
    (*ii)->owner.unlock_and_clear();
    ++retval;
  }
  locks.clear();
  return retval;
}

unsigned LockManagerBase::releaseAllChecked() {
  unsigned retval = 0;
  for (auto ii = locks.begin(), ee = locks.end(); ii != ee; ++ii) {
    if ((*ii)->owner.getValue() == this) {
      (*ii)->owner.unlock_and_clear();
      ++retval;
    }
  }
  locks.clear();
  return retval;
}

void LockManagerBase::dump(std::ostream& os) {
  os << "{" << this << ":";
  for (auto ii = locks.begin(), ee = locks.end(); ii != ee; ++ii)
    os << " " << *ii << "," << (*ii)->owner.getValue();
  os << "}";
}
