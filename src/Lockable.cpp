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
LockManagerBase::tryAcquire(Galois::Runtime::Lockable* lockable, bool readonly) {
  assert(lockable);
  // XXX(ddn): Hand inlining this code makes a difference on 
  // delaunaytriangulation (GCC 4.7.2)
  if (lockable->owner.getValue() == this) {
    return ALREADY_OWNER;
  } else if (!readonly && lockable->owner.try_lock()) {
    lockable->owner.setValue(this);
    rwlocks.push_back(lockable);
    return NEW_OWNER;
  } else if (readonly) {
    if (std::binary_search(rlocks.begin(), rlocks.end(), lockable)) {
      return ALREADY_OWNER;
    } else if (lockable->owner.try_lock_shared()) {
      auto ii = std::upper_bound(rlocks.begin(), rlocks.end(), lockable);
      rlocks.emplace(ii, lockable);
      return NEW_OWNER;
    }
  }
  return FAIL;
}

// LockManagerBase* LockManagerBase::forceAcquire(Lockable*) {
//   abort();
//   return nullptr;
// }

bool LockManagerBase::isAcquired(const Lockable* lockable) const {
  assert(lockable);
  if (lockable->owner.is_locked() && lockable->owner.getValue() == this)
    return true;
  if (lockable->owner.is_locked_shared() && std::binary_search(rlocks.begin(), rlocks.end(), lockable))
    return true;
  return false;
}

bool LockManagerBase::isAcquiredAny(const Lockable* lockable) {
  assert(lockable);
  return lockable->owner.getValue() != nullptr;
}

bool LockManagerBase::empty() const {
  return rlocks.empty() && rwlocks.empty();
}

// bool LockManagerBase::emptyChecked() {
//   if (empty()) return true;
//   for (auto ii = rwlocks.begin(), ee = rwlocks.end(); ii != ee; ++ii)
//     if ((*ii)->owner.getValue() == this)
//       return false;
//   return true;
// }

void LockManagerBase::releaseOne(Lockable* lockable) {
  assert(lockable);
  assert(isAcquired(lockable));
  if (lockable->owner.is_locked()) {
    lockable->owner.unlock_and_clear();
    auto ii = std::find(rwlocks.begin(), rwlocks.end(), lockable);
    assert(ii != rwlocks.end());
    rwlocks.erase(ii);
  } else {
    auto ii = std::lower_bound(rlocks.begin(), rlocks.end(), lockable);
    assert(ii != rlocks.end() && *ii == lockable);
    rlocks.erase(ii);
  }
}

std::pair<unsigned,unsigned> LockManagerBase::releaseAll() {
  unsigned retr = 0, retw = 0;
  for (auto p : rwlocks) {
    assert(p->owner.getValue() == this);
    p->owner.unlock_and_clear();
    ++retw;
  }
  rwlocks.clear();
  for (auto p : rlocks) {
    p->owner.unlock_shared();
    ++retr;
  }
  rlocks.clear();
  return std::make_pair(retr, retw);;
}

// unsigned LockManagerBase::releaseAllChecked() {
//   unsigned retval = 0;
//   for (auto ii = locks.begin(), ee = locks.end(); ii != ee; ++ii) {
//     if ((*ii)->owner.getValue() == this) {
//       (*ii)->owner.unlock_and_clear();
//       ++retval;
//     }
//   }
//   locks.clear();
//   return retval;
// }

void LockManagerBase::dump(std::ostream& os) {
  os << "{" << this << " R";
  for (auto p : rlocks) 
    os << " " << p;
  os << " W";
  for (auto p : rwlocks) 
    os << " " << p << "," << p->owner.getValue();
  os << "}";
}
