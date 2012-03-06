/** SimpleLocks -*- C++ -*-
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
 * This contains support for SimpleLock support code.
 * See SimpleLock.h.
 * See PaddedLock.h
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
*/

#include "Galois/Runtime/ll/SimpleLock.h"
#include "Galois/Runtime/ll/PaddedLock.h"

void GaloisRuntime::LL::LockPairOrdered(SimpleLock<true>& L1, SimpleLock<true>& L2) {
  assert(&L1 != &L2);
  if (&L1 < &L2) {
    L1.lock();
    L2.lock();
  } else {
    L2.lock();
    L1.lock();
  }   
}

bool GaloisRuntime::LL::TryLockPairOrdered(SimpleLock<true>& L1, SimpleLock<true>& L2) {
  assert(&L1 != &L2);
  bool T1, T2;
  if (&L1 < &L2) {
    T1 = L1.try_lock();
    T2 = L2.try_lock();
  } else {
    T2 = L2.try_lock();
    T1 = L1.try_lock();
  }
  if (T1 && T2)
    return true;
  if (T1)
    L1.unlock();
  if (T2)
    L2.unlock();
  return false;
}

void GaloisRuntime::LL::UnLockPairOrdered(SimpleLock<true>& L1, SimpleLock<true>& L2) {
  assert(&L1 != &L2);
  if (&L1 < &L2) {
    L1.unlock();
    L2.unlock();
  } else {
    L2.unlock();
    L1.unlock();
  }   
}

void GaloisRuntime::LL::LockPairOrdered(SimpleLock<false>& L1, SimpleLock<false>& L2) {
}

bool GaloisRuntime::LL::TryLockPairOrdered(SimpleLock<false>& L1, SimpleLock<false>& L2) {
  return true;
}

void GaloisRuntime::LL::UnLockPairOrdered(SimpleLock<false>& L1, SimpleLock<false>& L2) {
}

void GaloisRuntime::LL::LockPairOrdered(PaddedLock<true>& L1, PaddedLock<true>& L2) {
  LockPairOrdered(L1.Lock.data, L2.Lock.data);
}

bool GaloisRuntime::LL::TryLockPairOrdered(PaddedLock<true>& L1, PaddedLock<true>& L2) {
  return TryLockPairOrdered(L1.Lock.data, L2.Lock.data);
}

void GaloisRuntime::LL::UnLockPairOrdered(PaddedLock<true>& L1, PaddedLock<true>& L2) {
  UnLockPairOrdered(L1.Lock.data, L2.Lock.data);
}

void GaloisRuntime::LL::LockPairOrdered(PaddedLock<false>& L1, PaddedLock<false>& L2) {
}

bool GaloisRuntime::LL::TryLockPairOrdered(PaddedLock<false>& L1, PaddedLock<false>& L2) {
  return true;
}

void GaloisRuntime::LL::UnLockPairOrdered(PaddedLock<false>& L1, PaddedLock<false>& L2) {
}
