/** Galois Distributed Directory -*- C++ -*-
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
 * @author Manoj Dhanapal <madhanap@cs.utexas.edu>
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#include "Galois/Runtime/DistSupport.h"
#include "Galois/Runtime/Directory.h"
#include "Galois/Runtime/ll/TID.h"

#include <mutex>

using namespace Galois::Runtime;

SimpleRuntimeContext& Galois::Runtime::getTransCnx() {
  static PerThreadStorage<Galois::Runtime::SimpleRuntimeContext> obj;
  return *obj.getLocal();
}

SimpleRuntimeContext& Galois::Runtime::getAbortCnx() {
  static Galois::Runtime::SimpleRuntimeContext obj;
  return obj;
}

uint32_t Directory::getCurLoc(fatPointer ptr) {
  return 0;
}

void Directory::notifyWhenAvailable(fatPointer ptr, std::function<void(fatPointer)> func) {
  //FIXME: check of obj is resident
  Lock.lock();
  notifiers.insert(std::make_pair(ptr,func));
  Lock.unlock();
  //  if (checkAvailable(ptr))
  //    notify(ptr);
}

void Directory::notify(fatPointer ptr) {
  std::vector<std::function<void(fatPointer)>> V;
  Lock.lock();
  std::for_each(notifiers.lower_bound(ptr), notifiers.upper_bound(ptr),
                [&V] (decltype(*notifiers.begin())& val) { V.push_back(val.second); });
  notifiers.erase(notifiers.lower_bound(ptr), notifiers.upper_bound(ptr));
  Lock.unlock();
  std::for_each(V.begin(), V.end(), [ptr] (std::function<void(fatPointer)>& f) { f(ptr); });
}

void Directory::delayWork(std::function<void ()> f) {
  std::lock_guard<LL::SimpleLock<true>> L(Lock);
  pending.push_back(f);
}

void Directory::doPendingWork() {
  Lock.lock();
  std::vector<std::function<void ()>> mypending;
  mypending.swap(pending);
  Lock.unlock();
  LL::compilerBarrier(); // This appears to be needed for a gcc bug
  for (auto iter = mypending.begin(); iter != mypending.end(); ++iter)
    (*iter)();
}

size_t Directory::pendingSize() {
  std::lock_guard<LL::SimpleLock<true>> L(Lock);
  return pending.size();
}

Lockable* Directory::remoteResolve(fatPointer ptr) {
  assert(ptr.first != networkHostID);
  std::lock_guard<LL::SimpleLock<true> > lg(remoteLock);
  auto iter = remoteObjects.find(ptr);
  if (iter == remoteObjects.end())
    return nullptr;
  else
    return iter->second;
}

void Directory::remoteSet(fatPointer ptr, Lockable* obj) {
  assert (ptr.first != networkHostID);
  std::lock_guard<LL::SimpleLock<true> > lg(remoteLock);
  assert(!remoteObjects.count(ptr));
  remoteObjects[ptr] = obj;
}

void Directory::remoteClear(fatPointer ptr) {
  assert (ptr.first != networkHostID);
  std::lock_guard<LL::SimpleLock<true> > lg(remoteLock);
  assert(remoteObjects.count(ptr));
  remoteObjects.erase(ptr);
}


void Directory::sendRequest(fatPointer ptr, uint32_t msgTo, recvFuncTy f, uint32_t objTo) {
  SendBuffer sbuf;
  gSerialize(sbuf, ptr, objTo);
  getSystemNetworkInterface().send(msgTo, f, sbuf);
}

void Directory::fetchObj(fatPointer ptr, uint32_t msgTo, recvFuncTy f, uint32_t objTo) {
  fetchLock.lock();
  if (!pendingFetches.count(ptr)) {
    pendingFetches.insert(ptr);
    fetchLock.unlock();
    //don't need lock to send request
    sendRequest(ptr, msgTo, f, objTo);
  } else {
    fetchLock.unlock();
  }
}

void Directory::dropPending(fatPointer ptr) {
  std::lock_guard<LL::SimpleLock<true> > lg(fetchLock);
  pendingFetches.erase(ptr);
}

bool Directory::isPending(fatPointer ptr) {
  std::lock_guard<LL::SimpleLock<true> > lg(fetchLock);
  return pendingFetches.count(ptr);
}

////////////////////////////////////////////////////////////////////////////////

Directory& Galois::Runtime::getSystemRemoteDirectory() {
  return getSystemDirectory();
}

Directory& Galois::Runtime::getSystemLocalDirectory() {
  return getSystemDirectory();
}

Directory& Galois::Runtime::getSystemDirectory() {
  static Directory obj;
  return obj;
}
