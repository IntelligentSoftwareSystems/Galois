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

Galois::Runtime::SimpleRuntimeContext& Galois::Runtime::Distributed::getTransCnx() {
  static Galois::Runtime::SimpleRuntimeContext obj;
  return obj;
}

////////////////////////////////////////////////////////////////////////////////
// Local Directory
////////////////////////////////////////////////////////////////////////////////

// NOTE: make sure that the handleReceives() is called by another thread
void LocalDirectory::recallObj(Lockable* ptr, uint32_t remote, recvFuncTy pad) {
  //LL::gDebug("LD: ", networkHostID, " recalling: ", networkHostID, " ", ptr, " from: ", remote);
  SendBuffer buf;
  gSerialize(buf,ptr,networkHostID);
  getSystemNetworkInterface().send (remote, pad, buf);
}

//FIXME: remove all blocking calls from the directory
void LocalDirectory::recall(Galois::Runtime::Lockable* ptr, bool blocking) {
  if (Galois::Runtime::isAcquiredBy(ptr, this)) {
    Lock.lock();
    if (Galois::Runtime::isAcquiredBy(ptr, this)) {
      assert(curobj.count(ptr));
      objstate& os = curobj[ptr];
      if (!os.recalled) {
	recallObj(ptr, os.sent_to, os.pad);
	os.recalled = true;
      }
    }
    Lock.unlock();
    if (!LL::getTID()) {
      do { //always handle recieves once
	getSystemNetworkInterface().handleReceives();
      } while (blocking && Galois::Runtime::isAcquiredBy(ptr, this));
    } else {
      while (blocking && Galois::Runtime::isAcquiredBy(ptr, this)) {}
    }
  }
}

void LocalDirectory::dump() {
  LL::gDebug("Local Directory ", networkHostID, " ", Lock.is_locked()
	 , " ", curobj.size(), " ", pending.size());
  std::ostringstream os;
  os << "STATE: ";
  for (auto iter = curobj.begin(); iter != curobj.end(); ++iter) {
    os << "{" << iter->first << "," << iter->second.sent_to << "," << iter->second.recalled;
    if (pending.find(iter->first) != pending.end())
      os << ";p";
    os <<"} ";
  }
  LL::gDebug(os.str());
}

void LocalDirectory::makeProgress() { 
  PendingLock.lock();
  std::set<Lockable*> toTest;
  for(auto iter = pending.begin(); iter != pending.end(); ++iter)
    toTest.insert(iter->first);
  PendingLock.unlock();

  for (auto ii = toTest.begin(), ee = toTest.end(); ii != ee; ++ii) {
    PendingLock.lock();
    auto iter = pending.find(*ii);
    if (iter != pending.end()) {
      auto func = iter->second;
      pending.erase(iter);
      PendingLock.unlock();
      func();
    } else { 
      PendingLock.unlock();
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

RemoteDirectory& Galois::Runtime::Distributed::getSystemRemoteDirectory() {
  static RemoteDirectory obj;
  return obj;
}

LocalDirectory& Galois::Runtime::Distributed::getSystemLocalDirectory() {
  static LocalDirectory obj;
  return obj;
}

PersistentDirectory& Galois::Runtime::Distributed::getSystemPersistentDirectory() {
  static PersistentDirectory obj;
  return obj;
}




////////////////////////////////////////////////////////////////////////////////

void RemoteDirectory::dump() {
  LL::gDebug("Remote Directory ", networkHostID, " ", Lock.is_locked()
	     , " ", curobj.size(), " ", pending.size());
  std::ostringstream os;
  os << "STATE: ";
  for (auto iter = curobj.begin(); iter != curobj.end(); ++iter) {
    os << "{" << iter->first.first << "," << iter->first.second << "," << iter->second << "} }";
  }
  LL::gDebug(os.str());
}

void RemoteDirectory::clearSharedRemCache() {
  SLock.lock();
  for (auto ii = shared.begin(), ee = shared.end(); ii != ee; ++ii) {
    delete (*ii).second;
  }
  shared.clear();
  SLock.unlock();
}

void RemoteDirectory::makeProgress() {
  Lock.lock();
  std::vector<std::function<void ()>> mypending;
  mypending.swap(pending);
  Lock.unlock();
  LL::compilerBarrier(); // This appears to be needed for a gcc bug
  for (auto iter = mypending.begin(); iter != mypending.end(); ++iter) {
 // assert(!Lock.is_locked());
    (*iter)();
  }
}

////////////////////////////////////////////////////////////////////////////////

static void clearSharedCache_landing_pad(RecvBuffer& buf) {
  getSystemRemoteDirectory().clearSharedRemCache();
}

void Galois::Runtime::Distributed::clearSharedCache() {
  // should be called from outside the ForEach loop
  assert(!Galois::Runtime::inGaloisForEach);
  if (Galois::Runtime::Distributed::networkHostNum > 1) {
    SendBuffer b;
    getSystemNetworkInterface().broadcast(clearSharedCache_landing_pad,b);
  }
  getSystemRemoteDirectory().clearSharedRemCache();
  return;
}

