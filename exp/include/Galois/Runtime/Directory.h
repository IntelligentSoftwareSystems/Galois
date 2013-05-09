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

#ifndef GALOIS_RUNTIME_DIRECTORY_H
#define GALOIS_RUNTIME_DIRECTORY_H

#include "Galois/Runtime/ll/TID.h"
#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/Network.h"
#include "Galois/Runtime/Tracer.h"
#include "Galois/Runtime/Support.h"
#include "Galois/Runtime/ll/SimpleLock.h"
#include "Galois/Runtime/ll/gio.h"

#include <boost/utility.hpp>

#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <functional>

namespace Galois {
namespace Runtime {

SimpleRuntimeContext& getAbortCnx();

template<typename T>
class gptr;

template<typename a>
struct pairhash {
private:
  const std::hash<typename a::first_type> ah;
  const std::hash<typename a::second_type> bh;
public:
  pairhash() : ah(), bh() {}
  size_t operator()(const a& p) const {
    return ah(p.first) ^ bh(p.second);
  }
};

typedef std::pair<uint32_t, Lockable*> fatPointer;

class Directory : public SimpleRuntimeContext, private boost::noncopyable {

  LL::SimpleLock<true> Lock;

  std::vector<std::function<void ()> > pending;
  std::multimap<fatPointer, std::function<void (fatPointer)> > notifiers;

  uint32_t getCurLoc(fatPointer);

  //////////////////////////////////////////////////////////////////////////////
  // remote handling
  //////////////////////////////////////////////////////////////////////////////

  LL::SimpleLock<true> remoteLock;
  std::unordered_map<fatPointer, Lockable*, pairhash<fatPointer> > remoteObjects;

  //resolve remote pointer
  Lockable* remoteResolve(fatPointer ptr);

  //set the resolution of ptr to obj
  void remoteSet(fatPointer ptr, Lockable* obj);

  //clear the resolution of ptr
  void remoteClear(fatPointer ptr);

  //////////////////////////////////////////////////////////////////////////////
  // fetch handling
  //////////////////////////////////////////////////////////////////////////////

  LL::SimpleLock<true> fetchLock;
  std::unordered_set<fatPointer, pairhash<fatPointer> > pendingFetches;

  //Central request sender
  //by default, request object be sent to this host
  void sendRequest(fatPointer ptr, uint32_t msgTo, recvFuncTy f, uint32_t objTo = networkHostID);

  //Fetch object.  Checks for duplicate requests
  //by default, request object be sent to this host
  void fetchObj(fatPointer ptr, uint32_t msgTo, recvFuncTy f, uint32_t objTo = networkHostID);

  //mark an object as recieved
  void dropPending(fatPointer ptr);

  //check if object is pending
  bool isPending(fatPointer ptr);

protected:
  void delayWork(std::function<void ()> f);
  void doPendingWork();
  size_t pendingSize();
  void notify(fatPointer ptr);

  //Generic landing pad for requests
  template<typename T>
  static void recvRequest(RecvBuffer&);

  template<typename T>
  void doRequest(fatPointer ptr, uint32_t remoteHost) {
    //requested objects can be forwarded when they arrive
    if (isPending(ptr)) {
      notifyWhenAvailable(ptr, std::bind(&Directory::doRequest<T>, this, std::placeholders::_1, remoteHost));
      return;
    }

    //Get the object
    T* obj = nullptr;
    if (ptr.first == networkHostID) {
      obj = static_cast<T*>(ptr.second);
    } else {
      Lockable* tmp = remoteResolve(ptr);
      if (tmp)
        obj = static_cast<T*>(tmp);
    }

    if (obj) {
      int val = SimpleRuntimeContext::try_acquire(obj);
      switch(val) {
      case 0: { // Local iteration has the lock
        delayWork(std::bind(&Directory::doRequest<T>, this, ptr, remoteHost));
      } break;
      case 1: { // Now owner (was free before)
        SendBuffer sbuf;
        gSerialize(sbuf, ptr, *static_cast<T*>(obj));
        getSystemNetworkInterface().send(remoteHost, &recvObj<T>, sbuf);
        if (ptr.first != networkHostID) {
          remoteClear(ptr);
          delete obj;
        }
      } break;
      default: // already owner or unknown condition
        abort();
      }
    } else {
      //we don't have the object, forward request to owner
      sendRequest(ptr, ptr.first, &recvRequest<T>, remoteHost);
    }
  }

  //Generic landing pad for objects
  template<typename T>
  static void recvObj(RecvBuffer&);

  //Generic handling of received objects
  template<typename T>
  void doObj(fatPointer ptr, RecvBuffer& buf) {
    T* obj = nullptr;
    if (ptr.first == networkHostID) {
      //recieved local object back
      obj = static_cast<T*>(ptr.second);
      assert(isAcquiredBy(obj, this));
    } else {
      //recieved remote object
      obj = new T();
      SimpleRuntimeContext::try_acquire(obj);
      remoteSet(ptr, obj);
    }

    gDeserialize(buf, *obj);
    dropPending(ptr);
    release(obj);
    notify(ptr);
  }

public:
  Directory() : SimpleRuntimeContext(false) {}
  
  void notifyWhenAvailable(fatPointer ptr, std::function<void(fatPointer)> func);

  template<typename T>
  void recall(fatPointer ptr) {
    assert(ptr.first == networkHostID);
    fetchObj(ptr, getCurLoc(ptr), &recvRequest<T>);
  }

  template<typename T>
  T* resolve(fatPointer ptr) {
    assert(ptr.first != networkHostID);
    Lockable* obj = remoteResolve(ptr);
    if (!obj) {
      fetchObj(ptr, ptr.first, &recvRequest<T>);
      return nullptr;
    }
    return static_cast<T*>(obj);
  }

  void makeProgress() { doPendingWork(); }

};

Directory& getSystemRemoteDirectory();
Directory& getSystemLocalDirectory();
Directory& getSystemDirectory();

//! Make progress in the network
inline void doNetworkWork() {
  if ((networkHostNum > 1) && (LL::getTID() == 0)) {
    getSystemDirectory().makeProgress();
    getSystemNetworkInterface().handleReceives();
  }
}

} //Runtime
} //Galois

//Generic landing pad for objects
template<typename T>
void Galois::Runtime::Directory::recvObj(RecvBuffer& buf) {
  fatPointer ptr;
  gDeserialize(buf, ptr);
  trace_obj_recv(ptr.first, ptr.second);
  getSystemDirectory().doObj<T>(ptr, buf);
}

//Generic landing pad for requests
template<typename T>
void Galois::Runtime::Directory::recvRequest(RecvBuffer& buf) {
  fatPointer ptr;
  uint32_t remoteHost;
  gDeserialize(buf, ptr, remoteHost);
  getSystemDirectory().doRequest<T>(ptr, remoteHost);
}

#endif
