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
#include <boost/intrusive_ptr.hpp>

#include <mutex>
#include <unordered_map>
#include <set>
#include <functional>

namespace Galois {
namespace Runtime {

SimpleRuntimeContext& getAbortCnx();

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
typedef pairhash<fatPointer> ptrHash;

struct remote_ex { fatPointer ptr; };


// requests for objects go to owner first
// owner forwards a request to current owner to recall object
// if recall host has existential lock, only sends obj if higher prioirty host
// else obj sent to owner
// owner forwards obj to highest priority host

class Directory;

class remoteObj {
  std::atomic<unsigned> count;
public:
  remoteObj() :count(0) {}
  virtual void sendObj(recvFuncTy f, fatPointer ptr, uint32_t dest) = 0;
  virtual Lockable* getObj() = 0;
  void decLock() { --count; }
  void incLock() { ++count; }
  bool emptyLock() { return count == 0; }
};

template<typename T>
class remoteObjImpl : public remoteObj {
  friend class Directory;
  T obj;
public:
  virtual void sendObj(recvFuncTy f, fatPointer ptr, uint32_t dest) {
    SendBuffer sbuf;
    gSerialize(sbuf, ptr, obj);
    getSystemNetworkInterface().send(dest, f, sbuf);
  }
  virtual Lockable* getObj() {
    return &obj;
  }
};

inline void intrusive_ptr_add_ref(remoteObj* ptr) {
  ptr->incLock();
}

inline void intrusive_ptr_release(remoteObj* ptr) {
  ptr->decLock();
}


//track here to send objects
class fetchTracking {
  LL::SimpleLock<true> fetchLock;
  std::unordered_map<fatPointer, std::set<uint32_t>, pairhash<fatPointer> > requestors;
  
public:
  // add the request, returning the current lowest
  uint32_t addRequest(fatPointer ptr, uint32_t host) {
    std::lock_guard<LL::SimpleLock<true> > lg(fetchLock);
    auto& entry = requestors[ptr];
    entry.insert(host);
    return *(entry.begin());
  }
  
  //del the request
  void delRequest(fatPointer ptr, uint32_t host) {
    std::lock_guard<LL::SimpleLock<true> > lg(fetchLock);
    auto iter = requestors.find(ptr);
    if (iter != requestors.end()) {
      iter->second.erase(host);
      if (iter->second.empty())
        requestors.erase(iter);
    }
  }
  
  void clearRequest(fatPointer ptr) {
    std::lock_guard<LL::SimpleLock<true> > lg(fetchLock);
    requestors.erase(ptr);
  }

  std::pair<bool, uint32_t> getRequest(fatPointer ptr) {
    std::lock_guard<LL::SimpleLock<true> > lg(fetchLock);
    auto iter = requestors.find(ptr);
    if (iter != requestors.end() && !iter->second.empty())
      return std::make_pair(true, *iter->second.begin());
    return std::make_pair(false, 0);
  }
};

class Directory : public SimpleRuntimeContext, private boost::noncopyable {

  template<unsigned bit> bool setBit(uintptr_t& val) {
    bool retval = val & (1UL << bit);
    if (!retval)
      val |= (1UL << bit);
    return retval;
  }

  template<unsigned bit> bool getBit(uintptr_t& val) {
    return val & (1UL << bit);
  }

  template<unsigned bit> bool delBit(uintptr_t& val) {
    bool retval = val & (1UL << bit);
    if (retval)
      val ^= (1UL << bit);
    return retval;
  }

  //////////////////////////////////////////////////////////////////////////////
  // remote obj handling
  //////////////////////////////////////////////////////////////////////////////

  LL::SimpleLock<true> remoteLock;
  std::unordered_map<fatPointer, boost::intrusive_ptr<remoteObj>, pairhash<fatPointer> > remoteObjects;

  template<typename T>
  boost::intrusive_ptr<remoteObjImpl<T>> remoteResolve(fatPointer ptr) {
    std::lock_guard<LL::SimpleLock<true>> lg(remoteLock);
    boost::intrusive_ptr<remoteObj>& retval = remoteObjects[ptr];
    if (!retval) {
      auto t = new remoteObjImpl<T>();
      retval = t;
      SimpleRuntimeContext::try_acquire(&t->obj);
    }
    return boost::static_pointer_cast<remoteObjImpl<T>>(retval);
  }

  //this may return a null pointer equivalent
  boost::intrusive_ptr<remoteObj> remoteWeakResolve(fatPointer ptr) {
    std::lock_guard<LL::SimpleLock<true>> lg(remoteLock);
    boost::intrusive_ptr<remoteObj> retval;
    if ( remoteObjects.count(ptr))
      retval = remoteObjects[ptr];
    return retval;
  }

  void remoteClear(fatPointer ptr) {
    std::lock_guard<LL::SimpleLock<true>> lg(remoteLock);
    assert(remoteObjects.count(ptr));
    remoteObjects.erase(ptr);
  }

  //May return a null pointer
  std::pair<Lockable*, boost::intrusive_ptr<remoteObj>> getObjUntyped(fatPointer ptr);

  //always returns a valid pointer
  template<typename T>
  std::pair<T*, boost::intrusive_ptr<remoteObjImpl<T> > > getObj(fatPointer ptr) {
    //Get the object
    std::pair<T*, boost::intrusive_ptr<remoteObjImpl<T> > > retval;
    if (ptr.first == networkHostID) {
      retval.first = static_cast<T*>(ptr.second);
    } else {
      retval.second = remoteResolve<T>(ptr);
      retval.first = &retval.second->obj;
    }
    return retval;
  }

  //////////////////////////////////////////////////////////////////////////////
  // fetch handling
  //////////////////////////////////////////////////////////////////////////////
  bool getPending(Lockable* ptr) {
    return getBit<32>(ptr->auxData);
  }

  bool delPending(Lockable* ptr) {
    return delBit<32>(ptr->auxData);
  }

  bool setPending(Lockable* ptr) {
    return setBit<32>(ptr->auxData);
  }

  bool getPriority(Lockable* ptr) {
    return getBit<33>(ptr->auxData);
  }

  bool delPriority(Lockable* ptr) {
    return delBit<33>(ptr->auxData);
  }

  bool setPriority(Lockable* ptr) {
    return setBit<33>(ptr->auxData);
  }

  uint32_t getCurLoc(fatPointer ptr) {
    if (ptr.first == networkHostID) {
      return (uint32_t) ptr.second->auxData;
    }
    return ptr.first;
  }
  
  void setCurLoc(fatPointer ptr, uint32_t host) {
    if (ptr.first == networkHostID) {
      uintptr_t val = ptr.second->auxData >> 32;
      val <<= 32;
      val |= host;
      ptr.second->auxData = val;
    }
  }

  fetchTracking fetches;

  //////////////////////////////////////////////////////////////////////////////
  // external notification
  //////////////////////////////////////////////////////////////////////////////

  LL::SimpleLock<true> Lock;
  std::multimap<fatPointer, std::function<void (fatPointer)> > notifiers;
  void notify(fatPointer ptr);
  void tryNotifyAll();

  //////////////////////////////////////////////////////////////////////////////
  // pending handling
  //////////////////////////////////////////////////////////////////////////////

  //Central request sender
  //by default, request object be sent to this host
  void sendRequest(fatPointer ptr, uint32_t msgTo, recvFuncTy f, uint32_t objTo = networkHostID);
  //Fetch object.  Checks for duplicate requests
  //by default, request object be sent to this host
  void fetchObj(Lockable* obj, fatPointer ptr, uint32_t msgTo, recvFuncTy f, uint32_t objTo = networkHostID);

  //////////////////////////////////////////////////////////////////////////////
  // Communication
  //////////////////////////////////////////////////////////////////////////////

  //Generic landing pad for requests
  template<typename T>
  static void recvRequest(RecvBuffer&);

  template<typename T>
  void doRequest(fatPointer ptr, uint32_t remoteHost) {
     //Get the object
    std::pair<T*, boost::intrusive_ptr<remoteObjImpl<T>> > obj = getObj<T>(ptr);

    //delay processing while we have an existential lock and request is low priority
    if (remoteHost > networkHostID && getPriority(obj.first)) {
      fetches.addRequest(ptr, remoteHost);
      return;
    }

    int val = SimpleRuntimeContext::try_acquire(obj.first);
    switch(val) {
    case 0: { // Local iteration has the lock
      fetches.addRequest(ptr, remoteHost);
    } break;
    case 1: { // Now owner (was free and on this host)
      //compute who to send the object too
      //non-owner sends back to owner
      uintptr_t dest = ptr.first;
      if (ptr.first == networkHostID) {
        //owner sends to highest priority requestor
        dest = fetches.addRequest(ptr, remoteHost);
      }
      trace_obj_send(ptr.first, ptr.second, dest);
      SendBuffer sbuf;
      gSerialize(sbuf, ptr, *static_cast<T*>(obj.first));
      getSystemNetworkInterface().send(dest, &recvObj<T>, sbuf);
      setCurLoc(ptr, dest);
      if (ptr.first == networkHostID) {
        fetches.delRequest(ptr, dest);
      } else {
        fetches.clearRequest(ptr);
        remoteClear(ptr);
      }
    } break;
    case 2: { // already owner, so obj is somewhere else
      fetchObj(obj.first, ptr, getCurLoc(ptr), &recvRequest<T>);
    } break;
    default: // unknown condition
      abort();
    }
  }

  //Generic landing pad for objects
  template<typename T>
  static void recvObj(RecvBuffer&);

  //Generic handling of received objects
  template<typename T>
  void doObj(fatPointer ptr, RecvBuffer& buf) {
    std::pair<T*, boost::intrusive_ptr<remoteObjImpl<T>> > obj = getObj<T>(ptr);
    assert(isAcquiredBy(obj.first, this));
    gDeserialize(buf, *obj.first);
    delPending(obj.first);
    release(obj.first);
    notify(ptr);
  }

public:
  Directory() : SimpleRuntimeContext(false) {}
  
  void notifyWhenAvailable(fatPointer ptr, std::function<void(fatPointer)> func);

  template<typename T>
  void recall(fatPointer ptr) {
    assert(ptr.first == networkHostID);
    assert(getCurLoc(ptr) != networkHostID);
    ////do a self request to participate in prioritization
    //doRequest<T>(ptr, networkHostID);
    fetchObj(ptr.second, ptr, getCurLoc(ptr), &recvRequest<T>);
  }

  template<typename T>
  boost::intrusive_ptr<remoteObjImpl<T>> resolve(fatPointer ptr) {
    assert(ptr.first != networkHostID);
    auto i = remoteResolve<T>(ptr);
    if (isAcquiredBy(&i->obj, this))
      fetchObj(&i->obj, ptr, ptr.first, &recvRequest<T>);
    return i;
  }

  void setContended(fatPointer ptr);
  void clearContended(fatPointer ptr);

  void makeProgress() { 
    tryNotifyAll();
  }

};

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
