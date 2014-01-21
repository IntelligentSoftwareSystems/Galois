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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Manoj Dhanapal <madhanap@cs.utexas.edu>
 */

#ifndef GALOIS_RUNTIME_DIRECTORY_H
#define GALOIS_RUNTIME_DIRECTORY_H

#include "Galois/gstl.h"
#include "Galois/Runtime/ll/TID.h"
//#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/Network.h"
#include "Galois/Runtime/Tracer.h"
#include "Galois/Runtime/Support.h"
#include "Galois/Runtime/ll/SimpleLock.h"
#include "Galois/Runtime/ll/gio.h"
#include "Galois/Runtime/FatPointer.h"
#include "Galois/Runtime/CacheManager.h"

#include <boost/utility.hpp>
#include <boost/intrusive_ptr.hpp>

#include <mutex>
#include <unordered_map>
#include <set>
#include <functional>
#include <array>

namespace Galois {
namespace Runtime {

enum ResolveFlag {INV=0, RO=1, RW=2};

class DirectoryNG {

  struct metadata {
    enum StateFlag {
      INVALID=0,    //Not present and not requested
      PENDING_RO=1, //Not present and requested RO
      PENDING_RW=2, //Not present and requested RW
      RO=3,         //present as RO
      RW=4,         //present as RW
      UPGRADE=5     //present as RO and requested RW
    };
    LL::SimpleLock lock;
    StateFlag state;
    Lockable* obj;

    metadata() :state(INVALID), obj(0) {}

    void dump(std::ostream& os) const {
      static const char* StateFlagNames[] = {"I", "PR", "PW", "RO", "RW", "UW"};
      os << "{" << StateFlagNames[state] << "," << obj << "}";
    }
  };

  std::unordered_map<fatPointer, metadata> md;
  LL::SimpleLock md_lock;

  metadata* getMD(fatPointer ptr) {
    LL::SLguard lg(md_lock);
    return &md[ptr];
  }

  template<typename T>
  void request(fatPointer ptr, ResolveFlag flag) {
    SendBuffer sbuf;
    gSerialize(sbuf, ptr, NetworkInterface::ID, flag);
    getSystemNetworkInterface().send(ptr.getHost(), recvRequest<T>, sbuf);
    std::cerr << "\nREQEST SENT\n";
  }

  //Generic landing pad for objects
  template<typename T>
  static void recvObj(RecvBuffer& buf);

  //Generic landing pad for requests
  template<typename T>
  static void recvRequest(RecvBuffer& buf);

  void recvObjImpl(fatPointer ptr, Lockable* actual, ResolveFlag flag) {
    metadata* md = getMD(ptr);
    LL::SLguard lg(md->lock);
    switch (md->state) {
    case metadata::INVALID:
      abort();
      break;
    case metadata::PENDING_RO:
      if (flag == RO) {
        md->state = metadata::RO;
      } else if (flag == RW) {
        md->state = metadata::RW;
      } else {
        abort();
      }
      md->obj = actual;
      break;
    case metadata::PENDING_RW:
      if (flag == RO) {
        md->state = metadata::UPGRADE;
      } else if (flag == RW) {
        md->state = metadata::RW;
      } else {
        abort();
      }
      md->obj = actual;
      break;
    case metadata::RO:
      if (flag == RO) {
        abort();
      } else if (flag == RW) {
        md->state = metadata::RW;
      } else if (flag == INV) {
        md->state = metadata::INVALID;
        md->obj = nullptr;
      } else {
        abort();
      }
      break;
    case metadata::RW:
      if (flag == INV) {
        md->state = metadata::INVALID;
        md->obj = nullptr;
      } else {
        abort();
      }
      break;
    case metadata::UPGRADE:
      if (flag == RW) {
        md->state = metadata::RW;
      } else if (flag == RO) {
        abort();
      } else if (flag == INV) {
        md->state = metadata::PENDING_RW;
        md->obj = nullptr;
      } else {
        abort();
      }
      break;
    default:
      abort();
    }
    assert(md->obj == actual);
  }

public:

  template<typename T>
  T* resolve(fatPointer ptr, ResolveFlag flag) {
    metadata* md = getMD(ptr);
    LL::SLguard lg(md->lock);
    switch (md->state) {
    case metadata::INVALID:
      //request object
      request<T>(ptr, flag);
      md->state = flag == RO ? metadata::PENDING_RO : metadata::PENDING_RW;
      return nullptr;
    case metadata::PENDING_RO:
      if (flag == RW) {
        request<T>(ptr, flag);
        md->state = metadata::PENDING_RW;
      }
      return nullptr;
    case metadata::PENDING_RW:
      return nullptr;
    case metadata::RO:
      if (flag == RW) { // upgrade
        request<T>(ptr, flag);
        md->state = metadata::UPGRADE;
        return nullptr;
      } else {
        return static_cast<T*>(md->obj);
      }
    case metadata::RW:
      return static_cast<T*>(md->obj);
    case metadata::UPGRADE:
      if (flag == RW) {
        return nullptr;
      } else {
        return static_cast<T*>(md->obj);
      }
    default:
      abort();
    } //switch
    abort();
    return nullptr;
  }

  void dump(std::ostream& os, fatPointer ptr) {
    metadata* md = getMD(ptr);
    LL::SLguard lg(md->lock);
    md->dump(os);
  }
  
};

DirectoryNG& getSystemDirectoryNG();

//Generic landing pad for objects
template<typename T>
void DirectoryNG::recvObj(RecvBuffer& buf) {
  fatPointer ptr;
  ResolveFlag flag;
  T* actual = new T();
  gDeserialize(buf, ptr, flag, *actual);
  std::cerr << "\nObjRecv\n";
  getSystemDirectoryNG().recvObjImpl(ptr, actual, flag);
}

//Generic landing pad for requests
template<typename T>
void DirectoryNG::recvRequest(RecvBuffer& buf) {
  fatPointer ptr;
  uint32_t dest;
  ResolveFlag flag;
  gDeserialize(buf, ptr, dest, flag);
  std::cerr << "\n REQUEST RECV\n";
  SendBuffer sbuf;
  gSerialize(sbuf, ptr, flag, *static_cast<T*>(static_cast<Lockable*>(ptr.getObj())));
  getSystemNetworkInterface().send(dest, recvObj<T>, sbuf);
}




//SimpleRuntimeContext& getAbortCnx();

struct remote_ex { fatPointer ptr; };

// requests for objects go to owner first
// owner forwards a request to current owner to recall object
// if recall host has existential lock, only sends obj if higher prioirty host
// else obj sent to owner
// owner forwards obj to highest priority host

class Directory;

//These wrap type information for various dispatching purposes.  This
//let's us keep vtables out of user objects
class typeHelper {
  template<typename T> recvFuncTy getRecvRequestImpl() const;
  template<typename T> recvFuncTy getRecvObjImpl() const;

public:
  virtual void sendObject(fatPointer ptr, Lockable* obj, uint32_t dest) const = 0;
  virtual void sendRequest(fatPointer ptr, uint32_t dest, uint32_t reqFor) const = 0;
};

template<typename T>
class typeHelperImpl : public typeHelper {
public:
  virtual void sendObject(fatPointer ptr, Lockable* obj, uint32_t dest) const;
  virtual void sendRequest(fatPointer ptr, uint32_t dest, uint32_t reqFor) const;

  static typeHelperImpl* get() {
    static typeHelperImpl th;
    return &th;
  }
};

class tracking {
  std::set<uint32_t> requests;
  typeHelper* helper;
  bool recalled;
  uint32_t recalledFor;
  uint32_t contended;
  uint32_t curLoc;

  std::vector<std::function<void(fatPointer)> > notifiers;

  //set typehelper
  void setTypeHelper(typeHelper* th) {
    assert(!helper || helper == th);
    helper = th;
  }

public:

  void addNotify(std::function<void(fatPointer)> func) {
    notifiers.push_back(func);
  }

  void notifyAll(fatPointer ptr) {
    for (auto& func : notifiers)
      func(ptr);
    notifiers.clear();
  }

  //add a request to the queue
  void addRequest(uint32_t remoteHost, typeHelper* th = nullptr) {
    requests.insert(remoteHost);
    if (th)
      setTypeHelper(th);
    assert(helper);
  }
  void delRequest(uint32_t remoteHost) {
    assert(requests.count(remoteHost));
    requests.erase(remoteHost);
  }
  void clearRequest()     { requests.clear(); }
  uint32_t getRequest()   { return *requests.begin(); }
  bool hasRequest() const { return !requests.empty(); }

  //set the object as being on this host
  void setLocal() {
    recalled = false;
    curLoc = NetworkInterface::ID;
  }

  void setCurLoc(uint32_t host) { curLoc = host; }
  uint32_t getCurLoc() const { return curLoc; }

  bool isRecalled() const { return recalled; }
  uint32_t getRecalled() const { assert(recalled); return recalledFor; }
  void setRecalled(uint32_t host) { recalled = true; recalledFor = host; }

  void setContended()      { ++contended; }
  void clearContended()    { --contended; }
  bool isContended() const { return contended; }

  typeHelper* getHelper() const { return helper; }
};


class Directory : public LockManagerBase, private boost::noncopyable {

  template<typename T>
  friend class typeHelperImpl;

  std::unordered_map<fatPointer, tracking, std::hash<fatPointer> > tracks;
  LL::SimpleLock trackLock;

  tracking& getTracking(LL::SLguard& lg, fatPointer ptr) {
    LL::SLguard lgt(trackLock);
    return tracks[ptr];
  }

  tracking& getExistingTracking(LL::SLguard& lg, fatPointer ptr) {
    LL::SLguard lgt(trackLock);
    assert(tracks.count(ptr));
    return tracks[ptr];
  }

  void delTracking(LL::SLguard& lg, fatPointer ptr) {
    LL::SLguard lgt(trackLock);
    assert(tracks.count(ptr));
    tracks.erase(ptr);
  }

  bool hasTracking(LL::SLguard& lg, fatPointer ptr) {
    LL::SLguard lgt(trackLock);
    return tracks.count(ptr);
 }

  std::vector<fatPointer> getTracks() {
    LL::SLguard lgt(trackLock);
    std::vector<fatPointer> retval;
    retval.reserve(tracks.size());
    for (auto& k : tracks)
      retval.push_back(k.first);
    return retval;
  }

  std::deque<fatPointer> pending;
  LL::SimpleLock pendingLock;

  void addPending(fatPointer ptr) {
    LL::SLguard lgp(pendingLock);
    pending.push_back(ptr);
  }

  fatPointer popPending() {
    LL::SLguard lgp(pendingLock);
    if (pending.empty())
      return fatPointer(0,0);
    fatPointer retval = pending.front();
    pending.pop_front();
    return retval;
  }

  // std::deque<fatPointer> getPending() {
  //   LL::SLguard lgp(pendingLock);
  //   std::deque<fatPointer> retval;
  //   retval.swap(pending);
  //   return retval;
  // }

  //main request processor
  void processObj(LL::SLguard& lg, fatPointer ptr, Lockable* obj);
 
  //Generic landing pad for requests
  template<typename T>
  static void recvRequest(RecvBuffer&);
  
  //update requests simply notify of a higher priority requestor.  If
  // the object has already been sent away, this does nothing
  void doRequest(fatPointer ptr, typeHelper* th, uint32_t remoteHost);

  //Generic landing pad for objects
  template<typename T>
  static void recvObj(RecvBuffer&);

  //Generic handling of received objects
  //returns whether release should happen
  bool doObj(fatPointer ptr, typeHelper* th);


  enum { numLocks = 1024 };
  std::array<LL::SimpleLock, numLocks> objLocks;
  std::hash<fatPointer> lockhash;
  LL::SimpleLock& getLock(fatPointer ptr) {
    return objLocks[lockhash(ptr) % numLocks];
  }

  template<typename T>
  static void sendObject(fatPointer ptr, Lockable* obj, uint32_t dest) {
    trace_obj_send(ptr.getHost(), ptr.getObj(), dest);
    SendBuffer sbuf;
    gSerialize(sbuf, ptr, *static_cast<T*>(obj));
    getSystemNetworkInterface().send(dest, recvObj<T>, sbuf);
  }

  template<typename T>
  static void sendRequest(fatPointer ptr, uint32_t dest, uint32_t reqFor) {
    trace_req_send(ptr.getHost(), ptr.getObj(), dest, reqFor);
    SendBuffer sbuf;
    gSerialize(sbuf, ptr, reqFor);
    getSystemNetworkInterface().send(dest, recvRequest<T>, sbuf);
  }

public:
  
  template<typename T>
  void fetch(fatPointer ptr, T* obj) {
    doRequest(ptr, typeHelperImpl<T>::get(), NetworkInterface::ID);
  }

  void setContended(fatPointer ptr);
  void clearContended(fatPointer ptr);

  void notifyWhenAvailable(fatPointer ptr, std::function<void(fatPointer)> func);

  void makeProgress() {
    auto& cm = getCacheManager();
    auto& net = getSystemNetworkInterface();
    fatPointer ptr;
    while ((ptr = popPending()) != fatPointer(0,0)) {
      auto& ptrlock = getLock(ptr);
      if (ptrlock.try_lock()) {
        LL::SLguard lg(getLock(ptr), std::adopt_lock_t());
        if (ptr.getHost() == NetworkInterface::ID) {
          processObj(lg, ptr, static_cast<Lockable*>(ptr.getObj()));
        } else {
          remoteObj* obj = cm.weakResolve(ptr);
          if (obj)
            processObj(lg, ptr, obj->getObj());
        }
      } else {
        addPending(ptr);
      }
      while (net.handleReceives()) {}
    }
  }

  void dump();

  void queryObj(fatPointer ptr, bool forward = true);
  static void queryObjRemote(fatPointer ptr, bool forward);
};

Directory& getSystemDirectory();

//! Make progress in the network
inline void doNetworkWork() {
  if ((NetworkInterface::Num > 1)) {// && (LL::getTID() == 0)) {
    auto& net = getSystemNetworkInterface();
    net.flush();
    while (net.handleReceives()) { net.flush(); }
    getSystemDirectory().makeProgress();
    net.flush();
    while (getSystemNetworkInterface().handleReceives()) { net.flush(); }
  }
}

} //Runtime
} //Galois

//Generic landing pad for objects
template<typename T>
void Galois::Runtime::Directory::recvObj(RecvBuffer& buf) {
  fatPointer ptr;
  gDeserialize(buf, ptr);
  trace_obj_recv(ptr.getHost(), ptr.getObj());
  T* obj = nullptr;
  if (ptr.getHost() == NetworkInterface::ID) {
    obj = static_cast<T*>(ptr.getObj());
  } else {
    obj = static_cast<T*>(getCacheManager().resolve<T>(ptr)->getObj());
  }
  gDeserialize(buf, *obj);
  auto& dir = getSystemDirectory();
  if (dir.doObj(ptr, typeHelperImpl<T>::get())) {
    assert(isAcquiredBy(obj, &dir));
    dir.releaseOne(obj);
  }
}

//Generic landing pad for requests
template<typename T>
void Galois::Runtime::Directory::recvRequest(RecvBuffer& buf) {
  fatPointer ptr;
  uint32_t remoteHost;
  gDeserialize(buf, ptr, remoteHost);
  trace_req_recv(ptr.getHost(), ptr.getObj(), remoteHost);
  getSystemDirectory().doRequest(ptr, typeHelperImpl<T>::get(), remoteHost);
}

template<typename T>
void Galois::Runtime::typeHelperImpl<T>::sendObject(Galois::Runtime::fatPointer ptr, Lockable* obj, uint32_t dest) const {
  Directory::sendObject<T>(ptr, obj, dest);
}

template<typename T>
void Galois::Runtime::typeHelperImpl<T>::sendRequest(fatPointer ptr, uint32_t dest, uint32_t reqFor) const {
  Directory::sendRequest<T>(ptr, dest, reqFor);
}

#if 0
template<typename T>
Galois::Runtime::remoteObjImpl<T>* Galois::Runtime::CacheManager::resolve(fatPointer ptr) {
  assert(ptr.getHost() != NetworkInterface::ID);
  LL::SLguard lgr(Lock);
  remoteObj*& retval = remoteObjects[ptr];
  if (!retval) {
    auto t = new remoteObjImpl<T>();
    retval = t;
    auto& dir = getSystemDirectory();
    dir.tryAcquire(t->getObj());
  }
  return static_cast<remoteObjImpl<T>*>(retval);
}
#endif

#endif
