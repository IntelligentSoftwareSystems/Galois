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

//These wrap type information for various dispatching purposes.  This
//let's us keep vtables out of user objects
class typeHelper {
protected:
  typeHelper(recvFuncTy rRP, recvFuncTy rOP, recvFuncTy lRP, recvFuncTy lOP)
    : remoteRequestPad(rRP), remoteObjectPad(rOP), localRequestPad(lRP), localObjectPad(lOP)
  {}
public:
  recvFuncTy remoteRequestPad;
  recvFuncTy remoteObjectPad;
  recvFuncTy localRequestPad;
  recvFuncTy localObjectPad;

  virtual Lockable* duplicate(Lockable* obj) const = 0;
};

template<typename T>
class typeHelperImpl : public typeHelper {
  typeHelperImpl();
public:
  virtual Lockable* duplicate(Lockable* obj) const {
    return new T(*static_cast<T*>(obj));
  }

  static typeHelperImpl* get() {
    static typeHelperImpl th;
    return &th;
  }
};


//handle local objects
class LocalDirectory {
  struct metadata {
    //Lock protecting this structure
    LL::SimpleLock lock;
    //Locations which have the object in RO state
    std::set<uint32_t> locRO;
    //Location which has the object in RW state
    uint32_t locRW;
    //Lowest if for which a recall has been sent
    uint32_t recalledFor;
    //outstanding requests
    std::set<uint32_t> reqsRO;
    std::set<uint32_t> reqsRW;
    //Type aware helper functions
    typeHelper* t;

    metadata() :locRW(~0), recalledFor(~0), t(nullptr) {}
  };

  std::unordered_map<Lockable*, metadata> dir;
  LL::SimpleLock dir_lock;
  
  metadata* getMD(Lockable*);
  metadata& getOrCreateMD(Lockable*);

  std::atomic<int> outstandingReqs;

  void recvRequestImpl(fatPointer ptr, ResolveFlag flag, uint32_t dest, typeHelper* th);

  void recvObjectImpl(fatPointer ptr);

public:
  //Recieve a request
  template<typename T>
  static void recvRequest(RecvBuffer& buf);
  //Recieve an object
  template<typename T>
  static void recvObject(RecvBuffer& buf);

  void makeProgress();
  void dump();
};

LocalDirectory& getLocalDirectory();

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

//Generic landing pad for requests
template<typename T>
void LocalDirectory::recvRequest(RecvBuffer& buf) {
  fatPointer ptr;
  ResolveFlag flag;
  uint32_t dest;
  gDeserialize(buf, ptr, flag, dest);
  getLocalDirectory().recvRequestImpl(ptr, flag, dest, typeHelperImpl<T>::get());
}

template<typename T>
void LocalDirectory::recvObject(RecvBuffer& buf) {
  fatPointer ptr;
  gDeserialize(buf, ptr);
  T* obj = static_cast<T*>(static_cast<Lockable*>(ptr.getObj()));
  //FIXME: assert object is locked by directory
  gDeserialize(buf, *obj);
  //std::cerr << "ObjRecv on " << NetworkInterface::ID << " getting " << ptr.getObj() << "\n";
  getLocalDirectory().recvObjectImpl(ptr);
}

////////////////////////////////////////////////////////////////////////////////


class RemoteDirectory {

  //metadata for an object.
  struct metadata {
    enum StateFlag {
      INVALID=0,    //Not present and not requested
      PENDING_RO=1, //Not present and requested RO
      PENDING_RW=2, //Not present and requested RW
      HERE_RO=3,         //present as RO
      HERE_RW=4,         //present as RW
      UPGRADE=5     //present as RO and requested RW
    };
    LL::SimpleLock lock;
    StateFlag state;
    Lockable* obj;

    metadata() :state(INVALID), obj(nullptr) {}

    void dump(std::ostream& os) const;
  };

  std::unordered_map<fatPointer, metadata> md;
  LL::SimpleLock md_lock;

  std::deque<std::pair<fatPointer, Lockable*>> writeback;

  //get metadata for pointer
  metadata* getMD(fatPointer ptr);

  //Recieve OK to upgrade RO -> RW
  void recvUpgrade(fatPointer ptr);

  //Recieve request to invalidate object
  void recvInvalidate(fatPointer ptr);

  //Recieve request to transition from RW -> RO
  void recvDemote(fatPointer ptr, typeHelper* th);

  //Dispatch request
  void recvRequestImpl(fatPointer ptr, ResolveFlag flag, typeHelper* th);

  //handle object ariving
  void recvObjectImpl(fatPointer ptr, ResolveFlag flag, Lockable* obj);

  //handle missing objects (or needing to upgrade)
  void resolveNotPresent(fatPointer ptr, ResolveFlag flag, metadata* md, typeHelper* th);

public:
  //Remote portion of API

  //Recieve a request
  template<typename T>
  static void recvRequest(RecvBuffer& buf);
  //Recieve an object
  template<typename T>
  static void recvObject(RecvBuffer& buf);

  //Local portion of API

  void makeProgress();

  template<typename T>
  T* resolve(fatPointer ptr, ResolveFlag flag);

  void setContended(fatPointer ptr);
  void clearContended(fatPointer ptr);
  void dump(fatPointer ptr); //dump one object info
  void dump(); //dump direcotry status
};

RemoteDirectory& getRemoteDirectory();

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template<typename T>
T* RemoteDirectory::resolve(fatPointer ptr, ResolveFlag flag) {
  trace("RemoteDirectory::resolve % %\n", ptr, flag);
  metadata* md = getMD(ptr);
  std::lock_guard<LL::SimpleLock> lg(md->lock);
  if ((flag == RO && (md->state == metadata::HERE_RO || md->state == metadata::UPGRADE)) ||
      (flag == RW &&  md->state == metadata::HERE_RW)) {
    assert(md->obj);
    return static_cast<T*>(md->obj);
  } else {
    resolveNotPresent(ptr, flag, md, typeHelperImpl<T>::get());
    return nullptr;
  }
}

//Generic landing pad for requests
template<typename T>
void RemoteDirectory::recvRequest(RecvBuffer& buf) {
  fatPointer ptr;
  ResolveFlag flag;
  gDeserialize(buf, ptr, flag);
  getRemoteDirectory().recvRequestImpl(ptr, flag, typeHelperImpl<T>::get());
}

template<typename T>
void RemoteDirectory::recvObject(RecvBuffer& buf) {
  fatPointer ptr;
  ResolveFlag flag;
  gDeserialize(buf, ptr, flag);
  assert(flag == RW || flag == RO);
  //T* obj = new T(buf);
  T* obj = new T();
  gDeserialize(buf, *obj);
  getRemoteDirectory().recvObjectImpl(ptr, flag, static_cast<Lockable*>(obj));
}

////////////////////////////////////////////////////////////////////////////////

template<typename T>
typeHelperImpl<T>::typeHelperImpl()
  : typeHelper(&RemoteDirectory::recvRequest<T>,
               &RemoteDirectory::recvObject<T>,
               &LocalDirectory::recvRequest<T>,
               &LocalDirectory::recvObject<T>)
{}

////////////////////////////////////////////////////////////////////////////////

struct remote_ex { fatPointer ptr; };

////////////////////////////////////////////////////////////////////////////////



#if 0



  LockManagerBase loaned;

  struct pendingRecv {
    fatPointer ptr;
    uint32_t dest;
    ResolveFlag flag;
    std::function<void(Directory*,fatPointer,uint32_t,ResolveFlag)> func;
    std::set<uint32_t> waiting;
  };
  std::deque<pendingRecv> pending;

  metadata* getMD(fatPointer ptr) {
    LL::SLguard lg(md_lock);
    return &md[ptr];
  }

  template<typename T>
  void request(fatPointer ptr, ResolveFlag flag) {
    request<T>(ptr, flag, ptr.getHost());
  }

  template<typename T>
  void request(fatPointer ptr, ResolveFlag flag, uint32_t reqTo) {
    SendBuffer sbuf;
    gSerialize(sbuf, ptr, NetworkInterface::ID, flag);
    getSystemNetworkInterface().send(reqTo, recvRequest<T>, sbuf);
    std::cerr << "REQUEST SENT on " << NetworkInterface::ID << " to " << reqTo << " for " << ptr.getObj() << "\n";
  }

  //Generic landing pad for objects
  template<typename T>
  static void recvObj(RecvBuffer& buf);

  //Generic landing pad for requests
  template<typename T>
  static void recvRequest(RecvBuffer& buf);

  template<typename T>
  void recvRequestImpl(fatPointer ptr, uint32_t dest, ResolveFlag flag);

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

  bool isRemote(Lockable* ptr) {
    return loaned.isAcquired(ptr);
  }

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

  void setContended(fatPointer)   {}
  void clearContended(fatPointer) {}
  void queryObj(fatPointer ptr, bool forward = true) {}
  void dump() {}
  void makeProgress() {
    if (!pending.empty()) {
      auto& foo = pending.front();
      //std::cerr << "Pending " << foo.ptr.getObj() << "\n";
      foo.func(this,foo.ptr, foo.dest, foo.flag);
      pending.pop_front();
    }
  }

  void dump(std::ostream& os, fatPointer ptr) {
    metadata* md = getMD(ptr);
    LL::SLguard lg(md->lock);
    md->dump(os);
  }
  
};

Directory& getSystemDirectory();

//Generic landing pad for objects
template<typename T>
void Directory::recvObj(RecvBuffer& buf) {
  fatPointer ptr;
  ResolveFlag flag;
  T* actual = new T();
  gDeserialize(buf, ptr, flag, *actual);
  std::cerr << "ObjRecv on " << NetworkInterface::ID << " getting " << ptr.getObj() << "\n";
  getSystemDirectory().recvObjImpl(ptr, actual, flag);
}

//Generic landing pad for requests
template<typename T>
void Directory::recvRequest(RecvBuffer& buf) {
  fatPointer ptr;
  uint32_t dest;
  ResolveFlag flag;
  gDeserialize(buf, ptr, dest, flag);
  std::cerr << "REQUEST RECV on " << NetworkInterface::ID << " for " << ptr.getObj() << "\n";
  getSystemDirectory().recvRequestImpl<T>(ptr, dest, flag);
}

template<typename T>
void Directory::recvRequestImpl(fatPointer ptr, uint32_t dest, ResolveFlag flag) {
  auto thisfunc = std::mem_fn(&Directory::recvRequestImpl<T>);
  metadata* md = getMD(ptr);
  if (ptr.getHost() == NetworkInterface::ID) { //local obj
    auto aq = loaned.tryAcquire(static_cast<T*>(ptr.getObj()));
    switch (aq) {
    case LockManagerBase::FAIL: {
      pending.push_back(pendingRecv{ptr, dest, flag, thisfunc});
      break;
    }
    case LockManagerBase::NEW_OWNER: {
      SendBuffer sbuf;
      gSerialize(sbuf, ptr, flag, *static_cast<T*>(static_cast<Lockable*>(ptr.getObj())));
      getSystemNetworkInterface().send(dest, recvObj<T>, sbuf);
      md->location = dest;
      break;
    }
    case LockManagerBase::ALREADY_OWNER: {
      if (md->location != dest) {
        request<T>(ptr, flag, md->location);
        pending.push_back(pendingRecv{ptr, dest, flag, thisfunc});
      }
      break;
    }
    }
  } else { //remote obj
    //Lookup obj pointer
    metadata* md = getMD(ptr);
    Lockable* lptr = md->obj;
    auto aq = loaned.tryAcquire(lptr);
    //try locking
    switch (aq) {
    case LockManagerBase::FAIL: {
      pending.push_back(pendingRecv{ptr, dest, flag, thisfunc});
      break;
    }
    case LockManagerBase::NEW_OWNER: {
      SendBuffer sbuf;
      gSerialize(sbuf, ptr, flag, *static_cast<T*>(lptr));
      getSystemNetworkInterface().send(dest, recvObj<T>, sbuf);
      break;
    }
    case LockManagerBase::ALREADY_OWNER: {
      //A request for an object which already was sent away or isn't here yet
      //FIXME:
      pending.push_back(pendingRecv{ptr, dest, flag, thisfunc});
      break;
    }
    }
  }
}



} //Runtime
} //Galois

#if 0

//SimpleRuntimeContext& getAbortCnx();

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

#endif

//! Make progress in the network
inline void doNetworkWork() {
  if ((NetworkInterface::Num > 1)) {// && (LL::getTID() == 0)) {
    auto& net = getSystemNetworkInterface();
    net.flush();
    while (net.handleReceives()) { net.flush(); }
    getRemoteDirectory().makeProgress();
    getLocalDirectory().makeProgress();
    net.flush();
    while (getSystemNetworkInterface().handleReceives()) { net.flush(); }
  }
}


} // namespace Runtime

} // namespace Galois

#endif
