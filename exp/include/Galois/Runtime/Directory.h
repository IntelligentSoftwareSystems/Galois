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

//Base class for common directory operations
class BaseDirectory {

protected:

  //These wrap type information for various dispatching purposes.  This
  //let's us keep vtables out of user objects
  class typeHelper {
  protected:
    typeHelper(recvFuncTy rRP, recvFuncTy rOP, recvFuncTy lRP, recvFuncTy lOP);

    virtual void vSerialize(SendBuffer&, Lockable*) const = 0;
    
    recvFuncTy remoteRequestPad;
    recvFuncTy remoteObjectPad;
    recvFuncTy localRequestPad;
    recvFuncTy localObjectPad;
    
  public:
    
    //Send Local -> Remote messages
    void invalidate(Lockable* ptr, uint32_t dest, uint32_t cause = ~0) const;
    void sendObj(Lockable* ptr, uint32_t dest, ResolveFlag flag) const;
    void upgrade(Lockable* ptr, uint32_t dest) const;
    
    //Send Remote -> Local messages
    void requestObj(fatPointer ptr, ResolveFlag flag);
    void writebackObj(fatPointer ptr, Lockable* obj);
  };
  
  template<typename T>
  class typeHelperImpl : public typeHelper {
    typeHelperImpl();
    virtual void vSerialize(SendBuffer&, Lockable*) const;
  public:
    virtual Lockable* duplicate(Lockable* obj) const {
      return new T(*static_cast<T*>(obj));
    }
    
    static typeHelperImpl* get() {
      static typeHelperImpl th;
      return &th;
    }
  };
    
  LockManagerBase dirContext;
  LL::SimpleLock dirContext_lock;

  bool dirAcquire(Lockable*);
  void dirRelease(Lockable*);
  bool dirOwns(Lockable*);

};

//handle local objects
class LocalDirectory : public BaseDirectory {
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
    bool here() {
      return locRO.empty() && locRW == ~0;
    }
  };

  std::unordered_map<Lockable*, metadata> dir;
  LL::SimpleLock dir_lock;
  
  metadata* getMD(Lockable*);
  metadata& getOrCreateMD(Lockable*);

  std::atomic<int> outstandingReqs;

  bool updateObjState(Lockable*, metadata&);

  void recvRequestImpl(fatPointer ptr, ResolveFlag flag, uint32_t dest, typeHelper* th);
  void recvObjectImpl(fatPointer ptr);

public:
  bool fetch(Lockable* ptr) {
    metadata* md = getMD(ptr);
    if (!md)
      return true;
    std::lock_guard<LL::SimpleLock> lg(md->lock, std::adopt_lock);
    if (md->here())
      return true;
    md->reqsRW.insert(NetworkInterface::ID);
    outstandingReqs = 1;
    return false;
  }

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
  gDeserialize(buf, *obj);
  getLocalDirectory().recvObjectImpl(ptr);
}

////////////////////////////////////////////////////////////////////////////////


class RemoteDirectory : public BaseDirectory {

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

  std::deque<std::tuple<fatPointer, Lockable*, typeHelper*>> writeback;

  //get metadata for pointer
  metadata* getMD(fatPointer ptr);

  //Recieve OK to upgrade RO -> RW
  void recvUpgrade(fatPointer ptr, typeHelper* th);

  //Recieve request to invalidate object
  void recvInvalidate(fatPointer ptr, uint32_t cause, typeHelper* th);

  //Dispatch request
  void recvRequestImpl(fatPointer ptr, ResolveFlag flag, uint32_t cause, typeHelper* th);

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
  //  trace("RemoteDirectory::resolve for % flag %\n", ptr, flag);
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
  uint32_t cause;
  gDeserialize(buf, ptr, flag, cause);
  getRemoteDirectory().recvRequestImpl(ptr, flag, cause, typeHelperImpl<T>::get());
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
BaseDirectory::typeHelperImpl<T>::typeHelperImpl()
  : typeHelper(&RemoteDirectory::recvRequest<T>,
               &RemoteDirectory::recvObject<T>,
               &LocalDirectory::recvRequest<T>,
               &LocalDirectory::recvObject<T>)
{}

template<typename T>
void BaseDirectory::typeHelperImpl<T>::vSerialize(SendBuffer& buf, Lockable* ptr) const {
  gSerialize(buf, *static_cast<T*>(ptr));
}

////////////////////////////////////////////////////////////////////////////////

struct remote_ex { fatPointer ptr; };

////////////////////////////////////////////////////////////////////////////////


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
