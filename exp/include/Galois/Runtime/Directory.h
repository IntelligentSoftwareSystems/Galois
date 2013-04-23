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
#include <map>
#include <set>

namespace Galois {
namespace Runtime {
namespace Distributed {

SimpleRuntimeContext& getAbortCnx();

template<typename T>
class gptr;

template<typename a, typename b>
struct pairhash {
private:
  const std::hash<a> ah;
  const std::hash<b> bh;
public:
  pairhash() : ah(), bh() {}
  size_t operator()(const std::pair<a, b> &p) const {
    return ah(p.first) ^ bh(p.second);
  }
};

struct PreventLiveLock: public Galois::Runtime::Lockable {
  int i;
  // serialization functions
  typedef int tt_has_serialize;
  void serialize(SerializeBuffer& s) const {
    gSerialize(s,i);
  }
  void deserialize(DeSerializeBuffer& s) {
    gDeserialize(s,i);
  }
};

extern gptr<PreventLiveLock> lock_sync;

class RemoteDirectory: public SimpleRuntimeContext, private boost::noncopyable {

  Galois::Runtime::LL::SimpleLock<true> Lock;
  Galois::Runtime::LL::SimpleLock<true> SLock;

  std::unordered_map<std::pair<uint32_t, Lockable*>, Lockable*, 
		     pairhash<uint32_t, Lockable*> > curobj;

  std::unordered_map<std::pair<uint32_t, Lockable*>, Lockable*, 
		     pairhash<uint32_t, Lockable*> > shared;

  std::vector<std::function<void ()>> pending;

  std::unordered_map<Lockable*,std::function<Lockable* ()> > resolve_callback;

  template<typename T>
  Lockable* resolve_call(uint32_t owner, Lockable* ptr);

  template<typename T>
  void repeatRecall(uint32_t owner, Lockable* ptr);

  template<typename T>
  void doRecall(uint32_t owner, Lockable* ptr);

  template<typename T>
  void doObj(uint32_t owner, Lockable* ptr, RecvBuffer& buf);

  template<typename T>
  void doSharedObj(uint32_t owner, Lockable* ptr, RecvBuffer& buf);

  static std::pair<uint32_t, Lockable*> k(uint32_t owner, Lockable* ptr) {
    return std::make_pair(owner, ptr);
  }

public:
  RemoteDirectory(): SimpleRuntimeContext(false) {}

  // Handles incoming requests for remote objects
  template<typename T>
  static void recallLandingPad(RecvBuffer &);

  // Landing Pad for incoming remote objects
  template<typename T>
  static void objLandingPad(RecvBuffer &);

  template<typename T>
  static void sharedObjLandingPad(RecvBuffer &);

  //resolve a pointer, owner pair
  template<typename T>
  T* resolve(uint32_t, Lockable*);

  //resolve a shared object
  template<typename T>
  T* sresolve(uint32_t, Lockable*);

  //clear the shared cache
  void clearSharedRemCache();

  Lockable* get_latest(Lockable*);

  void makeProgress();

  void dump();

};

class LocalDirectory: public SimpleRuntimeContext, private boost::noncopyable {

  struct objstate {
    uint32_t sent_to;  // host with object currently or first pending requester
    recvFuncTy pad; // function to call remotely to initiate recall
    bool recalled;
  };

  Galois::Runtime::LL::SimpleLock<true> Lock;
  //use a map for stable iterators
  std::map<Lockable*, objstate> curobj;

  Galois::Runtime::LL::SimpleLock<true> PendingLock;
  std::multimap<Lockable*, std::function<void ()>> pending;

  // places a remote request for the node
  void recallObj(Lockable* ptr, uint32_t remote, recvFuncTy pad);

  template<typename T>
  void sendObj(T* ptr, uint32_t remote, bool shared);

  template<typename T>
  void doRequest(Lockable* ptr, uint32_t remote, bool shared);

  template<typename T>
  void doObj(T* obj);

public:

  LocalDirectory(): SimpleRuntimeContext(false) {}

  //recalls a local object form a remote host.  May be blocking
  void recall(Lockable* ptr, bool blocking);

  // forward the request if the state is remote
  // send the object if local and not locked, also mark as remote
  template<typename T>
  static void reqLandingPad(RecvBuffer &);

  // send the object if local, not locked and mark obj as remote
  template<typename T>
  static void objLandingPad(RecvBuffer &);

  void makeProgress();

  void dump();
};

class PersistentDirectory: public SimpleRuntimeContext, private boost::noncopyable {
  Galois::Runtime::LL::SimpleLock<true> Lock;
  std::unordered_map<std::pair<uintptr_t, uint32_t>, volatile uintptr_t, pairhash<uintptr_t, uint32_t>> perobj;

  template<typename T>
  static std::pair<uintptr_t, uint32_t> k(T* ptr, uint32_t owner) {
    return std::make_pair(reinterpret_cast<uintptr_t>(ptr), owner);
  }
  static std::pair<uintptr_t, uint32_t> k(uintptr_t ptr, uint32_t owner) {
    return std::make_pair(ptr, owner);
  }

public:

  // forward the request if the state is remote
  // send the object if local and not locked, also mark as remote
  template<typename T>
  static void reqLandingPad(RecvBuffer &);

  // send the object if local, not locked and mark obj as remote
  template<typename T>
  static void objLandingPad(RecvBuffer &);

  // resolve a pointer
  template<typename T>
  T* resolve(T* ptr, uint32_t owner);
};

RemoteDirectory& getSystemRemoteDirectory();

LocalDirectory& getSystemLocalDirectory();

PersistentDirectory& getSystemPersistentDirectory();

} //Distributed
} //Runtime
} //Galois

using namespace Galois::Runtime::Distributed; // XXX

template<typename T>
T* RemoteDirectory::resolve(uint32_t owner, Lockable* ptr) {
  assert(ptr);
  assert(owner != networkHostID);
  Lock.lock();
  Lockable*& obj = curobj[k(owner,ptr)];
  if (!obj) {
    //no copy
    obj = new T();
    bool s = try_acquire(obj);
    assert(s);
    //send request
    SendBuffer sbuf;
    bool shared = false;
    gSerialize(sbuf, ptr, networkHostID, shared);
    //LL::gDebug("RD: ", networkHostID, " requesting: ", owner, " ", ptr);
    getSystemNetworkInterface().send(owner,&LocalDirectory::reqLandingPad<T>,sbuf);
    resolve_callback[obj] = std::bind(&RemoteDirectory::resolve_call<T>,this,owner,ptr);
  }
  T* retval = static_cast<T*>(obj);
  Lock.unlock();
  return retval;
}

template<typename T>
Galois::Runtime::Lockable* RemoteDirectory::resolve_call(uint32_t owner, Lockable* ptr) {
  T* obj = resolve<T>(owner,ptr);
  return static_cast<Lockable*>(obj);
}

inline Galois::Runtime::Lockable* RemoteDirectory::get_latest(Lockable* obj) {
  Lockable* new_obj;
  if(!resolve_callback[obj]) return obj;
  new_obj = (resolve_callback[obj])();
  return new_obj;
}

template<typename T>
T* RemoteDirectory::sresolve(uint32_t owner, Lockable* ptr) {
  assert(ptr);
  assert(owner != networkHostID);
  SLock.lock();
  Lockable*& obj = shared[k(owner,ptr)];
  if (!obj) {
    //no copy
    obj = new T();
    bool s = try_acquire(obj);
    assert(s);
    //send request
    SendBuffer sbuf;
    bool shared = true;
    gSerialize(sbuf, ptr, networkHostID, shared);
    //LL::gDebug("RD: ", networkHostID, " requesting: ", owner, " ", ptr);
    getSystemNetworkInterface().send(owner,&LocalDirectory::reqLandingPad<T>,sbuf);
  }
  T* retval = static_cast<T*>(obj);
  SLock.unlock();
  return retval;
}

template<typename T>
void RemoteDirectory::repeatRecall(uint32_t owner, Lockable* ptr) {
  //LL::gDebug("RD: ", networkHostID, " repeat recall for: ", owner, " ", ptr);
  doRecall<T>(owner, ptr);
}

template<typename T>
void RemoteDirectory::doRecall(uint32_t owner, Lockable* ptr) {
  Lock.lock();
  assert(curobj.find(k(owner, ptr)) != curobj.end());
  Lockable* obj = curobj[k(owner,ptr)];
  if (int val = try_acquire(obj)) {
    trace_obj_send(owner, ptr, owner);
    //LL::gDebug("RD: ", networkHostID, " returning: ", owner, " ", ptr);
    assert(val == 1); //New owner.  Violation implies recall happened before recieve
    //we have the lock so we can send the object
    SendBuffer sbuf;
    gSerialize(sbuf, ptr, *static_cast<T*>(obj));
    getSystemNetworkInterface().send(owner,&LocalDirectory::objLandingPad<T>,sbuf);
    release(obj);
    curobj.erase(k(owner,ptr));
    //delete static_cast<T*>(obj);
  } else {
    //currently locked locally, delay recall
    pending.push_back(std::bind(&RemoteDirectory::repeatRecall<T>, this, owner, ptr) );
  }
  Lock.unlock();
}

template<typename T>
void RemoteDirectory::recallLandingPad(RecvBuffer &buf) {
  uint32_t owner;
  Lockable* ptr;
  gDeserialize(buf,ptr, owner);
  //LL::gDebug("RD: ", networkHostID, " recall for: ", owner, " ", ptr);
  getSystemRemoteDirectory().doRecall<T>(owner, ptr);
}

template<typename T>
void RemoteDirectory::doObj(uint32_t owner, Lockable* ptr, RecvBuffer& buf) {
  //LL::gDebug("RD: ", networkHostID, " receiving: ", owner, " ", ptr);
  trace_obj_recv(owner, ptr);
  Lock.lock();
  assert(curobj.find(k(owner,ptr)) != curobj.end());
  Lockable* obj = curobj[k(owner,ptr)];
  assert(obj);
  assert(isAcquiredBy(obj, this));
  gDeserialize(buf,*static_cast<T*>(obj));
  // use the object atleast once
  if (inGaloisForEach && getSystemRemoteObjects().find(obj)) {
    // swap_lock should succeed!
    if (!swap_lock(obj,&getAbortCnx()))
      abort();
    // move from Requested to Received map - callback resch_recv in ParallelWork.h
    (getSystemRemoteObjects().get_remove(obj))();
  }
  else
    release(obj);
  Lock.unlock();
}

template<typename T>
void RemoteDirectory::objLandingPad(RecvBuffer &buf) {
  // Lockable*, src, data
  Lockable* ptr;
  uint32_t owner;
  gDeserialize(buf,ptr,owner);
  getSystemRemoteDirectory().doObj<T>(owner,ptr, buf);
}

template<typename T>
void RemoteDirectory::doSharedObj(uint32_t owner, Lockable* ptr, RecvBuffer& buf) {
  //LL::gDebug("RD: ", networkHostID, " receiving: ", owner, " ", ptr);
  trace_obj_recv(owner, ptr);
  SLock.lock();
  assert(shared.find(k(owner,ptr)) != shared.end());
  Lockable* obj = shared[k(owner,ptr)];
  assert(obj);
  SLock.unlock();
  assert(isAcquiredBy(obj, this));
  gDeserialize(buf,*static_cast<T*>(obj));
  release(obj);
}

template<typename T>
void RemoteDirectory::sharedObjLandingPad(RecvBuffer &buf) {
  // Lockable*, src, data
  Lockable* ptr;
  uint32_t owner;
  gDeserialize(buf,ptr,owner);
  getSystemRemoteDirectory().doSharedObj<T>(owner,ptr, buf);
}

////////////////////////////////////////////////////////////////////////////////

//Receive a remote request for a local object
template<typename T>
void LocalDirectory::reqLandingPad(RecvBuffer &buf) {
  uint32_t remote_to;
  Lockable* ptr;
  bool shared;
  gDeserialize(buf,ptr,remote_to,shared);
  getSystemLocalDirectory().doRequest<T>(ptr, remote_to, shared);
}

template<typename T>
void LocalDirectory::doRequest(Lockable* ptr, uint32_t remote, bool shared) {
  //LL::gDebug("LD: ", networkHostID, " remote req : ", networkHostID, " ", ptr, " from: ", remote);
  int val = try_acquire(ptr);
  switch (val) {
  case 0: //local iteratrion has the lock
    PendingLock.lock();
    pending.insert(std::make_pair(ptr, std::bind(&LocalDirectory::doRequest<T>, this, ptr, remote, shared)));
    PendingLock.unlock();
    break;
  case 1: //new owner (obj is local)
    {
      if (!shared) {
        Lock.lock();
        objstate& os = curobj[ptr];
        os.pad = &RemoteDirectory::recallLandingPad<T>;
        os.sent_to = remote;
        os.recalled = false;
        sendObj(static_cast<T*>(ptr), remote, shared);
        Lock.unlock();
      } else {
	sendObj(static_cast<T*>(ptr), remote, shared);
	release(ptr);
      }
      break;
    }
  case 2: //we were the owner (obj is remote)
    {
      // shared objects can't be remote
      assert(!shared);
      PendingLock.lock();
      bool doFetch = !pending.count(ptr);
      pending.insert(std::make_pair(ptr, std::bind(&LocalDirectory::doRequest<T>, this, ptr, remote, false)));
      PendingLock.unlock();
      if (doFetch) {
	Lock.lock();
	objstate& os = curobj[ptr];
	assert(os.sent_to != remote);
	if (!os.recalled) {
	  recallObj(ptr, os.sent_to, os.pad);
	  os.recalled = true;
	}
	Lock.unlock();
      }
      break;
    }
  default:
    abort();
  }
}

//receive local data from remote host after a recall
template<typename T>
void LocalDirectory::objLandingPad(RecvBuffer &buf) {
  Lockable *ptr;
  gDeserialize(buf,ptr);
  gDeserialize(buf,*static_cast<T*>(ptr)); //We know T is the actual object type
  getSystemLocalDirectory().doObj(static_cast<T*>(ptr));
}

template<typename T>
void LocalDirectory::doObj(T* ptr) {
  trace_obj_recv(networkHostID, static_cast<Lockable*>(ptr));
  //LL::gDebug("LD: ", networkHostID, " recieving: ", networkHostID, " ", ptr);
  Lock.lock();
  assert(curobj.count(ptr) && "Obj not in directory");
  curobj.erase(ptr);
  // use the object atleast once
  if (inGaloisForEach && getSystemRemoteObjects().find(ptr)) {
    // swap_lock should succeed!
    if (!swap_lock(ptr,&getAbortCnx()))
      abort();
    // move from Requested to Received map - callback resch_recv in ParallelWork.h
    (getSystemRemoteObjects().get_remove(ptr))();
  }
  else
    release(ptr);
  Lock.unlock();
}

template<typename T>
void LocalDirectory::sendObj(T* ptr, uint32_t dest, bool shared) {
  trace_obj_send(networkHostID, static_cast<Lockable*>(ptr), dest);
  //LL::gDebug("LD: ", networkHostID, " sending : ", networkHostID, " ", ptr, " to: ", dest);
  SendBuffer sbuf;
  gSerialize(sbuf, static_cast<Lockable*>(ptr), networkHostID, *ptr);
  if (shared)
    getSystemNetworkInterface().send(dest, &RemoteDirectory::sharedObjLandingPad<T>, sbuf);
  else
    getSystemNetworkInterface().send(dest, &RemoteDirectory::objLandingPad<T>, sbuf);
}

////////////////////////////////////////////////////////////////////////////////

// should be blocking if not in for each
template<typename T>
T* PersistentDirectory::resolve(T* ptr, uint32_t owner) {
  assert(ptr);
  assert(owner != networkHostID);
  Lock.lock();
  auto iter = perobj.find(k(ptr,owner));
  if (iter != perobj.end() && iter->second) {
    T* rptr = reinterpret_cast<T*>(iter->second);
    Lock.unlock();
    return rptr;
  }
  //Lock.lock();
  if (iter == perobj.end()) {
    //first request
    perobj[k(ptr,owner)] = 0;
    SendBuffer buf;
    gSerialize(buf, ptr, networkHostID);
    getSystemNetworkInterface().send(owner, &reqLandingPad<T>, buf);
  } //else someone already sent the request
  //wait until valid without holding lock
  Lock.unlock();
  do {
    if (!Galois::Runtime::inGaloisForEach || !LL::getTID())
      getSystemNetworkInterface().handleReceives();
    Lock.lock();
    T* rptr = reinterpret_cast<T*>(perobj[k(ptr,owner)]);
    Lock.unlock();
    if (rptr)
      return rptr;
  } while(true);
}

// everytime there's a request send the object!
// always runs on the host owning the object
template<typename T>
void PersistentDirectory::reqLandingPad(RecvBuffer &buf) {
  uintptr_t ptr;
  uint32_t remote;
  gDeserialize(buf, ptr, remote);
  // object should be sent to the remote host
  SendBuffer sbuf;
  gSerialize(sbuf, ptr, networkHostID, *reinterpret_cast<T*>(ptr));
  getSystemNetworkInterface().send(remote,&PersistentDirectory::objLandingPad<T>,sbuf);
}

template<typename T>
void PersistentDirectory::objLandingPad(RecvBuffer &buf) {
  uint32_t owner;
  uintptr_t ptr;
  gDeserialize(buf, ptr, owner);
  PersistentDirectory& pd = getSystemPersistentDirectory();
  pd.Lock.lock();
  try {
    pd.perobj[k(ptr,owner)] = reinterpret_cast<uintptr_t>(new T(buf));
  } catch (...) {
    abort();
  }
  pd.Lock.unlock();
}

#endif
