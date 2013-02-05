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

#include <boost/unordered_map.hpp>
#include "Galois/Runtime/ll/TID.h"
#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/Network.h"
#include "Galois/Runtime/Support.h"
#include "Galois/Runtime/ll/SimpleLock.h"

#define INELIGIBLE_COUNT 12
using namespace std;

namespace Galois {
namespace Runtime {
namespace Distributed {

//Objects with this tag should make blocking calls
BOOST_MPL_HAS_XXX_TRAIT_DEF(tt_dir_blocking)
template<typename T>
struct dir_blocking : public has_tt_dir_blocking<T> {};

class RemoteDirectory: public SimpleRuntimeContext {

  struct objstate {
    // Remote - The object has been returned to the owner
    // Local  - Local object eligible for use as soon as received
    //          Inelgible for transfer till INELI2ELI_COUNT reqs or local use
    enum ObjStates { Remote, Local };

    uintptr_t localobj;
    enum ObjStates state;
    int count;
  };

  struct ohash : public unary_function<std::pair<uintptr_t, uint32_t>, size_t> {
    size_t operator()(const std::pair<uintptr_t, uint32_t>& v) const {
      return std::hash<uintptr_t>()(v.first) ^ std::hash<uint32_t>()(v.second);
    }
  };

  boost::unordered_map<std::pair<uintptr_t, uint32_t>, objstate, ohash> curobj;
  Galois::Runtime::LL::SimpleLock<true> Lock;

  //returns a valid local pointer to the object if it exists
  //or returns null
  uintptr_t haveObject(uintptr_t ptr, uint32_t owner);

  // places a remote request for the node
  void fetchRemoteObj(uintptr_t ptr, uint32_t owner, recvFuncTy pad);

  // tries to acquire a lock and returns true or false if acquired
  // used before sending an object and freeing it
  bool diracquire(Lockable* L);

public:

  // Handles incoming requests for remote objects
  // if Ineligible, transfer to Eligible after INELI2ELI_COUNT requests
  // if Eligible return the object back to owner and mark as Remote
  template<typename T>
  static void remoteReqLandingPad(RecvBuffer &);

  // Landing Pad for incoming remote objects
  template<typename T>
  static void remoteDataLandingPad(RecvBuffer &);

  //resolve a pointer, owner pair
  //precondition: owner != networkHostID
  template<typename T>
  T* resolve(uintptr_t ptr, uint32_t owner);
};

class LocalDirectory: public SimpleRuntimeContext {

  struct objstate {
    // Remote - Object passed to a remote host
    // Local - Local object may be locked
    enum ObjStates { Remote, Local };

    uint32_t sent_to;  // valid only for remote objects
    enum ObjStates state;
  };

  boost::unordered_map<uintptr_t, objstate> curobj;
  Galois::Runtime::LL::SimpleLock<true> Lock;

  // returns a valid local pointer to the object if not remote
  uintptr_t haveObject(uintptr_t ptr, uint32_t &remote);

  // places a remote request for the node
  void fetchRemoteObj(uintptr_t ptr, uint32_t remote, recvFuncTy pad);

  // needed for locking objects inside the LocalDirectory
  virtual void sub_acquire(Lockable* L);

  // tries to acquire a lock and returns true or false if acquired
  // used before sending an object and marking it remote
  bool diracquire(Lockable* L);

public:

  LocalDirectory(): SimpleRuntimeContext(true) {}

  // forward the request if the state is remote
  // send the object if local and not locked, also mark as remote
  template<typename T>
  static void localReqLandingPad(RecvBuffer &);

  // send the object if local, not locked and mark obj as remote
  template<typename T>
  static void localDataLandingPad(RecvBuffer &);

  // resolve a pointer
  template<typename T>
  T* resolve(uintptr_t ptr);
};

class PersistentDirectory: public SimpleRuntimeContext {

  struct objstate {
    uintptr_t localobj;
    bool requested;
  };

  struct ohash : public unary_function<std::pair<uintptr_t, uint32_t>, size_t> {
    size_t operator()(const std::pair<uintptr_t, uint32_t>& v) const {
      return std::hash<uintptr_t>()(v.first) ^ std::hash<uint32_t>()(v.second);
    }
  };

  boost::unordered_map<std::pair<uintptr_t, uint32_t>, objstate, ohash> perobj;
  Galois::Runtime::LL::SimpleLock<true> Lock;

  // returns a valid local pointer to the object if not remote
  uintptr_t haveObject(uintptr_t ptr, uint32_t owner);

  // places a remote request for the node
  void fetchRemoteObj(uintptr_t ptr, uint32_t owner, recvFuncTy pad);

public:

  // forward the request if the state is remote
  // send the object if local and not locked, also mark as remote
  template<typename T>
  static void persistentReqLandingPad(RecvBuffer &);

  // send the object if local, not locked and mark obj as remote
  template<typename T>
  static void persistentDataLandingPad(RecvBuffer &);

  // resolve a pointer
  template<typename T>
  T* resolve(uintptr_t ptr, uint32_t owner);
};

RemoteDirectory& getSystemRemoteDirectory();

LocalDirectory& getSystemLocalDirectory();

PersistentDirectory& getSystemPersistentDirectory();

} //Distributed
} //Runtime
} //Galois

using namespace Galois::Runtime::Distributed;

// should be blocking if not in for each
template<typename T>
T* RemoteDirectory::resolve(uintptr_t ptr, uint32_t owner) {
  assert(ptr);
  assert(owner != networkHostID);
  uintptr_t p = haveObject(ptr, owner);
  while (!p) {
    NetworkInterface& net = getSystemNetworkInterface();
    fetchRemoteObj(ptr, owner, &LocalDirectory::localReqLandingPad<T>);
    // abort the iteration if inside for each and dir_blocking not defined
    if (Galois::Runtime::inGaloisForEach && !dir_blocking<T>::value)
      throw remote_ex{ptr, owner};
    p = haveObject(ptr, owner);
    // call handleReceives if only thread outside for_each
    // or is the first thread
    if (!Galois::Runtime::inGaloisForEach || !LL::getTID())
      net.handleReceives();
  }
  return reinterpret_cast<T*>(p);
}

template<typename T>
void RemoteDirectory::remoteReqLandingPad(RecvBuffer &buf) {
  uint32_t owner;
  T *data;
  uintptr_t ptr;
  RemoteDirectory& rd = getSystemRemoteDirectory();
#define OBJSTATE (*iter).second
  rd.Lock.lock();
  buf.deserialize(ptr);
  buf.deserialize(owner);
  auto iter = rd.curobj.find(make_pair(ptr,owner));
  // check if the object can be sent
  if ((iter == rd.curobj.end()) || (OBJSTATE.state == RemoteDirectory::objstate::Remote)) {
    // object can't be remote if the owner makes a request
    // abort();
    // object might have been sent after this request was made by owner
  }
  else if (OBJSTATE.state == RemoteDirectory::objstate::Local) {
    bool flag = true;
    data = reinterpret_cast<T*>(OBJSTATE.localobj);
    Lockable *L = reinterpret_cast<Lockable*>(data);
/*
 * ENABLE THIS FOR GRAPH TEST CASE
 * Making sure atleast one iteration runs before returning the obj
    // check if eligible or ineligible to be sent
    if (isMagicLock(L)) {
      // ineligible state - check number of requests
      OBJSTATE.count++;
      if (OBJSTATE.count < INELIGIBLE_COUNT) {
        // the data should not be sent
        flag = false;
      }
    }
 */
    // if eligible and acquire lock so that no iteration begins using the object
    if (flag && rd.diracquire(L)) {
      // object should be sent to the remote host
      SendBuffer sbuf;
      size_t size = sizeof(*data);
      NetworkInterface& net = getSystemNetworkInterface();
      sbuf.serialize(ptr);
      sbuf.serialize(size);
      sbuf.serialize(*data);
      rd.curobj.erase(make_pair(ptr,owner));
      net.sendMessage(owner,&LocalDirectory::localDataLandingPad<T>,sbuf);
      delete data;
    }
  }
  else {
    assert(0 && "Unexpected state in remoteReqLandingPad");
    abort();
  }
  rd.Lock.unlock();
#undef OBJSTATE
  return;
}

template<typename T>
void RemoteDirectory::remoteDataLandingPad(RecvBuffer &buf) {
  uint32_t owner;
  size_t size;
  T *data;
  Lockable *L;
  uintptr_t ptr;
  RemoteDirectory& rd = getSystemRemoteDirectory();
#define OBJSTATE (*iter).second
  rd.Lock.lock();
  buf.deserialize(ptr);
  buf.deserialize(owner);
  auto iter = rd.curobj.find(make_pair(ptr,owner));
  buf.deserialize(size);
  data = new T();
  buf.deserialize((*data));
  OBJSTATE.state = RemoteDirectory::objstate::Local;
  L = reinterpret_cast<Lockable*>(data);
  // lock the object with magic num to mark ineligible
  setMagicLock(L);
  OBJSTATE.localobj = (uintptr_t)data;
  OBJSTATE.count = 0;
  rd.Lock.unlock();
#undef OBJSTATE
  return;
}

// should be blocking outside for each
template<typename T>
T* LocalDirectory::resolve(uintptr_t ptr) {
  uint32_t sent = 0;
  uintptr_t p = haveObject(ptr, sent);
  while (!p) {
    NetworkInterface& net = getSystemNetworkInterface();
    fetchRemoteObj(ptr, sent, &RemoteDirectory::remoteReqLandingPad<T>);
    // abort the iteration if inside for each and dir_blocking not defined
    if (Galois::Runtime::inGaloisForEach && !dir_blocking<T>::value)
      throw remote_ex{ptr, networkHostID};
    p = haveObject(ptr, sent);
    // call handleReceives if only thread outside for_each
    // or is the first thread
    if (!Galois::Runtime::inGaloisForEach || !LL::getTID())
      net.handleReceives();
  }
  return reinterpret_cast<T*>(p);
}

template<typename T>
void LocalDirectory::localReqLandingPad(RecvBuffer &buf) {
  uint32_t remote_to;
  T *data;
  Lockable *L;
  uintptr_t ptr;
#define OBJSTATE (*iter).second
  LocalDirectory& ld = getSystemLocalDirectory();
  ld.Lock.lock();
  buf.deserialize(ptr);
  data = reinterpret_cast<T*>(ptr);
  L = reinterpret_cast<Lockable*>(data);
  auto iter = ld.curobj.find(ptr);
  buf.deserialize(remote_to);
  // add object to list if it's not already there
  if (iter == ld.curobj.end()) {
    LocalDirectory::objstate list_obj;
    list_obj.state = LocalDirectory::objstate::Local;
    ld.curobj[ptr] = list_obj;
    iter = ld.curobj.find(ptr);
  }
  // check if the object can be sent
  if (OBJSTATE.state == LocalDirectory::objstate::Remote) {
    // object is remote so place a return request
    // place the request only if a different node is asking for the object
    if (remote_to != OBJSTATE.sent_to)
      ld.fetchRemoteObj(ptr, OBJSTATE.sent_to, &RemoteDirectory::remoteReqLandingPad<T>);
  }
  else if ((OBJSTATE.state == LocalDirectory::objstate::Local) && ld.diracquire(L)) {
    // object should be sent to the remote host
    // diracquire locks with the LocalDirectory object so that local iterations fail
    SendBuffer sbuf;
    size_t size = sizeof(*data);
    uint32_t host = networkHostID;
    NetworkInterface& net = getSystemNetworkInterface();
    sbuf.serialize(ptr);
    sbuf.serialize(host);
    sbuf.serialize(size);
    sbuf.serialize(*data);
    OBJSTATE.sent_to = remote_to;
    OBJSTATE.state = LocalDirectory::objstate::Remote;
    net.sendMessage(remote_to,&RemoteDirectory::remoteDataLandingPad<T>,sbuf);
  }
  else if (OBJSTATE.state != LocalDirectory::objstate::Local) {
    assert(0 && "Unexpected state in localReqLandingPad");
    abort();
  }
  ld.Lock.unlock();
#undef OBJSTATE
  return;
}

template<typename T>
void LocalDirectory::localDataLandingPad(RecvBuffer &buf) {
  size_t size;
  T *data;
  Lockable *L;
  uintptr_t ptr;
#define OBJSTATE (*iter).second
  LocalDirectory& ld = getSystemLocalDirectory();
  ld.Lock.lock();
  buf.deserialize(ptr);
  data = reinterpret_cast<T*>(ptr);
  auto iter = ld.curobj.find(ptr);
  buf.deserialize(size);
  buf.deserialize(*data);
  L = reinterpret_cast<Lockable*>(data);
  OBJSTATE.state = LocalDirectory::objstate::Local;
  unlock(L);
  ld.Lock.unlock();
#undef OBJSTATE
  return;
}

// should be blocking if not in for each
template<typename T>
T* PersistentDirectory::resolve(uintptr_t ptr, uint32_t owner) {
  assert(ptr);
  assert(owner != networkHostID);
  uintptr_t p = haveObject(ptr, owner);
  while (!p) {
    NetworkInterface& net = getSystemNetworkInterface();
    fetchRemoteObj(ptr, owner, &PersistentDirectory::persistentReqLandingPad<T>);
    // abort the iteration if inside for each and dir_blocking not defined
    if (Galois::Runtime::inGaloisForEach && !dir_blocking<T>::value)
      throw remote_ex{ptr, owner};
    p = haveObject(ptr, owner);
    // call handleReceives if only thread outside for_each
    // or is the first thread
    if (!Galois::Runtime::inGaloisForEach || !LL::getTID())
      net.handleReceives();
  }
  return reinterpret_cast<T*>(p);
}

// everytime there's a request send the object!
// always runs on the host owning the object
template<typename T>
void PersistentDirectory::persistentReqLandingPad(RecvBuffer &buf) {
  T *data;
  uintptr_t ptr;
  SendBuffer sbuf;
  uint32_t remote, owner;
  owner = networkHostID;
  NetworkInterface& net = getSystemNetworkInterface();
  buf.deserialize(ptr);
  buf.deserialize(remote);
  data = reinterpret_cast<T*>(ptr);
  // object should be sent to the remote host
  sbuf.serialize(ptr);
  sbuf.serialize(owner);
  sbuf.serialize(*data);
  net.sendMessage(remote,&PersistentDirectory::persistentDataLandingPad<T>,sbuf);
  return;
}

template<typename T>
void PersistentDirectory::persistentDataLandingPad(RecvBuffer &buf) {
  uint32_t owner;
  T *data;
  uintptr_t ptr;
  PersistentDirectory& pd = getSystemPersistentDirectory();
#define OBJSTATE (*iter).second
  pd.Lock.lock();
  buf.deserialize(ptr);
  buf.deserialize(owner);
  auto iter = pd.perobj.find(make_pair(ptr,owner));
  data = new T();
  buf.deserialize((*data));
  OBJSTATE.localobj = (uintptr_t)data;
  pd.Lock.unlock();
#undef OBJSTATE
  return;
}

#endif
