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

#include <mutex>

#define INELIGIBLE_COUNT 12
using namespace std;

namespace Galois {
namespace Runtime {
namespace Distributed {

struct remote_ex {
  Distributed::recvFuncTy pad;
  uintptr_t ptr;
  uint32_t owner;
};

template<typename T>
class gptr;

template<typename T>
remote_ex make_remote_ex(const gptr<T>&);

template<typename T>
remote_ex make_remote_ex(uintptr_t ptr, uint32_t owner);

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

  // using RAII for locking, done to handle lock release on exceptions
  typedef Galois::Runtime::LL::SimpleLock<true> glock;
  Galois::Runtime::LL::SimpleLock<true> Lock;
  boost::unordered_map<std::pair<uintptr_t, uint32_t>, objstate, ohash> curobj;

  //returns a valid locked local pointer to the object if it exists
  //or returns null
  uintptr_t haveObject(uintptr_t ptr, uint32_t owner, SimpleRuntimeContext *cnx);

  // isavail returns if the object is available in this host
  uintptr_t transientHaveObj(uintptr_t ptr, uint32_t owner, bool& isavail);

  // tries to acquire a lock and returns true or false if acquired
  // used before sending an object and freeing it
  // use steal to specify if the lock can be acquired as soon as received
  bool dirAcquire(Lockable* L, bool steal);

  // releases an object acquired with dirAcquire
  void dirRelease(uintptr_t ptr, uint32_t owner);

public:

  // places a remote request for the node
  void fetchRemoteObj(uintptr_t ptr, uint32_t owner, recvFuncTy pad);

  // Handles incoming requests for remote objects
  // if Ineligible, transfer to Eligible after INELI2ELI_COUNT requests
  // if Eligible return the object back to owner and mark as Remote
  template<typename T>
  static void remoteReqLandingPad(RecvBuffer &);

  // Landing Pad for incoming remote objects
  template<typename T>
  static void remoteDataLandingPad(RecvBuffer &);

  //prefetch a pointer, owner pair
  //precondition: owner != networkHostID
  //don't lock obj, just return if found and place a request if not found
  template<typename T>
  void prefetch(uintptr_t ptr, uint32_t owner);

  //resolve a pointer, owner pair
  //precondition: owner != networkHostID
  template<typename T>
  T* resolve(uintptr_t ptr, uint32_t owner, SimpleRuntimeContext *cnx);

  //resolve a pointer, owner pair
  //blocking acquire to lock object with the directory object
  //precondition: owner != networkHostID
  template<typename T>
  T* transientAcquire(uintptr_t ptr, uint32_t owner);

  //release the lock got using transientAcquire
  void transientRelease(uintptr_t ptr, uint32_t owner);
};

class LocalDirectory: public SimpleRuntimeContext {

  struct objstate {
    // Remote - Object passed to a remote host
    // Local - Local object may be locked
    enum ObjStates { Remote, Local };

    uint32_t sent_to;  // valid only for remote objects
    enum ObjStates state;
  };

  // using RAII for locking, done to handle lock release on exceptions
  typedef Galois::Runtime::LL::SimpleLock<true> glock;
  Galois::Runtime::LL::SimpleLock<true> Lock;
  boost::unordered_map<uintptr_t, objstate> curobj;

  // returns a valid locked local pointer to the object if not remote
  uintptr_t haveObject(uintptr_t ptr, uint32_t &remote, SimpleRuntimeContext *cnx);

  // places a remote request for the node
  void fetchRemoteObj(uintptr_t ptr, uint32_t remote, recvFuncTy pad);

  // blocking acquire to lock object with the directory object
  // isavail returns if the object is available on this host
  uintptr_t transientHaveObj(uintptr_t ptr, uint32_t& remote, bool& isavail);

  // tries to acquire a lock and returns true or false if acquired
  // used before sending an object and marking it remote
  bool dirAcquire(Lockable* L);

  // releases an object acquired with dirAcquire
  void dirRelease(uintptr_t ptr);

public:

  LocalDirectory(): SimpleRuntimeContext(true) {}

  // forward the request if the state is remote
  // send the object if local and not locked, also mark as remote
  template<typename T>
  static void localReqLandingPad(RecvBuffer &);

  // send the object if local, not locked and mark obj as remote
  template<typename T>
  static void localDataLandingPad(RecvBuffer &);

  //prefetch a pointer
  //don't lock obj, just return if found and place a request if not found
  template<typename T>
  void prefetch(uintptr_t ptr);

  // resolve a pointer
  template<typename T>
  T* resolve(uintptr_t ptr, SimpleRuntimeContext *cnx);

  //resolve a pointer
  //blocking acquire to lock object with the directory object
  template<typename T>
  T* transientAcquire(uintptr_t ptr);

  //release the lock got using transientAcquire
  void transientRelease(uintptr_t ptr);
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

  // using RAII for locking, done to handle lock release on exceptions
  typedef Galois::Runtime::LL::SimpleLock<true> glock;
  Galois::Runtime::LL::SimpleLock<true> Lock;
  boost::unordered_map<std::pair<uintptr_t, uint32_t>, objstate, ohash> perobj;

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

// should never block, just place a request if not found
template<typename T>
void RemoteDirectory::prefetch(uintptr_t ptr, uint32_t owner) {
  assert(ptr);
  assert(owner != networkHostID);
  // don't lock the object if not found
  uintptr_t p = haveObject(ptr, owner, NULL);
  // don't block just place a request if not local
  if (!p) {
    NetworkInterface& net = getSystemNetworkInterface();
    fetchRemoteObj(ptr, owner, &LocalDirectory::localReqLandingPad<T>);
    // call handleReceives if only thread outside for_each
    // or is the first thread
    if (!Galois::Runtime::inGaloisForEach || !LL::getTID())
      net.handleReceives();
  }
}

// should always be a blocking call
template<typename T>
T* RemoteDirectory::transientAcquire(uintptr_t ptr, uint32_t owner) {
  assert(ptr);
  assert(owner != networkHostID);
  bool isLocallyAvail;
  uintptr_t p = transientHaveObj(ptr, owner, isLocallyAvail);
  while (!p) {
    NetworkInterface& net = getSystemNetworkInterface();
    // make fetch object request only if remote
    if (!isLocallyAvail)
      fetchRemoteObj(ptr, owner, &LocalDirectory::localReqLandingPad<T>);
    // call handleReceives if only thread outside for_each
    // or is the first thread
    if (!Galois::Runtime::inGaloisForEach || !LL::getTID())
      net.handleReceives();
    p = transientHaveObj(ptr, owner, isLocallyAvail);
  }
  return reinterpret_cast<T*>(p);
}

// should be blocking if not in for each
template<typename T>
T* RemoteDirectory::resolve(uintptr_t ptr, uint32_t owner, SimpleRuntimeContext *cnx) {
  assert(ptr);
  assert(owner != networkHostID);
  uintptr_t p = haveObject(ptr, owner, cnx);
  while (!p) {
    NetworkInterface& net = getSystemNetworkInterface();
    fetchRemoteObj(ptr, owner, &LocalDirectory::localReqLandingPad<T>);
    // abort the iteration if inside for each and dir_blocking not defined
    if (Galois::Runtime::inGaloisForEach && !dir_blocking<T>::value)
      throw make_remote_ex<T>(ptr, owner);
    // call handleReceives if only thread outside for_each
    // or is the first thread
    if (!Galois::Runtime::inGaloisForEach || !LL::getTID())
      net.handleReceives();
    p = haveObject(ptr, owner, cnx);
  }
  return reinterpret_cast<T*>(p);
}

template<typename T>
void RemoteDirectory::remoteReqLandingPad(RecvBuffer &buf) {
  uint32_t owner;
  T *data;
  uintptr_t ptr;
  RemoteDirectory& rd = getSystemRemoteDirectory();
  lock_guard<glock> lock(rd.Lock);
  gDeserialize(buf,ptr, owner);
  auto iter = rd.curobj.find(make_pair(ptr,owner));
  // check if the object can be sent
  if ((iter == rd.curobj.end()) || (iter->second.state == RemoteDirectory::objstate::Remote)) {
    // object can't be remote if the owner makes a request
    // abort();
    // object might have been sent after this request was made by owner
  }
  else if (iter->second.state == RemoteDirectory::objstate::Local) {
    bool flag = true;
    data = reinterpret_cast<T*>(iter->second.localobj);
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
    // disable stealing so that atleast one iteration succeeds
    if (flag && rd.dirAcquire(L,true)) {
      // object should be sent to the remote host
      SendBuffer sbuf;
      size_t size = sizeof(*data);
      NetworkInterface& net = getSystemNetworkInterface();
      gSerialize(sbuf, ptr, size, *data);
      rd.curobj.erase(make_pair(ptr,owner));
      net.sendMessage(owner,&LocalDirectory::localDataLandingPad<T>,sbuf);
      delete data;
    }
  }
  else {
    assert(0 && "Unexpected state in remoteReqLandingPad");
    abort();
  }
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
  lock_guard<glock> lock(rd.Lock);
  gDeserialize(buf,ptr,owner,size);
  auto iter = rd.curobj.find(make_pair(ptr,owner));
  data = new T();
  gDeserialize(buf,*data);
  iter->second.state = RemoteDirectory::objstate::Local;
  L = reinterpret_cast<Lockable*>(data);
  // lock the object with magic num to mark ineligible
  setMagicLock(L);
  iter->second.localobj = (uintptr_t)data;
  iter->second.count = 0;
  return;
}

// should never block, just place a request if not found
template<typename T>
void LocalDirectory::prefetch(uintptr_t ptr) {
  uint32_t sent = 0;
  assert(ptr);
  // don't lock the object if not found
  uintptr_t p = haveObject(ptr, sent, NULL);
  // don't block just place a request if not local
  if (!p) {
    NetworkInterface& net = getSystemNetworkInterface();
    fetchRemoteObj(ptr, sent, &RemoteDirectory::remoteReqLandingPad<T>);
    // call handleReceives if only thread outside for_each
    // or is the first thread
    if (!Galois::Runtime::inGaloisForEach || !LL::getTID())
      net.handleReceives();
  }
}

// should always be blocking
template<typename T>
T* LocalDirectory::transientAcquire(uintptr_t ptr) {
  uint32_t sent = 0;
  bool isLocallyAvail;
  uintptr_t p = transientHaveObj(ptr, sent, isLocallyAvail);
  while (!p) {
    NetworkInterface& net = getSystemNetworkInterface();
    // fetch only if remote
    if (!isLocallyAvail)
      fetchRemoteObj(ptr, sent, &RemoteDirectory::remoteReqLandingPad<T>);
    // call handleReceives if only thread outside for_each
    // or is the first thread
    if (!Galois::Runtime::inGaloisForEach || !LL::getTID())
      net.handleReceives();
    p = transientHaveObj(ptr, sent, isLocallyAvail);
  }
  return reinterpret_cast<T*>(p);
}

// should be blocking outside for each
template<typename T>
T* LocalDirectory::resolve(uintptr_t ptr, SimpleRuntimeContext *cnx) {
  uint32_t sent = 0;
  uintptr_t p = haveObject(ptr, sent, cnx);
  while (!p) {
    NetworkInterface& net = getSystemNetworkInterface();
    fetchRemoteObj(ptr, sent, &RemoteDirectory::remoteReqLandingPad<T>);
    // abort the iteration if inside for each and dir_blocking not defined
    if (Galois::Runtime::inGaloisForEach && !dir_blocking<T>::value)
      throw make_remote_ex<T>(ptr, networkHostID);
    // call handleReceives if only thread outside for_each
    // or is the first thread
    if (!Galois::Runtime::inGaloisForEach || !LL::getTID())
      net.handleReceives();
    p = haveObject(ptr, sent, cnx);
  }
  return reinterpret_cast<T*>(p);
}

template<typename T>
void LocalDirectory::localReqLandingPad(RecvBuffer &buf) {
  uint32_t remote_to;
  T *data;
  Lockable *L;
  uintptr_t ptr;
  LocalDirectory& ld = getSystemLocalDirectory();
  lock_guard<glock> lock(ld.Lock);
  gDeserialize(buf,ptr,remote_to);
  data = reinterpret_cast<T*>(ptr);
  L = reinterpret_cast<Lockable*>(data);
  auto iter = ld.curobj.find(ptr);
  // add object to list if it's not already there
  if (iter == ld.curobj.end()) {
    LocalDirectory::objstate list_obj;
    list_obj.state = LocalDirectory::objstate::Local;
    ld.curobj[ptr] = list_obj;
    iter = ld.curobj.find(ptr);
  }
  // check if the object can be sent
  if (iter->second.state == LocalDirectory::objstate::Remote) {
    // object is remote so place a return request
    // place the request only if a different node is asking for the object
    if (remote_to != iter->second.sent_to)
      ld.fetchRemoteObj(ptr, iter->second.sent_to, &RemoteDirectory::remoteReqLandingPad<T>);
  }
  else if ((iter->second.state == LocalDirectory::objstate::Local) && ld.dirAcquire(L)) {
    // object should be sent to the remote host
    // dirAcquire locks with the LocalDirectory object so that local iterations fail
    SendBuffer sbuf;
    size_t size = sizeof(*data);
    uint32_t host = networkHostID;
    NetworkInterface& net = getSystemNetworkInterface();
    gSerialize(sbuf,ptr, host, size, *data);
    iter->second.sent_to = remote_to;
    iter->second.state = LocalDirectory::objstate::Remote;
    net.sendMessage(remote_to,&RemoteDirectory::remoteDataLandingPad<T>,sbuf);
  }
  else if (iter->second.state != LocalDirectory::objstate::Local) {
    assert(0 && "Unexpected state in localReqLandingPad");
    abort();
  }
  return;
}

template<typename T>
void LocalDirectory::localDataLandingPad(RecvBuffer &buf) {
  size_t size;
  T *data;
  Lockable *L;
  uintptr_t ptr;
  LocalDirectory& ld = getSystemLocalDirectory();
  lock_guard<glock> lock(ld.Lock);
  gDeserialize(buf,ptr,size);
  data = reinterpret_cast<T*>(ptr);
  auto iter = ld.curobj.find(ptr);
  gDeserialize(buf,*data);
  L = reinterpret_cast<Lockable*>(data);
  iter->second.state = LocalDirectory::objstate::Local;
  unlock(L);
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
      throw make_remote_ex<T>(ptr, owner);
    // call handleReceives if only thread outside for_each
    // or is the first thread
    if (!Galois::Runtime::inGaloisForEach || !LL::getTID())
      net.handleReceives();
    p = haveObject(ptr, owner);
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
  gDeserialize(buf, ptr, remote);
  data = reinterpret_cast<T*>(ptr);
  // object should be sent to the remote host
  gSerialize(sbuf,ptr, owner, *data);
  net.sendMessage(remote,&PersistentDirectory::persistentDataLandingPad<T>,sbuf);
  return;
}

template<typename T>
void PersistentDirectory::persistentDataLandingPad(RecvBuffer &buf) {
  uint32_t owner;
  T *data;
  uintptr_t ptr;
  PersistentDirectory& pd = getSystemPersistentDirectory();
  lock_guard<glock> lock(pd.Lock);
  data = new T();
  gDeserialize(buf, ptr, owner,*data);
  auto iter = pd.perobj.find(make_pair(ptr,owner));
  iter->second.localobj = (uintptr_t)data;
  return;
}

#endif
