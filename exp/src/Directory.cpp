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

#include "Galois/Runtime/Directory.h"

using namespace std;
using namespace Galois::Runtime::Distributed;

uintptr_t RemoteDirectory::haveObject(uintptr_t ptr, uint32_t owner, SimpleRuntimeContext *cnx) {
#define OBJSTATE (*iter).second
  lock_guard<glock> lock(Lock);
  auto iter = curobj.find(make_pair(ptr,owner));
  uintptr_t retval = 0;
  // add object to list if it's not already there
  if (iter == curobj.end()) {
    objstate list_obj;
    list_obj.count = 0;
    list_obj.localobj = 0;
    list_obj.state = objstate::Remote;
    curobj[make_pair(ptr,owner)] = list_obj;
    iter = curobj.find(make_pair(ptr,owner));
  }
  // Returning the object even if locked as the call to acquire would fail
  if (OBJSTATE.state != objstate::Remote)
    retval = OBJSTATE.localobj;
  // acquire the lock if inside for_each
  if (retval && cnx && inGaloisForEach) {
    Lockable *L = reinterpret_cast<Lockable*>(retval);
    cnx->acquire(L);
  }
#undef OBJSTATE
  return retval;
}

uintptr_t RemoteDirectory::transientHaveObj(uintptr_t ptr, uint32_t owner, bool& isavail) {
#define OBJSTATE (*iter).second
  lock_guard<glock> lock(Lock);
  auto iter = curobj.find(make_pair(ptr,owner));
  uintptr_t retval = 0;
  isavail = false;
  // add object to list if it's not already there
  if (iter == curobj.end()) {
    objstate list_obj;
    list_obj.count = 0;
    list_obj.localobj = 0;
    list_obj.state = objstate::Remote;
    curobj[make_pair(ptr,owner)] = list_obj;
    iter = curobj.find(make_pair(ptr,owner));
  }
  // Returning the object even if locked as the call to acquire would fail
  if (OBJSTATE.state != objstate::Remote) {
    retval = OBJSTATE.localobj;
    isavail = true;
  }
  // acquire the lock if inside for_each
  if (retval) {
    Lockable *L = reinterpret_cast<Lockable*>(retval);
    if (!dirAcquire(L, true))
      retval = 0;
  }
#undef OBJSTATE
  return retval;
}

// NOTE: make sure that handleReceives() is called by another thread
void RemoteDirectory::fetchRemoteObj(uintptr_t ptr, uint32_t owner, recvFuncTy pad) {
  SendBuffer buf;
  uint32_t host = networkHostID;
  NetworkInterface& net = getSystemNetworkInterface();
  gSerialize(buf,ptr,host);
  net.sendMessage (owner, pad, buf);
  return;
}

uintptr_t LocalDirectory::haveObject(uintptr_t ptr, uint32_t &remote, SimpleRuntimeContext *cnx) {
#define OBJSTATE (*iter).second
  lock_guard<glock> lock(Lock);
  auto iter = curobj.find(ptr);
  uintptr_t retval = 0;
  // Returning the object even if locked as the call to acquire would fail
  // return the object even if it is not in the list
  if ((iter == curobj.end()) || (OBJSTATE.state == objstate::Local))
    retval = ptr;
  else if ((iter != curobj.end()) && (OBJSTATE.state == objstate::Remote))
    remote = OBJSTATE.sent_to;
  else
    printf ("Unrecognized state in LocalDirectory::haveObject\n");
  // acquire the lock if inside for_each
  if (retval && cnx && inGaloisForEach) {
    Lockable *L = reinterpret_cast<Lockable*>(retval);
    cnx->acquire(L);
  }
#undef OBJSTATE
  return retval;
}

uintptr_t LocalDirectory::transientHaveObj(uintptr_t ptr, uint32_t &remote, bool& isavail) {
#define OBJSTATE (*iter).second
  lock_guard<glock> lock(Lock);
  auto iter = curobj.find(ptr);
  uintptr_t retval = 0;
  isavail = false;
  // Returning the object even if locked as the call to acquire would fail
  // return the object even if it is not in the list
  if ((iter == curobj.end()) || (OBJSTATE.state == objstate::Local)) {
    retval = ptr;
    isavail = true;
  }
  else if ((iter != curobj.end()) && (OBJSTATE.state == objstate::Remote))
    remote = OBJSTATE.sent_to;
  else
    printf ("Unrecognized state in LocalDirectory::haveObject\n");
  // acquire the lock if inside for_each
  if (retval) {
    Lockable *L = reinterpret_cast<Lockable*>(retval);
    // fails if local but locked another thread
    if(!dirAcquire(L))
      retval = 0;
  }
#undef OBJSTATE
  return retval;
}

// NOTE: make sure that the handleReceives() is called by another thread
void LocalDirectory::fetchRemoteObj(uintptr_t ptr, uint32_t remote, recvFuncTy pad) {
  SendBuffer buf;
  uint32_t host = networkHostID;
  NetworkInterface& net = getSystemNetworkInterface();
  gSerialize(buf,ptr,host);
  net.sendMessage (remote, pad, buf);
  return;
}

uintptr_t PersistentDirectory::haveObject(uintptr_t ptr, uint32_t owner) {
#define OBJSTATE (*iter).second
  lock_guard<glock> lock(Lock);
  auto iter = perobj.find(make_pair(ptr,owner));
  uintptr_t retval = 0;
  // add object to list if it's not already there
  if (iter == perobj.end()) {
    objstate list_obj;
    list_obj.localobj = 0;
    list_obj.requested = false;
    perobj[make_pair(ptr,owner)] = list_obj;
    iter = perobj.find(make_pair(ptr,owner));
  }
  retval = OBJSTATE.localobj;
#undef OBJSTATE
  return retval;
}

// NOTE: make sure that handleReceives() is called by another thread
void PersistentDirectory::fetchRemoteObj(uintptr_t ptr, uint32_t owner, recvFuncTy pad) {
  SendBuffer buf;
  uint32_t host = networkHostID;
#define OBJSTATE (*iter).second
  lock_guard<glock> lock(Lock);
  auto iter = perobj.find(make_pair(ptr,owner));
  // return if already requested
  if (OBJSTATE.requested) {
    return;
  }
  OBJSTATE.requested = true;
#undef OBJSTATE
  // store that the object has been requested
  NetworkInterface& net = getSystemNetworkInterface();
  gSerialize(buf,ptr,host);
  net.sendMessage (owner, pad, buf);
  return;
}

bool LocalDirectory::dirAcquire(Galois::Runtime::Lockable* L) {
  if (L->Owner.try_lock()) {
    assert(!L->Owner.getValue());
    assert(!L->next);
    L->Owner.setValue(this);
    return true;
  }
  else if (L->Owner.stealing_CAS(USE_LOCK,this)) {
    assert(!L->next);
    return true;
  }
  return false;
}

void LocalDirectory::dirRelease(uintptr_t ptr) {
  lock_guard<glock> lock(Lock);
  auto iter = curobj.find(ptr);
  // assert if the object is Remote
  assert((iter == curobj.end()) || ((*iter).second.state == objstate::Local));
  Lockable *L = reinterpret_cast<Lockable*>(ptr);
  L->Owner.unlock_and_clear();
}

//release the lock got using transientAcquire
void LocalDirectory::transientRelease(uintptr_t ptr) {
  dirRelease(ptr);
}

bool RemoteDirectory::dirAcquire(Galois::Runtime::Lockable* L, bool steal) {
  if (L->Owner.try_lock()) {
    assert(!L->Owner.getValue());
    assert(!L->next);
    L->Owner.setValue(this);
    return true;
  }
 /* 
  * Letting atleast one iteration proceed before sending the data
  * ENABLE THIS WHEN RUNNING AN ACTUAL TEST CASE
  * NOTE: steal should be enabled for transientAcquire() calls!
  */
  else if (steal && L->Owner.stealing_CAS(USE_LOCK,this)) {
    assert(!L->next);
    return true;
  }
  return false;
}

void RemoteDirectory::dirRelease(uintptr_t ptr, uint32_t owner) {
  lock_guard<glock> lock(Lock);
  auto iter = curobj.find(make_pair(ptr,owner));
  assert(iter != curobj.end());
  assert((*iter).second.state == objstate::Local);
  uintptr_t retval = (*iter).second.localobj;
  Lockable *L = reinterpret_cast<Lockable*>(retval);
  L->Owner.unlock_and_clear();
}

//release the lock got using transientAcquire
void RemoteDirectory::transientRelease(uintptr_t ptr, uint32_t owner) {
  dirRelease(ptr, owner);
}

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

