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

#define INELIGIBLE_COUNT 12

using namespace std;
using namespace Galois::Runtime::Distributed;

template<typename T>
void remoteReqLandingPad(RecvBuffer &buf) {
  int owner;
  T *data;
  uintptr_t ptr;
  LocalDirectory& ld = getSystemLocalDirectory();
  RemoteDirectory& rd = getSystemRemoteDirectory();
#define OBJSTATE (*iter).second
  rd.Lock.lock();
  buf.deserialize(ptr);
  buf.deserialize(owner);
  auto iter = rd.curobj.find(make_pair(ptr,owner));
  // check if the object can be sent
  if ((iter == rd.curobj.end()) || (OBJSTATE.state == RemoteDirectory::objstate::Remote)) {
    // object can't be remote if the owner makes a request
    abort();
  }
  else if (OBJSTATE.state == RemoteDirectory::objstate::Local) {
// modify the check here to use acquire and free instead of Eligible and Ineligible states
    // check number of requests and make eligible if needed
    OBJSTATE.count++;
    if (OBJSTATE.count == INELIGIBLE_COUNT) {
      // the data should be sent
//    OBJSTATE.state = RemoteDirectory::objstate::Eligible;
    }
//  if (OBJSTATE.state == RemoteDirectory::objstate::Eligible) {
      // object should be sent to the remote host
      data = reinterpret_cast<T*>(OBJSTATE.localobj);
      SendBuffer sbuf;
      NetworkInterface& net = getSystemNetworkInterface();
      sbuf.serialize(ptr);
      sbuf.serialize(sizeof(*data));
      sbuf.serialize(*data);
      rd.curobj.erase(make_pair(ptr,owner));
      net.sendMessage(owner,&localDataLandingPad<T>,sbuf);
      free(data);
//  }
  }
  else {
    cout << "Unexpected state in remoteReqLandingPad" << endl;
  }
  rd.Lock.unlock();
#undef OBJSTATE
  return;
}

template<typename T>
void remoteDataLandingPad(RecvBuffer &buf) {
  uint32_t owner;
  int size;
  T *data;
  uintptr_t ptr;
  LocalDirectory& ld = getSystemLocalDirectory();
  RemoteDirectory& rd = getSystemRemoteDirectory();
#define OBJSTATE (*iter).second
  rd.Lock.lock();
  buf.deserialize(ptr);
  buf.deserialize(owner);
  auto iter = rd.curobj.find(make_pair(ptr,owner));
  buf.deserialize(size);
  data = (T*)calloc(1,size);
  buf.deserialize((*data));
  OBJSTATE.state = RemoteDirectory::objstate::Local;
// lock the object with the special value to mark eligible
  OBJSTATE.localobj = (uintptr_t)data;
  OBJSTATE.count = 0;
  rd.Lock.unlock();
#undef OBJSTATE
  return;
}

uintptr_t RemoteDirectory::haveObject(uintptr_t ptr, uint32_t owner) {
#define OBJSTATE (*iter).second
  Lock.lock();
  auto iter = curobj.find(make_pair(ptr,owner));
  uintptr_t retval = 0;
  // add object to list if it's not already there
  if (iter == curobj.end()) {
    objstate list_obj;
    list_obj.count = 0;
    OBJSTATE.localobj = 0;
    list_obj.state = objstate::Remote;
    curobj[make_pair(ptr,owner)] = list_obj;
  }
  // Returning the object even if locked as the call to acquire would fail
  if (OBJSTATE.state != objstate::Remote)
    retval = OBJSTATE.localobj;
  Lock.unlock();
#undef OBJSTATE
  return retval;
}

// NOTE: make sure that handleReceives() is called by another thread
void RemoteDirectory::fetchRemoteObj(uintptr_t ptr, uint32_t owner, recvFuncTy pad) {
  SendBuffer buf;
  NetworkInterface& net = getSystemNetworkInterface();
  buf.serialize(ptr);
  buf.serialize(networkHostID);
  net.sendMessage (owner, pad, buf);
  return;
}

template<typename T>
void localDataLandingPad(RecvBuffer &buf) {
  int size;
  T *data;
  uintptr_t ptr;
#define OBJSTATE (*iter).second
  LocalDirectory& ld = getSystemLocalDirectory();
  RemoteDirectory& rd = getSystemRemoteDirectory();
  ld.Lock.lock();
  buf.deserialize(ptr);
  data = reinterpret_cast<T*>(ptr);
  auto iter = ld.curobj.find(ptr);
  buf.deserialize(size);
  buf.deserialize((*data));
  OBJSTATE.state = LocalDirectory::objstate::Local;
  ld.Lock.unlock();
#undef OBJSTATE
  return;
}

uintptr_t LocalDirectory::haveObject(uintptr_t ptr, int *remote) {
#define OBJSTATE (*iter).second
  Lock.lock();
  auto iter = curobj.find(ptr);
  uintptr_t retval = 0;
  // Returning the object even if locked as the call to acquire would fail
  // return the object even if it is not in the list
  if (iter == curobj.end() || OBJSTATE.state != objstate::Remote)
    retval = ptr;
  else if (iter != curobj.end() && OBJSTATE.state == objstate::Remote)
    *remote = OBJSTATE.sent_to;
  Lock.unlock();
#undef OBJSTATE
  return retval;
}

// NOTE: make sure that the handleReceives() is called by another thread
void LocalDirectory::fetchRemoteObj(uintptr_t ptr, uint32_t remote, recvFuncTy pad) {
  SendBuffer buf;
  NetworkInterface& net = getSystemNetworkInterface();
  buf.serialize(ptr);
  buf.serialize(networkHostID);
  net.sendMessage (remote, pad, buf);
  return;
}

template<typename T>
T* LocalDirectory::resolve(uintptr_t ptr) {
  int sent = 0;
  RemoteDirectory& rd = getSystemRemoteDirectory();
  uintptr_t p = haveObject(ptr, &sent);
  if (!p)
    fetchRemoteObj(ptr, sent, &remoteReqLandingPad<T>);
  return reinterpret_cast<T*>(p);
}

RemoteDirectory& Galois::Runtime::Distributed::getSystemRemoteDirectory() {
  static RemoteDirectory obj;
  return obj;
}

LocalDirectory& Galois::Runtime::Distributed::getSystemLocalDirectory() {
  static LocalDirectory obj;
  return obj;
}

