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

#define DATA_MSG ((int)0x123)
#define REQ_MSG  ((int)0x456)

#define INELIGIBLE_COUNT 12

using namespace std;
using namespace Galois::Runtime::Distributed;

// NACK is a noop here too (NOT SENDING NACK)
// if Ineligible, transfer to Eligible after INELI2ELI_COUNT requests
// if Ineligible or Locked send a NACK
// if Eligible return the object back to owner and mark as Remote
template<typename T>
void RemoteDirectory::remoteLandingPad(RecvBuffer &buf) {
  int msg_type, owner, size;
  T *data;
  uintptr_t ptr;
#define OBJSTATE (*iter).second
  Lock.lock();
  LocalDirectory& ld = getSystemLocalDirectory();
  buf.deserialize(msg_type);
  buf.deserialize(ptr);
  buf.deserialize(owner);
  auto iter = curobj.find(make_pair(ptr,owner));
  if (msg_type == REQ_MSG) {
    // check if the object can be sent
    if ((iter == curobj.end()) || (OBJSTATE.state == objstate::Remote)) {
      // object can't be remote if the owner makes a request
      abort();
    }
    if (OBJSTATE.state == objstate::Ineligible) {
      // check number of requests and make eligible if needed
      OBJSTATE.count++;
      if (OBJSTATE.count == INELIGIBLE_COUNT) {
        // the data should be sent
        OBJSTATE.state = objstate::Eligible;
      }
    }
    if (OBJSTATE.state == objstate::Eligible) {
      // object should be sent to the remote host
      data = reinterpret_cast<T*>(OBJSTATE.localobj);
      SendBuffer sbuf;
      NetworkInterface& net = getSystemNetworkInterface();
      sbuf.serialize(DATA_MSG);
      sbuf.serialize(ptr);
      sbuf.serialize(sizeof(*data));
      sbuf.serialize(*data);
      curobj.erase(make_pair(ptr,owner));
      net.sendMessage(owner,&ld.ownerLandingPad<T>,sbuf);
      free(data);
    }
  }
  else if (msg_type == DATA_MSG) {
    buf.deserialize(size);
    data = (T*)calloc(1,size);
    buf.deserialize((*data));
    OBJSTATE.state = objstate::Ineligible;
    OBJSTATE.localobj = (uintptr_t)data;
    OBJSTATE.count = 0;
  }
  Lock.unlock();
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
  buf.serialize((int)REQ_MSG);
  buf.serialize(ptr);
  buf.serialize(networkHostID);
  net.sendMessage (owner, pad, buf);
  return;
}

//resolve a pointer, owner pair
//precondition: owner != networkHostID
template<typename T>
T* RemoteDirectory::resolve(uintptr_t ptr, uint32_t owner) {
  assert(owner != networkHostID);
  LocalDirectory& ld = getSystemLocalDirectory();
  uintptr_t p = haveObject(ptr, owner);
  if (!p)
    fetchRemoteObj(ptr, owner, &ld.ownerLandingPad<T>);
  return reinterpret_cast<T*>(p);
}

// NACK is a noop on the owner (NOT SENDING NACK)
// fwd the request if state is remote
// send a NACK if locked
// send the object if local and mark obj as remote
template<typename T>
void LocalDirectory::ownerLandingPad(RecvBuffer &buf) {
  int msg_type, remote_to, size;
  T *data;
  uintptr_t ptr;
#define OBJSTATE (*iter).second
  Lock.lock();
  RemoteDirectory& rd = getSystemRemoteDirectory();
  buf.deserialize(msg_type);
  buf.deserialize(ptr);
  data = reinterpret_cast<T*>(ptr);
  auto iter = curobj.find(ptr);
  if (msg_type == REQ_MSG) {
    buf.deserialize(remote_to);
    // add object to list if it's not already there
    if (iter == curobj.end()) {
      objstate list_obj;
      list_obj.sent_to = remote_to;
      list_obj.state = objstate::Local;
      curobj[ptr] = list_obj;
    }
    // check if the object can be sent
    if (OBJSTATE.state == objstate::Remote) {
      // object is remote so place a return request
      fetchRemoteObj(ptr, OBJSTATE.sent_to, &rd.remoteLandingPad<T>);
    }
    else if (OBJSTATE.state == objstate::Local) {
      // object should be sent to the remote host
      SendBuffer sbuf;
      NetworkInterface& net = getSystemNetworkInterface();
      sbuf.serialize(DATA_MSG);
      sbuf.serialize(ptr);
      sbuf.serialize(networkHostID);
      sbuf.serialize(sizeof(*data));
      sbuf.serialize(*data);
      OBJSTATE.state = objstate::Remote;
      net.sendMessage(remote_to,&rd.remoteLandingPad<T>,sbuf);
    }
  }
  else if (msg_type == DATA_MSG) {
    buf.deserialize(size);
    buf.deserialize((*data));
    OBJSTATE.state = objstate::Local;
  }
  Lock.unlock();
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
  buf.serialize((int)REQ_MSG);
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
    fetchRemoteObj(ptr, sent, &rd.remoteLandingPad<T>);
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

