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

using namespace Galois::Runtime::Distributed;

uintptr_t RemoteDirectory::haveObject(uintptr_t ptr, uint32_t owner) {
#define OBJSTATE (*iter).second
  Lock.lock();
  auto iter = curobj.find(std::make_pair(ptr,owner));
  uintptr_t retval = 0;
  // Returning the object even if locked as the call to acquire would fail
  // May have to account for the Requested state if it is introduced
  if (iter != curobj.end() && OBJSTATE.state != objstate::Remote)
    retval = OBJSTATE.localobj;
  Lock.unlock();
#undef OBJSTATE
  return retval;
}

// NOTE: make sure that the handleReceives() is called by another thread
void RemoteDirectory::fetchRemoteObj(uintptr_t ptr, uint32_t owner, recvFuncTy pad) {
  SendBuffer buf;
  NetworkInterface& net = getSystemNetworkInterface();
  buf.serialize(ptr);
  buf.serialize(networkHostID);
  net.sendMessage (owner, pad, buf);
  return;
}

template<typename T>
void ownerLandingPad(RecvBuffer &buf) {
  // NACK is a noop on the owner
  // fwd the request if state is remote
  // send a NACK if locked
  // send the object if local and mark obj as remote
}

uintptr_t LocalDirectory::haveObject(uintptr_t ptr, int *remote) {
#define OBJSTATE (*iter).second
  Lock.lock();
  auto iter = curobj.find(ptr);
  uintptr_t retval = 0;
  // Returning the object even if locked as the call to acquire would fail
  // return the object even if it is not 
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
void remoteLandingPad(RecvBuffer &buf) {
}

RemoteDirectory& Galois::Runtime::Distributed::getSystemRemoteDirectory() {
  static RemoteDirectory obj;
  return obj;
}

LocalDirectory& Galois::Runtime::Distributed::getSystemLocalDirectory() {
  static LocalDirectory obj;
  return obj;
}

