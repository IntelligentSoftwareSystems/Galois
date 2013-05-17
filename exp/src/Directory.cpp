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

#include "Galois/Runtime/DistSupport.h"
#include "Galois/Runtime/Directory.h"
#include "Galois/Runtime/ll/TID.h"

#include <mutex>

using namespace Galois::Runtime;

SimpleRuntimeContext& Galois::Runtime::getTransCnx() {
  static PerThreadStorage<Galois::Runtime::SimpleRuntimeContext> obj;
  return *obj.getLocal();
}

SimpleRuntimeContext& Galois::Runtime::getAbortCnx() {
  static Galois::Runtime::SimpleRuntimeContext obj;
  return obj;
}

std::pair<Lockable*, boost::intrusive_ptr<remoteObj>> Directory::getObjUntyped(fatPointer ptr) {
  //Get the object
  std::pair<Lockable*, boost::intrusive_ptr<remoteObj> > retval;
  if (ptr.first == networkHostID) {
    retval.first = ptr.second;
  } else {
    retval.second = remoteWeakResolve(ptr);
    if (retval.second)
      retval.first = retval.second->getObj();
  }
  return retval;
}


void Directory::notifyWhenAvailable(fatPointer ptr, std::function<void(fatPointer)> func) {
  Lock.lock();
  notifiers.insert(std::make_pair(ptr,func));
  Lock.unlock();

  //check for existance
  //Get the object
  std::pair<Lockable*, boost::intrusive_ptr<remoteObj> > obj = getObjUntyped(ptr);
  if (obj.first && !isAcquiredBy(obj.first, this)) {
    assert(!getPending(obj.first));
    notify(ptr);
  }
}

void Directory::notify(fatPointer ptr) {
  std::vector<std::function<void(fatPointer)>> V;
  Lock.lock();
  std::for_each(notifiers.lower_bound(ptr), notifiers.upper_bound(ptr),
                [&V] (decltype(*notifiers.begin())& val) { V.push_back(val.second); });
  notifiers.erase(notifiers.lower_bound(ptr), notifiers.upper_bound(ptr));
  Lock.unlock();
  std::for_each(V.begin(), V.end(), [ptr] (std::function<void(fatPointer)>& f) { f(ptr); });
}

void Directory::tryNotifyAll() {
  
}

void Directory::sendRequest(fatPointer ptr, uint32_t msgTo, recvFuncTy f, uint32_t objTo) {
  SendBuffer sbuf;
  gSerialize(sbuf, ptr, objTo);
  getSystemNetworkInterface().send(msgTo, f, sbuf);
}

void Directory::fetchObj(Lockable* obj, fatPointer ptr, uint32_t msgTo, recvFuncTy f, uint32_t objTo) {
  if (!getPending(obj)) {
    setPending(obj);
    sendRequest(ptr, msgTo, f, objTo);
  }
}

void Directory::setContended(fatPointer ptr) {
  //check for existance
  //Get the object
  std::pair<Lockable*, boost::intrusive_ptr<remoteObj> > obj = getObjUntyped(ptr);
  if (obj.first)
    setPriority(obj.first);
}

void Directory::clearContended(fatPointer ptr) {
  //check for existance
  //Get the object
  std::pair<Lockable*, boost::intrusive_ptr<remoteObj> > obj = getObjUntyped(ptr);
  if (obj.first) {
    delPriority(obj.first);
    notify(ptr);
  }
}

////////////////////////////////////////////////////////////////////////////////

Directory& Galois::Runtime::getSystemDirectory() {
  static Directory obj;
  return obj;
}
