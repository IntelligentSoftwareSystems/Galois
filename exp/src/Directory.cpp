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



void Directory::doRequest(fatPointer ptr, typeHelper* th, uint32_t remoteHost) {
  LL::SLguard lg(getLock(ptr));
  tracking& tr = getTracking(lg, ptr);
  tr.addRequest(remoteHost, th);
  addPending(ptr);
}

bool Directory::doObj(fatPointer ptr, typeHelper* th) {
  LL::SLguard lg(getLock(ptr));
  tracking& tr = getExistingTracking(lg, ptr);
  tr.setLocal();
  addPending(ptr);
  return true;
}

void Directory::setContended(fatPointer ptr) {
  LL::SLguard lg(getLock(ptr));
  tracking& tr = getTracking(lg, ptr);
  tr.setContended();
  addPending(ptr);
}

void Directory::clearContended(fatPointer ptr) {
  LL::SLguard lg(getLock(ptr));
  tracking& tr = getTracking(lg, ptr);
  tr.clearContended();
  addPending(ptr);
}


void Directory::notifyWhenAvailable(fatPointer ptr, std::function<void(fatPointer)> func) {
  LL::SLguard lg(getLock(ptr));
  tracking& tr = getExistingTracking(lg, ptr);
  tr.addNotify(func);
}

void Directory::processObj(LL::SLguard& lg, fatPointer ptr, Lockable* obj) {
  tracking& tr = getTracking(lg, ptr);

  //invalid objects just need requests
  if (isAcquiredBy(obj, this)) {
    if (tr.hasRequest()) {
      uint32_t wanter = tr.getRequest();
      if (!tr.isRecalled() || wanter < tr.getRecalled()) {
        tr.getHelper()->sendRequest(ptr, ptr.first == NetworkInterface::ID ? tr.getCurLoc() : ptr.first, wanter);
        tr.setRecalled(wanter);
      }
    }
    return;
  }

  //only worry about objects with outstanding requests
  if (tr.hasRequest()) {
    uint32_t wanter = tr.getRequest();
    //check if existential lock overwrites transfer
    if (wanter > NetworkInterface::ID && tr.isContended()) {
      LL::gDebug("Contended Protection of [", ptr.first, ",", ptr.second, "] from ", wanter);
      return;
    }
    //Don't need to move object if requestor is local
    if (wanter == NetworkInterface::ID) {
      tr.delRequest(NetworkInterface::ID); //eat local request
      if (tr.hasRequest())
        addPending(ptr);
      return;
    }
    //figure out to whom to send it 
    switch (SimpleRuntimeContext::tryAcquire(obj)) {
    case 0: // Local iteration has the lock
      addPending(ptr);
      return; // delay processing
    case 1: { //now owner (was free and on this host)
      //compute who to send the object too
      //non-owner sends back to owner
      uint32_t dest = ptr.first == NetworkInterface::ID ? wanter : ptr.first;
      tr.getHelper()->sendObject(ptr, obj, dest);
      if (ptr.first == NetworkInterface::ID) {
        //remember who has it
        tr.setCurLoc(dest);
        //we handled this request
        tr.delRequest(dest);
        if (tr.hasRequest())
          addPending(ptr);
      } else {
        //don't need to track pending requests for remote objects
        tr.clearRequest();
      }
      break;
    }
    case 2: //already owner, should have been caught above
    default: //unknown
      abort();
    }
  } else {
    //no outstanding requests? might be able to kill record
    assert(!tr.isRecalled());
    if (!tr.isContended())
      delTracking(lg ,ptr);
    return;
  }
}



void Directory::dump() {
#if 0
  LL::gDebug("Directory has ",
             delayed.size(), " delayed requests, ",
             notifiers.size(), " registered notifcations, ",
             fetches.size(), " pending fetches, ",
             remoteObjects.size(), " remote objects");
  if (fetches.size() < 10) {
    std::set<fatPointer> r = fetches.getAllRequests();
    for (auto k : r)
      LL::gDebug("Directory pending fetch: ", k.first, ",", k.second);
  }
  if (remoteObjects.size() < 10) {
    for (auto& k : remoteObjects)
      LL::gDebug("Directory remote object: ", k.first.first, ",", k.first.second);
  }
  if (notifiers.size() < 10) {
    for (auto& k : notifiers)
      LL::gDebug("Directory notification: ", k.first.first, ",", k.first.second);
  }
  if (delayed.size() < 10) {
    for (auto k : delayed)
      LL::gDebug("Directory delayed: ", k.first, ",", k.second);
  }
#endif
}

void Directory::queryObj(fatPointer ptr, bool forward) {
  LL::SLguard lg(getLock(ptr));
  tracking* tr = nullptr;
  if (hasTracking(lg, ptr))
    tr = &getExistingTracking(lg, ptr);
  LL::gDebug("Directory query for ", ptr.first, ",", ptr.second,
             ptr.first == NetworkInterface::ID ? " Mine" : " Other",
             tr ? " Tracked" : " **Untracked**",
             tr && tr->isRecalled() ? " Recalled" : "",
             tr && tr->isContended() ? " Contended" : "",
             "");

  if (forward) {
    if (ptr.first == NetworkInterface::ID && tr && tr->getCurLoc() != NetworkInterface::ID)
      getSystemNetworkInterface().sendAlt(tr->getCurLoc(), queryObjRemote, ptr, false);
    else if (ptr.first != NetworkInterface::ID)
      getSystemNetworkInterface().sendAlt(ptr.first, queryObjRemote, ptr, false);
  }
}

void Directory::queryObjRemote(fatPointer ptr, bool forward) {
  getSystemDirectory().queryObj(ptr, forward);
}

////////////////////////////////////////////////////////////////////////////////

Directory& Galois::Runtime::getSystemDirectory() {
  static Directory obj;
  return obj;
}

CacheManager& Galois::Runtime::getCacheManager() {
  static CacheManager obj;
  return obj;
}

