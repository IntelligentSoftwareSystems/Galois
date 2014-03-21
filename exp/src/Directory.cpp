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
#include <iostream>

using namespace Galois::Runtime;

#if 0

SimpleRuntimeContext& Galois::Runtime::getTransCnx() {
  static PerThreadStorage<Galois::Runtime::SimpleRuntimeContext> obj;
  return *obj.getLocal();
}

// SimpleRuntimeContext& Galois::Runtime::getAbortCnx() {
//   static Galois::Runtime::SimpleRuntimeContext obj;
//   return obj;
// }



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
  if (LockManagerBase::isAcquired(obj)) {
    if (tr.hasRequest()) {
      uint32_t wanter = tr.getRequest();
      if (!tr.isRecalled() || wanter < tr.getRecalled()) {
        tr.getHelper()->sendRequest(ptr, ptr.getHost() == NetworkInterface::ID ? tr.getCurLoc() : ptr.getHost(), wanter);
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
      LL::gDebug("Contended Protection of [", ptr.getHost(), ",", ptr.getObj(), "] from ", wanter);
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
    switch (LockManagerBase::tryAcquire(obj)) {
    case LockManagerBase::FAIL: // Local iteration has the lock
      addPending(ptr);
      return; // delay processing
    case LockManagerBase::NEW_OWNER: { //now owner (was free and on this host)
      //compute who to send the object too
      //non-owner sends back to owner
      uint32_t dest = ptr.getHost() == NetworkInterface::ID ? wanter : ptr.getHost();
      tr.getHelper()->sendObject(ptr, obj, dest);
      if (ptr.getHost() == NetworkInterface::ID) {
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
    case LockManagerBase::ALREADY_OWNER: //already owner, should have been caught above
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
      LL::gDebug("Directory pending fetch: ", k.getHost(), ",", k.getObj());
  }
  if (remoteObjects.size() < 10) {
    for (auto& k : remoteObjects)
      LL::gDebug("Directory remote object: ", k.first.getHost(), ",", k.fist.getObj());
  }
  if (notifiers.size() < 10) {
    for (auto& k : notifiers)
      LL::gDebug("Directory notification: ", k.first.getHost(), ",", k.first.getObj());
  }
  if (delayed.size() < 10) {
    for (auto k : delayed)
      LL::gDebug("Directory delayed: ", k.getHost(), ",", k.getObj());
  }
#endif
}

void Directory::queryObj(fatPointer ptr, bool forward) {
  LL::SLguard lg(getLock(ptr));
  tracking* tr = nullptr;
  if (hasTracking(lg, ptr))
    tr = &getExistingTracking(lg, ptr);
  LL::gDebug("Directory query for ", ptr.getHost(), ",", ptr.getObj(),
             ptr.getHost() == NetworkInterface::ID ? " Mine" : " Other",
             tr ? " Tracked" : " **Untracked**",
             tr && tr->isRecalled() ? " Recalled" : "",
             tr && tr->isContended() ? " Contended" : "",
             "");

  if (forward) {
    if (ptr.getHost() == NetworkInterface::ID && tr && tr->getCurLoc() != NetworkInterface::ID)
      getSystemNetworkInterface().sendAlt(tr->getCurLoc(), queryObjRemote, ptr, false);
    else if (ptr.getHost() != NetworkInterface::ID)
      getSystemNetworkInterface().sendAlt(ptr.getHost(), queryObjRemote, ptr, false);
  }
}

void Directory::queryObjRemote(fatPointer ptr, bool forward) {
  getSystemDirectory().queryObj(ptr, forward);
}

#endif

////////////////////////////////////////////////////////////////////////////////

void RemoteDirectory::makeProgress() {
  //FIXME: write
}

void RemoteDirectory::dump() {
  //FIXME: write
  std::lock_guard<LL::SimpleLock> lg(md_lock);
  for(auto& pair : md) {
    std::lock_guard<LL::SimpleLock> mdlg(pair.second.lock);
    std::cout << pair.first << ": ";
    pair.second.dump(std::cout);
    std::cout << "\n";
  }
}

void RemoteDirectory::dump(fatPointer ptr) {
  //FIXME: write
}

void RemoteDirectory::setContended(fatPointer ptr) {
  //FIXME: write
  trace("RemoteDirectory::setContended %\n", ptr);
}

void RemoteDirectory::clearContended(fatPointer ptr) {
  //FIXME: write
  trace("RemoteDirectory::clearContended %\n", ptr);
}

RemoteDirectory::metadata* RemoteDirectory::getMD(fatPointer ptr) {
  std::lock_guard<LL::SimpleLock> lg(md_lock);
  assert(ptr.getHost() != NetworkInterface::ID);
  return &md[ptr];
}

//Recieve OK to upgrade RO -> RW
void RemoteDirectory::recvUpgrade(fatPointer ptr) {
  trace("RemoteDirectory::recvUpgrade %\n", ptr);
  metadata* md = getMD(ptr);
  std::lock_guard<LL::SimpleLock> lg(md->lock);
  assert(md->state == metadata::UPGRADE);
  md->state = metadata::HERE_RW;
}

//Recieve request to invalidate object
void RemoteDirectory::recvInvalidate(fatPointer ptr) {
  trace("RemoteDirectory::recvInvalidate %\n", ptr);
  metadata* md = getMD(ptr);
  std::lock_guard<LL::SimpleLock> lg(md->lock);
  assert(md->state == metadata::HERE_RO || md->state == metadata::HERE_RW);
  if (md->state == metadata::HERE_RW)
    writeback.push_back({ptr, md->obj});
  md->state = metadata::INVALID;
  md->obj = nullptr;
}

//Recieve request to transition from RW -> RO
void RemoteDirectory::recvDemote(fatPointer ptr, typeHelper* th) {
  trace("RemoteDirectory::recvDemote % %\n", ptr, th);
  metadata* md = getMD(ptr);
  std::lock_guard<LL::SimpleLock> lg(md->lock);
  assert(md->state == metadata::HERE_RW);
  writeback.push_back({ptr, md->obj});
  //duplicate object so writeback can delete RW copy
  md->obj = th->duplicate(md->obj);
  md->state = metadata::HERE_RO;
}

void RemoteDirectory::recvRequestImpl(fatPointer ptr, ResolveFlag flag, typeHelper* th) {
  switch (flag) {
  case INV:
    recvInvalidate(ptr);
    break;
  case RO:
    recvDemote(ptr, th);
    break;
  default:
    abort();
  }
}

void RemoteDirectory::recvObjectImpl(fatPointer ptr, ResolveFlag flag, Lockable* obj) {
  trace("RemoteDirectory::recvObject % % %\n", ptr, flag, obj);
  metadata* md = getMD(ptr);
  std::lock_guard<LL::SimpleLock> lg(md->lock);
  assert(md->obj == nullptr);
  assert((md->state == metadata::PENDING_RO && flag == RO)
         || (md->state == metadata::PENDING_RW && flag == RW));
  md->state = flag == RW ? metadata::HERE_RW : metadata::HERE_RO;
  md->obj = obj;
}

//! precondition: md is locked
void RemoteDirectory::resolveNotPresent(fatPointer ptr, ResolveFlag flag, metadata* md, typeHelper* th) {
  if ((flag == RW && (md->state == metadata::UPGRADE || md->state == metadata::PENDING_RW)) ||
      (flag == RO && (md->state == metadata::PENDING_RO)))
    return;
  if (flag == RW) {
    switch(md->state) {
    case metadata::INVALID:
    case metadata::PENDING_RO:
      md->state = metadata::PENDING_RW;
      break;
    case metadata::PENDING_RW:
    case metadata::UPGRADE:
      return;
    default:
      abort();
    };
  } else if (flag == RO) {
    switch(md->state) {
    case metadata::INVALID:
      md->state = metadata::PENDING_RO;
      break;
    case metadata::PENDING_RO:
      return;
    default:
      abort();
    };
  }

  SendBuffer buf;
  gSerialize(buf, ptr, flag, NetworkInterface::ID);
  getSystemNetworkInterface().send(ptr.getHost(), th->localRequestPad, buf);
}

void RemoteDirectory::metadata::dump(std::ostream& os) const {
  static const char* StateFlagNames[] = {"I", "PR", "PW", "RO", "RW", "UW"};
  os << "{" << StateFlagNames[state] << " " << obj << "}";
}



////////////////////////////////////////////////////////////////////////////////

void LocalDirectory::recvRequestImpl(fatPointer ptr, ResolveFlag flag, uint32_t dest, typeHelper* th) {
  std::lock_guard<LL::SimpleLock> lg(reqs_lock);
  assert(ptr.getHost() == NetworkInterface::ID);
  trace("LocalDirectory::recvRequestImpl % % %\n", ptr, flag, dest);
  reqs.insert(std::make_pair(static_cast<Lockable*>(ptr.getObj()), std::make_pair(dest, flag)));
}

void LocalDirectory::recvObjectImpl(fatPointer ptr) {
  metadata* md = getMD(static_cast<Lockable*>(ptr.getObj()));
  uint32_t self = NetworkInterface::ID;
  assert(ptr.getHost() == self);
  assert(md);
  assert(md->locRW != self);
  assert(md->locRO.empty());
  std::lock_guard<LL::SimpleLock> lg(md->lock);
  md->locRW = self;
}


void LocalDirectory::makeProgress() {
  //FIXME: write
  
}

void LocalDirectory::dump() {
  //FIXME: write
}

////////////////////////////////////////////////////////////////////////////////

LocalDirectory& Galois::Runtime::getLocalDirectory() {
  static LocalDirectory obj;
  return obj;
}

RemoteDirectory& Galois::Runtime::getRemoteDirectory() {
  static RemoteDirectory obj;
  return obj;
}

CacheManager& Galois::Runtime::getCacheManager() {
  static CacheManager obj;
  return obj;
}

