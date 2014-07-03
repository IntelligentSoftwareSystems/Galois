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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#include "Galois/Runtime/DistSupport.h"
#include "Galois/Runtime/Directory.h"
#include "Galois/Runtime/ll/TID.h"

#include <mutex>
#include <iostream>

using namespace Galois::Runtime;

////////////////////////////////////////////////////////////////////////////////
// Base Directory
////////////////////////////////////////////////////////////////////////////////

BaseDirectory::typeHelper::typeHelper(recvFuncTy rRW)
  : writeremote(rRW)
{}

bool BaseDirectory::dirAcquire(Lockable* ptr) {
  std::lock_guard<LL::SimpleLock> lg(dirContext_lock);
  auto rv = dirContext.tryAcquire(ptr);
  switch (rv) {
  case LockManagerBase::FAIL:      return false;
  case LockManagerBase::NEW_OWNER: return true;
  case LockManagerBase::ALREADY_OWNER: assert(0 && "Already owner?"); abort();
  }
}

void BaseDirectory::dirRelease(Lockable* ptr) {
  std::lock_guard<LL::SimpleLock> lg(dirContext_lock);
  dirContext.releaseOne(ptr);
}

bool BaseDirectory::dirOwns(Lockable* ptr) {
  std::lock_guard<LL::SimpleLock> lg(dirContext_lock);
  return dirContext.isAcquired(ptr);
}

////////////////////////////////////////////////////////////////////////////////
// Remote Directory
////////////////////////////////////////////////////////////////////////////////

RemoteDirectory::metadata& RemoteDirectory::getMD(fatPointer ptr) {
  std::lock_guard<LL::SimpleLock> lg(md_lock);
  assert(ptr.getHost() != NetworkInterface::ID);
  auto& retval = md[ptr];
  retval.lock.lock();
  return retval;
}

void RemoteDirectory::eraseMD(fatPointer ptr, metadata& m) {
  std::lock_guard<LL::SimpleLock> lg(md_lock);
  assert(md.find(ptr) != md.end());
  assert(&m == &md[ptr]);
  assert(m.lock.is_locked());
  md.erase(ptr);
}

void RemoteDirectory::addPendingReq(fatPointer ptr, uint32_t dest, ResolveFlag flag) {
  std::lock_guard<LL::SimpleLock> rlg(reqs_lock);
  auto ii = reqs.find(ptr);
  if (ii == reqs.end()) {
    reqs.emplace(std::make_pair(ptr, outstandingReq{dest, flag}));
  } else {
    ii->second.dest = std::min(ii->second.dest, dest);
    ii->second.flag = flag; // FIXME
  }
}

bool RemoteDirectory::tryWriteBack(metadata& md, fatPointer ptr) {
  auto& cm = getCacheManager();
  auto* vobj = cm.resolveIncRef(ptr);
  assert(vobj);
  assert(vobj->getObj());
  Lockable* obj = static_cast<Lockable*>(vobj->getObj());
  bool retval = false;
  if (dirAcquire(obj)) {
    trace("RemoteDirectory::tryWriteBack % md %\n", ptr, md);
    assert(cm.isCurrent(ptr, vobj->getObj()));
    cm.evict(ptr); //IncRef keeps obj live for release
    md.th->send(ptr.getHost(), ptr, obj, RW);
    dirRelease(obj);
    eraseMD(ptr, md);
    retval = true;
  }
  vobj->decRef();
  return retval;
}

void RemoteDirectory::recvRequestImpl(fatPointer ptr, uint32_t dest, ResolveFlag flag) {
  metadata& md = getMD(ptr);
  std::lock_guard<LL::SimpleLock> lg(md.lock, std::adopt_lock);
  trace("RemoteDirectory::recvRequest % dest % md %\n", ptr, dest, md);
  if (md.contended && md.recalled > NetworkInterface::ID) {
    addPendingReq(ptr, dest, flag);
  } else {
    bool wasContended = md.contended;
    ResolveFlag oldMode = INV;
    typeHelper* th = md.th;
    switch(md.state) {
    case metadata::HERE_RW:
    case metadata::UPGRADE:
      oldMode = RW;
      break;
    case metadata::HERE_RO:
      oldMode = RO;
      break;
    default:
      assert(0 && "Invalid Mode");
      abort();
    }
    if (md.state == metadata::HERE_RW) {
      if (tryWriteBack(md, ptr)) {
        eraseMD(ptr, md);
        if (wasContended)
          fetchImpl(ptr, RW, th, true);
      } else {
        addPendingReq(ptr, dest, flag);
      }
    } else { // was RO
      eraseMD(ptr, md);
      getCacheManager().evict(ptr);
      th->request(ptr.getHost(), ptr, NetworkInterface::ID, INV);
      if (wasContended)
        fetchImpl(ptr, RO, th, true);
    }
  }
}

void RemoteDirectory::recvObjectImpl(fatPointer ptr, ResolveFlag flag, typeHelper* th, RecvBuffer& buf) {
  assert(flag == RW || flag == RO);
  metadata& md = getMD(ptr);
  std::lock_guard<LL::SimpleLock> lg(md.lock, std::adopt_lock);
  trace("RemoteDirectory::recvObject % flag % md %\n", ptr, flag, md);
  md.recvObj(flag);
  assert(md.th == th || !md.th);
  if (!md.th)
    md.th = th;
  th->cmCreate(ptr, flag, buf);
  //FIXME: handle RO locking
}

void RemoteDirectory::clearContended(fatPointer ptr) {
  metadata& md = getMD(ptr);
  std::lock_guard<LL::SimpleLock> lg(md.lock, std::adopt_lock);
  trace("RemoteDirectory::clearContended % md %\n", ptr, md);
  md.contended = false;
  assert(md.state != metadata::INVALID);

  bool doRecv = false;
  outstandingReq r;
  {
    std::lock_guard<LL::SimpleLock> lg(reqs_lock);
    auto ii = reqs.find(ptr);
    if (ii != reqs.end()) {
      doRecv = true;
      r = ii->second;
      reqs.erase(ii);
    }
  }
  if (doRecv)
    recvRequestImpl(ptr, r.dest, r.flag);
}

void RemoteDirectory::fetchImpl(fatPointer ptr, ResolveFlag flag, typeHelper* th, bool setContended) {
  metadata& md = getMD(ptr);
  std::lock_guard<LL::SimpleLock> lg(md.lock, std::adopt_lock);
  trace("RemoteDirectory::fetch for % flag % md %\n", ptr, flag, md);
  assert(md.th == th || !md.th);
  assert(!ptr.isLocal());
  if (!md.th)
    md.th = th;
  if (setContended && !md.contended)
    md.contended = true;
  ResolveFlag requestFlag = md.fetch(flag);
  if (requestFlag != INV)
    th->request(ptr.getHost(), ptr, NetworkInterface::ID, requestFlag);
}

////////////////////////////////////////////////////////////////////////////////
// Remote Directory Metadata
////////////////////////////////////////////////////////////////////////////////

RemoteDirectory::metadata::metadata() 
  :state(INVALID), recalled(~0), contended(false), th(nullptr)
{}

ResolveFlag RemoteDirectory::metadata::fetch(ResolveFlag flag) {
  assert(flag != INV);
  switch(flag) {
  case RW:
    switch (state) {
    case metadata::INVALID:
    case metadata::PENDING_RO:
      state = metadata::PENDING_RW;
      return RW;
    case metadata::HERE_RO:
      state = metadata::UPGRADE;
      return RW;
    default:
      return INV;
    }
  case RO:
    switch (state) {
    case metadata::INVALID:
      state = metadata::PENDING_RO;
      return RO;
    default:
      return INV;
    }
  default:
    assert(0 && "Invalid Flag");
    abort();
  }
}

void RemoteDirectory::metadata::recvObj(ResolveFlag flag) {
  assert(flag != INV);
  assert(state != INVALID);
  assert(recalled == ~0);
  if (flag == RW) {
    switch (state) {
    case PENDING_RW:
    case UPGRADE:
      state = HERE_RW;
      return;
    default:
      assert(0 && "Invalid state transition");
      abort();
    }
  } else {
    switch (state) {
    case PENDING_RO:
      state = HERE_RO;
      return;
    default:
      assert(0 && "Invalid state transition");
      abort();
    }
  }
}

std::ostream& Galois::Runtime::operator<<(std::ostream& os, const RemoteDirectory::metadata& md) {
  static const char* StateFlagNames[] = {"I", "PR", "PW", "RO", "RW", "UW"};
  return os << "state:" << StateFlagNames[md.state] << ",recalled:" << md.recalled << ",contended:" << md.contended << ",th:" << md.th;
}

////////////////////////////////////////////////////////////////////////////////
// Old stuff
////////////////////////////////////////////////////////////////////////////////

// template<typename T>
// T* RemoteDirectory::resolve(fatPointer ptr, ResolveFlag flag) {
//   //  trace("RemoteDirectory::resolve for % flag %\n", ptr, flag);
//   metadata* md = getMD(ptr);
//   std::lock_guard<LL::SimpleLock> lg(md->lock);
//   if ((flag == RO && (md->state == metadata::HERE_RO || md->state == metadata::UPGRADE)) ||
//       (flag == RW &&  md->state == metadata::HERE_RW)) {
//     assert(md->obj);
//     return static_cast<T*>(md->obj);
//   } else {
//     resolveNotPresent(ptr, flag, md, typeHelperImpl<T>::get());
//     return nullptr;
//   }
// }


void RemoteDirectory::makeProgress() {
  decltype(reqs) todo;
  {
    std::lock_guard<LL::SimpleLock> lg(reqs_lock);
    todo.swap(reqs);
  }
  for (auto p : todo)
    recvRequestImpl(p.first, p.second.dest, p.second.flag);
}

void RemoteDirectory::resetStats() {
  // sentRequests = 0;
  // sentInvalidateAcks = 0;
  // sentObjects = 0;
}

void RemoteDirectory::reportStats(const char* loopname) {
  // reportStat(loopname, "RD::sentRequests", sentRequests);
  // reportStat(loopname, "RD::sentInvalidateAcks", sentInvalidateAcks);
  // reportStat(loopname, "RD::sentObjects", sentObjects);
}

void RemoteDirectory::dump() {
  std::lock_guard<LL::SimpleLock> lg(md_lock);
  for(auto& pair : md) {
    std::lock_guard<LL::SimpleLock> mdlg(pair.second.lock);
    std::cout << pair.first << ": " << pair.second << "\n";
  }
}

void RemoteDirectory::dump(fatPointer ptr) {
  //FIXME: write
}


////////////////////////////////////////////////////////////////////////////////
// Local Directory
////////////////////////////////////////////////////////////////////////////////

LocalDirectory::metadata& LocalDirectory::getMD(Lockable* ptr) {
  std::lock_guard<LL::SimpleLock> lg(dir_lock);
  metadata& md = dir[ptr];
  md.lock.lock();
  return md;
}


void LocalDirectory::recvRequestImpl(fatPointer ptr, ResolveFlag flag, uint32_t dest, typeHelper* th) {
  assert(ptr.isLocal());
  metadata& md = getMD(static_cast<Lockable*>(ptr.getObj()));
  std::lock_guard<LL::SimpleLock> lg(md.lock, std::adopt_lock);
  trace("LocalDirectory::recvRequest % flag % dest % md %\n", ptr, flag, dest, md);
  assert(!md.th || md.th == th);
  if (!md.th)
    md.th = th;
  md.addReq(dest, flag);
  setPending(static_cast<Lockable*>(ptr.getObj()));
}

void LocalDirectory::ackInvalidateImpl(fatPointer ptr, uint32_t dest) {
  assert(ptr.isLocal());
  metadata& md = getMD(static_cast<Lockable*>(ptr.getObj()));
  std::lock_guard<LL::SimpleLock> lg(md.lock, std::adopt_lock);
  trace("LocalDirectory::ackInvalidate % dest % md %\n", ptr, dest, md);
  assert(md.locRO.count(dest));
  md.locRO.erase(dest);
  setPending(static_cast<Lockable*>(ptr.getObj()));
}

void LocalDirectory::ackInvalidate(RecvBuffer& buf) {
  fatPointer ptr;
  uint32_t dest;
  gDeserialize(buf, ptr, dest);
  getLocalDirectory().ackInvalidateImpl(ptr, dest);
}

void LocalDirectory::recvObject(RecvBuffer& buf) {
  fatPointer ptr;
  gDeserialize(buf, ptr);
  getLocalDirectory().recvObjectImpl(ptr, buf);
}

void LocalDirectory::recvObjectImpl(fatPointer ptr, RecvBuffer& buf) {
  assert(ptr.isLocal());
  metadata& md = getMD(static_cast<Lockable*>(ptr.getObj()));
  std::lock_guard<LL::SimpleLock> lg(md.lock, std::adopt_lock);
  trace("LocalDirectory::writebackObject % md %\n", ptr, md);
  Lockable* obj = static_cast<Lockable*>(ptr.getObj());
  //We can write back the object in place as we know we have the write lock
  assert(dirOwns(obj)); //Check that we have the write lock
  md.th->deserialize(buf, obj);
  dirRelease(obj); //Object is local now, so release.  Other requesters will be serviced on makeProgress
  assert(md.locRW != ~0);
  assert(md.locRO.empty());
  md.locRW = ~0;
  md.recalled = ~0;
  if (!md.reqsRW.empty() || !md.reqsRO.empty())
    setPending(static_cast<Lockable*>(ptr.getObj()));
}

void LocalDirectory::sendObj(metadata& md, uint32_t dest, Lockable* obj, ResolveFlag flag) {
  ++sentObjects;
  trace("LocalDirectory::sendObj % dest % flag % md %\n", obj, dest, flag, md);
  SendBuffer b;
  gSerialize(b, fatPointer(obj, fatPointer::thisHost), flag);
  md.th->serialize(b, obj);
  getSystemNetworkInterface().send(dest, md.th->writeremote, b);
  switch (flag) {
  case RO: md.locRO.insert(dest); break;
  case RW: md.locRW = dest; break;
  default: abort();
  }
}

void LocalDirectory::sendToReaders(metadata& md, Lockable* obj) {
  assert(md.locRW == ~0);
  //assert(obj is RO locked);
  for (auto dest : md.reqsRO) {
    assert(md.locRO.count(dest) == 0);
    sendObj(md, dest, obj, ResolveFlag::RO);
    assert(md.locRO.count(dest));
  }
  md.reqsRO.clear();
}

void LocalDirectory::invalidateReaders(metadata& md, Lockable* obj, uint32_t nextDest) {
  assert(md.locRW == ~0);
  for (auto dest : md.locRO) {
    ++sentInvalidates;
    md.th->request(dest, fatPointer(obj, fatPointer::thisHost), nextDest, INV);
    //leave in locRO until ack comes in
  }
}

void LocalDirectory::considerObject(metadata& md, Lockable* obj) {
  //Find next destination
  uint32_t nextDest = ~0;
  bool nextIsRW = false;
  std::tie(nextDest, nextIsRW) = md.getNextDest();
  //nothing to do?
  if(nextDest == ~0)
    return;

  trace("LocalDirectory::considerObject % has dest % RW % remote % md %\n", (void*)obj, nextDest, nextIsRW, (md.locRW != ~0 || !md.locRO.empty()) ? 'T' : 'F', md);

  setPending(obj);

  //Currently RO and next is RO
  if (!nextIsRW && !md.locRO.empty()) {
    assert(md.recalled == ~0);
    sendToReaders(md, obj);
    return;
  }
  //already recalled
  if (md.recalled == nextDest)
    return;
  //Active writer?
  if (md.locRW != ~0) {
    //newer requester than last recall, so update recall
    md.recalled = nextDest;
    ++sentRequests;
    trace("LocalDirectory::considerObject % recalling for % md(post) %\n", (void*)obj, nextDest, md);
    md.th->request(md.locRW, fatPointer(obj, fatPointer::thisHost), nextDest, RW);
    return; //nothing to do until we have the object
  }
  //Active Readers?
  if (!md.locRO.empty()) {
    //Issue invalidates to readers
    invalidateReaders(md, obj, nextDest);
    return; // nothing to do until we have acked the invalidates
  }
  //Object is local (locRW and locRO are empty)
  assert(md.locRO.empty() && md.locRW == ~0);
  //Next user is this host
  if (nextDest == NetworkInterface::ID) {
    if (md.contended) {
      //FIXME: notify
      return;
    } else {
      if (nextIsRW)
        md.reqsRW.erase(nextDest);
      else
        md.reqsRO.erase(nextDest);
      return;
    }
  }

  if (nextIsRW) {
    //lock RW
    if (dirAcquire(obj)) {
      assert(md.reqsRW.count(nextDest));
      md.reqsRW.erase(nextDest);
      sendObj(md, nextDest, obj, RW);
    }
  } else { //RO
    //FIXME: lock RO
    if (dirAcquire(obj)) {
      assert(md.reqsRO.count(nextDest));
      md.reqsRO.erase(nextDest);
      sendObj(md, nextDest, obj, RO);
    }
  }
}


void LocalDirectory::setContended(Lockable* ptr) {
  metadata& md = getMD(ptr);
  std::lock_guard<LL::SimpleLock> lg(md.lock, std::adopt_lock);
  trace("LocalDirectory::setContended % md %\n", ptr, md);
  if (!md.contended)
    md.contended = true;
}

void LocalDirectory::setContended(fatPointer ptr) {
  assert(ptr.isLocal());
  setContended(static_cast<Lockable*>(ptr.getObj()));
}


void LocalDirectory::clearContended(Lockable* ptr) {
  metadata& md = getMD(ptr);
  std::lock_guard<LL::SimpleLock> lg(md.lock, std::adopt_lock);
  trace("LocalDirectory::clearContended % md %\n", ptr, md);
  md.contended = false;
  considerObject(md, ptr);
}

void LocalDirectory::clearContended(fatPointer ptr) {
  assert(ptr.isLocal());
  clearContended(static_cast<Lockable*>(ptr.getObj()));
}

void LocalDirectory::makeProgress() {
  std::unordered_set<Lockable*> todo;
  {
    std::lock_guard<LL::SimpleLock> lg(pending_lock);
    todo.swap(pending);
  }

  for (Lockable* obj : todo) {
    metadata& md = getMD(obj);
    std::lock_guard<LL::SimpleLock> obj_lg(md.lock, std::adopt_lock);
    considerObject(md, obj);
  }
}

void LocalDirectory::resetStats() {
  sentRequests = 0;
  sentInvalidates = 0;
  sentObjects = 0;
}

void LocalDirectory::reportStats(const char* loopname) {
  reportStat(loopname, "LD::sentRequests", sentRequests);
  reportStat(loopname, "LD::sentInvalidates", sentInvalidates);
  reportStat(loopname, "LD::sentObjects", sentObjects);
}

void LocalDirectory::dump() {
  //FIXME: write
  //std::lock_guard<LL::SimpleLock> lg(dir_lock);
  trace("LocalDirectory::dump\n");
}

////////////////////////////////////////////////////////////////////////////////
// Local Directory Metadata
////////////////////////////////////////////////////////////////////////////////

void LocalDirectory::metadata::addReq(uint32_t dest, ResolveFlag flag) {
  switch (flag) {
  case RW: reqsRW.insert(dest); break;
  case RO: reqsRO.insert(dest); break;
  default: assert( 0 && "Unexpected flag"); abort();
  }
}

bool LocalDirectory::metadata::writeback() {
  assert(locRW != ~0);
  assert(locRW != NetworkInterface::ID);
  assert(locRO.empty());
  locRW = ~0;
  return !reqsRW.empty() || !reqsRO.empty();
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


