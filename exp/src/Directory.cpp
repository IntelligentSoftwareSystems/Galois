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
  //  md.erase(ptr);
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
  if (md.contended && dest > NetworkInterface::ID) {
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
        if (wasContended)
          fetchImpl(ptr, RW, th, true);
      } else {
        addPendingReq(ptr, dest, flag);
      }
    } else { // was RO, ack invalidate
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
  md.contended |= setContended;
  ResolveFlag requestFlag = md.fetch(flag);
  if (requestFlag != INV)
    th->request(ptr.getHost(), ptr, NetworkInterface::ID, requestFlag);
}

////////////////////////////////////////////////////////////////////////////////
// Remote Directory Metadata
////////////////////////////////////////////////////////////////////////////////

RemoteDirectory::metadata::metadata() 
  :state(INVALID), contended(false), th(nullptr)
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
  return os << "state:" << StateFlagNames[md.state] << ",contended:" << md.contended << ",th:" << md.th;
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

LocalDirectory::metadata& LocalDirectory::getMD(fatPointer ptr) {
  std::lock_guard<LL::SimpleLock> lg(dir_lock);
  metadata& md = dir[ptr];
  md.lock.lock();
  return md;
}

void LocalDirectory::eraseMD(fatPointer ptr, std::unique_lock<LL::SimpleLock>& mdl) {
  std::lock_guard<LL::SimpleLock> lg(dir_lock);
  assert(dir.find(ptr) != dir.end());
  assert(dir[ptr].lock.is_locked());
  mdl.unlock();
  dir.erase(ptr);
}

void LocalDirectory::recvObjectImpl(fatPointer ptr, ResolveFlag flag, 
                                    typeHelper* th, RecvBuffer& buf) {
  assert(ptr.isLocal());
  Lockable* obj = static_cast<Lockable*>(ptr.getObj());
  metadata& md = getMD(ptr);
  std::unique_lock<LL::SimpleLock> lg(md.lock, std::adopt_lock);
  trace("LocalDirectory::writebackObject % flag % md %\n", ptr, flag, md);
  //We can write back the object in place as we know we have the write lock
  assert(dirOwns(obj)); //Check that we have the write lock
  md.writeback();
  md.th->deserialize(buf, obj);

  auto p = md.getNextDest();
  if (p.first == ~0) {
    eraseMD(ptr, lg);
    dirRelease(obj); //Object is local now with no requesters, so release
  } else {
    considerObject(md, ptr);
  }
}

void LocalDirectory::fetchImpl(fatPointer ptr, ResolveFlag flag, typeHelper* th, bool setContended) {
  //FIXME: deal with RO
  assert(ptr.isLocal());
  metadata& md = getMD(ptr);
  std::unique_lock<LL::SimpleLock> lg(md.lock, std::adopt_lock);
  trace("LocalDirectory::fetch for % flag % md %\n", ptr, flag, md);
  assert(md.th);
  if (md.isHere()) {
    if (setContended)
      md.contended = true;
    else
      eraseMD(ptr, lg);
  } else {
    md.addReq(NetworkInterface::ID, flag);
    md.contended |= setContended;
    updatePendingRequests(md, ptr, lg);
  }
}

void LocalDirectory::clearContended(fatPointer ptr) {
  assert(ptr.isLocal());
  Lockable* obj = static_cast<Lockable*>(ptr.getObj());
  metadata& md = getMD(ptr);
  std::unique_lock<LL::SimpleLock> lg(md.lock, std::adopt_lock);
  trace("LocalDirectory::clearContended % md %\n", ptr, md);
  md.contended = false;
  if (md.isHere()) {
    auto p = md.getNextDest();
    if (p.first == ~0) //no requester
      eraseMD(ptr, lg);
    else //send it somewhere
      considerObject(md, ptr);
  }
}

void LocalDirectory::recvRequestImpl(fatPointer ptr, uint32_t dest, ResolveFlag flag, typeHelper* th) {
  assert(ptr.isLocal());
  Lockable* obj = static_cast<Lockable*>(ptr.getObj());
  metadata& md = getMD(ptr);
  std::unique_lock<LL::SimpleLock> lg(md.lock, std::adopt_lock);
  trace("LocalDirectory::recvRequest % flag % dest % md %\n", ptr, flag, dest, md);
  assert(!md.th || md.th == th);
  if (!md.th)
    md.th = th;
  if (flag == INV) {
    //ack of invalidate
    assert(md.locRO.count(dest));
    md.locRO.erase(dest);
    if (md.locRO.empty()) {
      auto p = md.getNextDest();
      if (p.first == ~0) //no requester
        eraseMD(ptr, lg);
      else //send it somewhere
        considerObject(md, ptr);
    } //else still waiting for invalidates
  } else {
    md.addReq(dest, flag);
    if (md.isHere())
      considerObject(md, ptr);
    else if (md.isRO())
      sendToReaders(md, ptr);
    else
      updatePendingRequests(md, ptr, lg);
  }
}


void LocalDirectory::sendToReaders(metadata& md, fatPointer ptr) {
  assert(md.locRW == ~0);
  //assert(obj is RO locked);
  for (auto dest : md.reqsRO) {
    assert(md.locRO.count(dest) == 0);
    md.th->send(dest, ptr, static_cast<Lockable*>(ptr.getObj()), RO);
    md.locRO.insert(dest);
  }
  md.reqsRO.clear();
}

void LocalDirectory::invalidateReaders(metadata& md, fatPointer ptr, uint32_t nextDest) {
  assert(md.locRW == ~0);
  for (auto dest : md.locRO) {
    md.th->request(dest, ptr, nextDest, INV);
    //leave in locRO until ack comes in
  }
}

void LocalDirectory::considerObject(metadata& md, fatPointer ptr) {
  //Find next destination
  uint32_t nextDest = ~0;
  bool nextIsRW = false;
  std::tie(nextDest, nextIsRW) = md.getNextDest();
  //caller should have checked that there is a request
  assert(nextDest != ~0);

  trace("LocalDirectory::considerObject % has dest % RW % remote % md %\n", ptr, nextDest, nextIsRW, (md.locRW != ~0 || !md.locRO.empty()) ? 'T' : 'F', md);

  Lockable* obj = static_cast<Lockable*>(ptr.getObj());

  //Only consider objects which are already present (but may be locked locally)
  assert(md.isHere());

  //Object is local (locRW and locRO are empty)
  assert(md.locRO.empty() && md.locRW == ~0);
  //Next user is this host
  if (nextDest == NetworkInterface::ID) {
    if (md.contended) {
      //FIXME: notify
    }
    if (nextIsRW) {
      md.reqsRW.erase(nextDest);
      dirRelease(obj);
    } else {
      md.reqsRO.erase(nextDest);
      //FIXME: release lock
    }
  } else {
    if (dirOwns(obj) || dirAcquire(obj)) {
      if (nextIsRW) {
        assert(md.reqsRW.count(nextDest));
        md.reqsRW.erase(nextDest);
        md.locRW = nextDest;
        md.th->send(nextDest, ptr, static_cast<Lockable*>(ptr.getObj()), RW);
      } else { //RO
        sendToReaders(md, ptr);
      }
    } else { //locked by local iteration, defer
      setPending(ptr); 
    }
  }
}

void LocalDirectory::updatePendingRequests(metadata& md, fatPointer ptr, std::unique_lock<LL::SimpleLock>& lg) {
  //Check that all needed messages have been sent
  //Find next destination
  uint32_t nextDest = ~0;
  bool nextIsRW = false;
  std::tie(nextDest, nextIsRW) = md.getNextDest();
  if (nextDest != ~0) {
    if (md.recalled != nextDest) {
      if (md.locRW != ~0) {
        md.th->request(md.locRW, ptr, nextDest, RW);
      } else {
        assert(!md.locRO.empty());
        invalidateReaders(md, ptr, nextDest);
      }
      md.recalled = nextDest;
    }
  } else {
    if (!md.contended)
      eraseMD(ptr, lg);
  }
}

void LocalDirectory::makeProgress() {
  std::unordered_set<fatPointer> todo;
  {
    std::lock_guard<LL::SimpleLock> lg(pending_lock);
    todo.swap(pending);
  }

  for (fatPointer ptr : todo) {
    metadata& md = getMD(ptr);
    std::lock_guard<LL::SimpleLock> obj_lg(md.lock, std::adopt_lock);
    if (md.isHere())
      considerObject(md, ptr);
    else
      setPending(ptr);
  }
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
  recalled =  ~0;
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


