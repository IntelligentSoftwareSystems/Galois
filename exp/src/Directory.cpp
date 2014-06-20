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


//FIXME: if contended, say we want it back.  What mode though?
void RemoteDirectory::doInvalidate(metadata& md, fatPointer ptr) {
  trace("RemoteDirectory::doInvalidate % md %\n", ptr, md);
  md.invalidate();
  getCacheManager().evict(ptr);
  SendBuffer b;
  gSerialize(b, ptr, NetworkInterface::ID);
  getSystemNetworkInterface().send(ptr.getHost(), &LocalDirectory::ackInvalidate, b);
}

//FIXME: if contended, say we want it back.  What mode though?
void RemoteDirectory::tryWriteBack(metadata& md, fatPointer ptr) {
  auto& cm = getCacheManager();
  auto* vobj = cm.resolveIncRef(ptr);
  assert(vobj);
  assert(vobj->getObj());
  Lockable* obj = static_cast<Lockable*>(vobj->getObj());
  if (dirAcquire(obj)) {
    trace("RemoteDirectory::tryWriteBack % md %\n", ptr, md);
    assert(cm.isCurrent(ptr, vobj->getObj()));
    md.invalidate();
    SendBuffer b;
    gSerialize(b, ptr);
    md.th->serialize(b, obj);
    cm.evict(ptr); //IncRef keeps obj live for release
    dirRelease(obj);
    getSystemNetworkInterface().send(ptr.getHost(), &LocalDirectory::recvObject, b);
  }
  vobj->decRef();
}

void RemoteDirectory::recvInvalidateImpl(fatPointer ptr, uint32_t dest) {
  metadata& md = getMD(ptr);
  std::lock_guard<LL::SimpleLock> lg(md.lock, std::adopt_lock);
  trace("RemoteDirectory::recvInvalidate % for % md %\n", ptr, dest, md);
  assert(md.state == metadata::HERE_RO);
  md.recalled = std::min(dest, md.recalled);
  if (md.contended && md.recalled > NetworkInterface::ID) {
  } else {
    doInvalidate(md, ptr);
  }
}

void RemoteDirectory::recvInvalidate(RecvBuffer& buf) {
  fatPointer ptr;
  uint32_t dest;
  gDeserialize(buf, ptr, dest);
  getRemoteDirectory().recvInvalidateImpl(ptr, dest);
}

void RemoteDirectory::recvRequestImpl(fatPointer ptr, uint32_t dest) {
  metadata& md = getMD(ptr);
  std::lock_guard<LL::SimpleLock> lg(md.lock, std::adopt_lock);
  trace("RemoteDirectory::recvRequest % dest % md %\n", ptr, dest, md);
  assert(md.state == metadata::HERE_RW);
  md.recalled = std::min(md.recalled, dest);
  if (md.contended && md.recalled > NetworkInterface::ID) {
  } else {
    tryWriteBack(md, ptr);
  }
}

void RemoteDirectory::recvRequest(RecvBuffer& buf) {
  fatPointer ptr;
  uint32_t dest;
  gDeserialize(buf, ptr, dest);
  getRemoteDirectory().recvRequestImpl(ptr, dest);
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

void RemoteDirectory::setContended(fatPointer ptr) {
  metadata& md = getMD(ptr);
  std::lock_guard<LL::SimpleLock> lg(md.lock, std::adopt_lock);
  trace("RemoteDirectory::setContended % md %\n", ptr, md);
  if (!md.contended) {
    md.contended = true;
    assert(md.state != metadata::INVALID);
  }
}

void RemoteDirectory::clearContended(fatPointer ptr) {
  metadata& md = getMD(ptr);
  std::lock_guard<LL::SimpleLock> lg(md.lock, std::adopt_lock);
  trace("RemoteDirectory::clearContended % md %\n", ptr, md);
  md.contended = false;
  if (md.recalled != ~0) {
    if (md.state == metadata::HERE_RO) {
      doInvalidate(md, ptr);
    } else {
      assert(md.state == metadata::HERE_RW);
      tryWriteBack(md, ptr);
    }
  }
}



////////////////////////////////////////////////////////////////////////////////
// Remote Directory Metadata
////////////////////////////////////////////////////////////////////////////////

RemoteDirectory::metadata::metadata() 
  :state(INVALID), recalled(~0), contended(false), th(nullptr)
{}

ResolveFlag RemoteDirectory::metadata::fetch(ResolveFlag flag) {
  assert(flag != INV);
  if (flag == RW) {
    switch (state) {
    case metadata::INVALID:
    case metadata::PENDING_RO:
      state = metadata::PENDING_RW;
      return flag;
    case metadata::HERE_RO:
      state = metadata::UPGRADE;
      return flag;
    default:
      return INV;
    }
  } else {
    switch (state) {
    case metadata::INVALID:
      state = metadata::PENDING_RO;
      return flag;
    default:
      return INV;
    }
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

void RemoteDirectory::metadata::invalidate() {
  state = INVALID;
  recalled = ~0;
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
  // //FIXME: make safe
  // decltype(writeback) q;
  // q.swap(writeback);
  // for (auto& wb : q) {
  //   if (dirAcquire(std::get<1>(wb))) {
  //     std::get<2>(wb)->writebackObj(std::get<0>(wb), std::get<1>(wb));
  //     dirRelease(std::get<1>(wb));
  //     delete std::get<1>(wb);
  //   } else {
  //     writeback.push_back(wb);
  //   }
  // }
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
  outstandingReqs = 1;
}

void LocalDirectory::ackInvalidateImpl(fatPointer ptr, uint32_t dest) {
  assert(ptr.isLocal());
  metadata& md = getMD(static_cast<Lockable*>(ptr.getObj()));
  std::lock_guard<LL::SimpleLock> lg(md.lock, std::adopt_lock);
  trace("LocalDirectory::ackInvalidate % dest % md %\n", ptr, dest, md);
  assert(md.locRO.count(dest));
  md.locRO.erase(dest);
  outstandingReqs = 1;
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
    outstandingReqs = 1; //reprocess outstanding reqs now that some may go forward
}

void LocalDirectory::sendObj(metadata& md, uint32_t dest, Lockable* obj, ResolveFlag flag) {
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

void LocalDirectory::invalidateReaders(metadata& md, Lockable* obj) {
  assert(md.locRW == ~0);
  auto& net = getSystemNetworkInterface();
  for (auto dest : md.locRO) {
    SendBuffer b;
    gSerialize(b, fatPointer(obj, fatPointer::thisHost));
    net.send(dest, &RemoteDirectory::recvInvalidate, b);
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
    trace("LocalDirectory::considerObject % recalling for % md(post) %\n", (void*)obj, nextDest, md);
    SendBuffer buf;
    gSerialize(buf, fatPointer(obj, fatPointer::thisHost), nextDest);
    getSystemNetworkInterface().send(md.locRW, &RemoteDirectory::recvRequest, buf);
    //FIXME: send request
    return; //nothing to do until we have the object
  }
  //Active Readers?
  if (!md.locRO.empty()) {
    //Issue invalidates to readers
    invalidateReaders(md, obj);
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
    } else {
      outstandingReqs = 1;
    }
  } else { //RO
    //FIXME: lock RO
    if (dirAcquire(obj)) {
      assert(md.reqsRO.count(nextDest));
      md.reqsRO.erase(nextDest);
      sendObj(md, nextDest, obj, RO);
    } else {
      outstandingReqs = 1;
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
  //  if (outstandingReqs) {
    outstandingReqs = 0; // clear flag before examining requests
    //inefficient, long held lock? (JUST A GUESS)
    std::lock_guard<LL::SimpleLock> lg(dir_lock);
    for (auto ii = dir.begin(), ee = dir.end(); ii != ee; ++ii) {
      std::lock_guard<LL::SimpleLock> obj_lg(ii->second.lock);
      considerObject(ii->second, ii->first);
    }
    //  }
}

void LocalDirectory::dump() {
  //FIXME: write
  std::lock_guard<LL::SimpleLock> lg(dir_lock);
  trace("LocalDirectory::dump % outstandingReqs\n");
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


