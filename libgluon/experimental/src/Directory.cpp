/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#include "galois/runtime/DistSupport.h"
#include "galois/runtime/Directory.h"
#include "galois/runtime/ll/TID.h"

#include <mutex>
#include <iostream>

using namespace galois::runtime;

////////////////////////////////////////////////////////////////////////////////
// MetaHolder
////////////////////////////////////////////////////////////////////////////////

template <typename metadata>
metadata& internal::MetaHolder<metadata>::getMD(fatPointer ptr,
                                                typeHelper* th) {
  std::lock_guard<LL::SimpleLock> lg(md_lock);
  auto ii = md.find(ptr);
  if (ii == md.end())
    ii = md.emplace(ptr, th).first;
  ii->second.lock.lock();
  return ii->second;
}

template <typename metadata>
metadata* internal::MetaHolder<metadata>::getMD_ifext(fatPointer ptr) {
  std::lock_guard<LL::SimpleLock> lg(md_lock);
  auto ii = md.find(ptr);
  if (ii != md.end()) {
    ii->second.lock.lock();
    return &ii->second;
  }
  return nullptr;
}

template <typename metadata>
size_t internal::MetaHolder<metadata>::mapSize() {
  return md.size();
}

template <typename metadata>
void internal::MetaHolder<metadata>::eraseMD(
    fatPointer ptr, std::unique_lock<LL::SimpleLock>& mdl) {
  // FIXME: will deadlock
  std::lock_guard<LL::SimpleLock> lg(md_lock);
  assert(md.find(ptr) != md.end());
  mdl.release();
  md.erase(ptr);
}

template <typename metadata>
void internal::MetaHolder<metadata>::dump() {
  std::lock_guard<LL::SimpleLock> lg(md_lock);
  for (auto& pair : md) {
    std::lock_guard<LL::SimpleLock> mdlg(pair.second.lock);
    std::cout << "R " << pair.first << ": " << pair.second << "\n";
  }
}
template <typename metadata>
void internal::MetaHolder<metadata>::dump(std::ofstream& dumpFileName) {
  // std::lock_guard<LL::SimpleLock> lg(md_lock);
  for (auto& pair : md) {
    // std::lock_guard<LL::SimpleLock> mdlg(pair.second.lock);
    dumpFileName << "< " << getSystemNetworkInterface().ID << " >"
                 << " R " << pair.first << ": " << pair.second << "\n";
  }
}

////////////////////////////////////////////////////////////////////////////////
// Base Directory
////////////////////////////////////////////////////////////////////////////////

bool BaseDirectory::dirAcquire(Lockable* ptr) { // FIXME: propper RO
  std::lock_guard<LL::SimpleLock> lg(dirContext_lock);
  auto rv = dirContext.tryAcquire(ptr, false);
  switch (rv) {
  case LockManagerBase::FAIL:
    return false;
  case LockManagerBase::NEW_OWNER:
    return true;
  case LockManagerBase::ALREADY_OWNER:
    assert(0 && "Already owner?");
    abort();
  default:
    abort();
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

void RemoteDirectory::addPendingReq(fatPointer ptr) {
  std::lock_guard<LL::SimpleLock> rlg(reqs_lock);
  reqs.insert(ptr);
}

bool RemoteDirectory::tryWriteBack(metadata& md, fatPointer ptr,
                                   std::unique_lock<LL::SimpleLock>& lg) {
  auto& cm   = getCacheManager();
  auto* vobj = cm.resolveIncRef(ptr);
  assert(vobj);
  assert(vobj->getObj());
  Lockable* obj = static_cast<Lockable*>(vobj->getObj());
  bool retval   = false;
  if (dirAcquire(obj)) {
    assert(cm.isCurrent(ptr, vobj->getObj()));
    cm.evict(ptr); // IncRef keeps obj live for release
    trace("RD::sendObject % to % md %\n", ptr, ptr.getHost(), md);
    md.th->send(ptr.getHost(), ptr, obj, RW);
    dirRelease(obj);
    retval = true;
  }
  vobj->decRef();
  return retval;
}

void RemoteDirectory::recvRequestImpl(fatPointer ptr, uint32_t dest,
                                      ResolveFlag flag) {
  metadata* md = dir.getMD_ifext(ptr);
  assert(md || flag == UP_RW || flag == UP_RO);

  // updates for already returned object
  if ((!md || md->state == metadata::PENDING_RW ||
       md->state == metadata::PENDING_RO) &&
      (flag == UP_RW || flag == UP_RO)) {
    // to make sure its lock is unlocked if md is not null
    if (md)
      std::unique_lock<LL::SimpleLock> lg(md->lock, std::adopt_lock);

    trace("RemoteDirectory::recvRequest % dest % DROPPING\n", ptr, dest);
    return;
  }
  std::unique_lock<LL::SimpleLock> lg(md->lock, std::adopt_lock);
  trace("RemoteDirectory::recvRequest % dest % flag % md %\n", ptr, dest, flag,
        *md);
  md->recvRequest(dest, flag);
  considerObject(*md, ptr, lg);
}

void RemoteDirectory::considerObject(metadata& md, fatPointer ptr,
                                     std::unique_lock<LL::SimpleLock>& lg) {
  assert(md.recalled != ~0U);
  // contended objects are event driven, so drop request
  if (md.contended && md.recalled > NetworkInterface::ID)
    return;

  assert(md.state == metadata::HERE_RO || md.state == metadata::HERE_RW ||
         md.state == metadata::UPGRADE);
  if (md.state == metadata::HERE_RO || md.state == metadata::UPGRADE) {
    // Recall is an invalidate
    ResolveFlag act = md.doInvalidateRO();
    // evict from cache
    getCacheManager().evict(ptr);
    // ack invalidate
    md.th->request(ptr.getHost(), ptr, NetworkInterface::ID, INV);
    if (act == INV) {
      // erase metadata
      trace("RD::ConsiderObject % md %\n", ptr, md);
      dir.eraseMD(ptr, lg);
    } else {
      // refetch
      fetchImpl(md, ptr, act, md.th, false, lg); // contended still set
    }
  } else {
    assert(md.state == metadata::HERE_RW);
    if (tryWriteBack(md, ptr, lg)) {
      if (md.doWriteBack())
        fetchImpl(md, ptr, RW, md.th, false, lg); // contended still set
      else {
        dir.eraseMD(ptr, lg);
      }
    } else {
      // wasn't able to write back, locking problems, defer
      addPendingReq(ptr);
    }
  }
}

void RemoteDirectory::recvObjectImpl(fatPointer ptr, ResolveFlag flag,
                                     internal::typeHelper* th,
                                     RecvBuffer& buf) {
  assert(flag == RW || flag == RO);
  decltype(metadata::notifyList) nl;

  { // guard lock scope
    metadata& md = dir.getMD(ptr, th);
    std::lock_guard<LL::SimpleLock> lg(md.lock, std::adopt_lock);
    trace("RD::recvObject % flag % md %\n", ptr, flag, md);
    md.recvObj(flag);
    th->cmCreate(ptr, flag, buf);
    // FIXME: handle RO locking
    std::swap(md.notifyList, nl);
  }
  // Do notifies
  for (auto& fn : nl)
    fn(ptr);
}

void RemoteDirectory::clearContended(fatPointer ptr) {
  metadata* md = dir.getMD_ifext(ptr);
  if (!md)
    return;
  std::unique_lock<LL::SimpleLock> lg(md->lock, std::adopt_lock);
  trace("RemoteDirectory::clearContended % md %\n", ptr, *md);
  md->doClearContended(); // FIXME: simple transfer rather than check
  if (md->recalled != ~0U)
    considerObject(*md, ptr, lg);
}

void RemoteDirectory::fetchImpl(metadata& md, fatPointer ptr, ResolveFlag flag,
                                internal::typeHelper* th, bool setContended,
                                std::unique_lock<LL::SimpleLock>& lg) {
  assert(!ptr.isLocal());
  ResolveFlag requestFlag = md.fetch(flag, setContended);
  if (requestFlag != INV)
    th->request(ptr.getHost(), ptr, NetworkInterface::ID, requestFlag);
}

void RemoteDirectory::fetchImpl(fatPointer ptr, ResolveFlag flag,
                                internal::typeHelper* th, bool setContended) {
  assert(!ptr.isLocal());
  metadata& md = dir.getMD(ptr, th);
  std::unique_lock<LL::SimpleLock> lg(md.lock, std::adopt_lock);
  trace("RemoteDirectory::fetch for % flag % setcont % md %\n", ptr, flag,
        setContended, md);
  fetchImpl(md, ptr, flag, th, setContended, lg);
}

bool RemoteDirectory::notify(fatPointer ptr, ResolveFlag flag,
                             std::function<void(fatPointer)> fnotify) {
  metadata* md = dir.getMD_ifext(ptr);
  if (!md)
    return false;
  std::lock_guard<LL::SimpleLock> lg(md->lock, std::adopt_lock);
  assert(flag == RW || flag == RO);
  // check if unnecessary
  if ((flag == RW && md->state == metadata::HERE_RW) ||
      (flag == RO && md->state == metadata::HERE_RO) ||
      (flag == RO && md->state == metadata::HERE_RW))
    return false;
  md->notifyList.push_back(fnotify);
  return true;
}

////////////////////////////////////////////////////////////////////////////////
// Remote Directory Metadata
////////////////////////////////////////////////////////////////////////////////

RemoteDirectory::metadata::metadata(internal::typeHelper* th)
    : state(INVALID), contended(false), recalled(~0U), th(th) {}

RemoteDirectory::metadata::~metadata() {
  assert(state == INVALID);
  assert(contended == false);
  assert(recalled == ~0U);
  assert(lock.is_locked());
}

// update state machine from local fetch and return message to send
// returning INV means send no message
ResolveFlag RemoteDirectory::metadata::fetch(ResolveFlag flag,
                                             bool setContended) {
  assert(flag != INV);
  if (!contended && setContended)
    contended = true;
  switch (flag) {
  case RW:
    switch (state) {
    case metadata::INVALID:
    case metadata::PENDING_RO: // FIXME: after PRO -> PRW, what happens on
                               // recvObj(RO)?
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

// update state machine from recieving a remote object
void RemoteDirectory::metadata::recvObj(ResolveFlag flag) {
  switch (flag) {
  case RW:
    switch (state) {
    case PENDING_RW:
    case UPGRADE:
      state = HERE_RW;
      return;
    default:
      assert(0 && "Invalid state transition");
      abort();
    }
  case RO:
    switch (state) {
    case PENDING_RO:
      state = HERE_RO;
      return;
    default:
      assert(0 && "Invalid state transition");
      abort();
    }
  default:
    assert(0 && "Invalid Flag");
    abort();
  }
}

void RemoteDirectory::metadata::recvRequest(uint32_t dest, ResolveFlag flag) {
  assert(state != INVALID);
  assert((flag == INV && (state == HERE_RO || state == UPGRADE)) ||
         flag != INV);
  assert(dest <= recalled);
  recalled = dest; // std::min(recalled, dest);
}

ResolveFlag RemoteDirectory::metadata::doInvalidateRO() {
  assert(state == HERE_RO || state == UPGRADE);
  ResolveFlag ReqMode = state == HERE_RO ? RO : RW;
  state               = INVALID;
  recalled            = ~0U;
  if (contended)
    return ReqMode;
  else
    return INV;
}

bool RemoteDirectory::metadata::doWriteBack() {
  assert(state == HERE_RW);
  state    = INVALID;
  recalled = ~0U;
  return contended;
}

// Returns whether object should be sent back
bool RemoteDirectory::metadata::doClearContended() {
  assert(state != INVALID);
  contended = false;
  return recalled != ~0U;
}

std::ostream& galois::runtime::operator<<(std::ostream& os,
                                          const RemoteDirectory::metadata& md) {
  static const char* StateFlagNames[] = {"I", "PR", "PW", "RO", "RW", "UW"};
  return os << "state:" << StateFlagNames[md.state]
            << ",contended:" << md.contended << ",recalled:" << md.recalled
            << ",th:" << md.th << " ,notifySize " << md.notifyList.size();
}

////////////////////////////////////////////////////////////////////////////////
// Old stuff
////////////////////////////////////////////////////////////////////////////////

// template<typename T>
// T* RemoteDirectory::resolve(fatPointer ptr, ResolveFlag flag) {
//   //  trace("RemoteDirectory::resolve for % flag %\n", ptr, flag);
//   metadata* md = dir.getMD(ptr);
//   std::lock_guard<LL::SimpleLock> lg(md->lock);
//   if ((flag == RO && (md->state == metadata::HERE_RO || md->state ==
//   metadata::UPGRADE)) ||
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
  for (auto& p : todo) {
    metadata* md = dir.getMD_ifext(p);
    if (md) {
      std::unique_lock<LL::SimpleLock> lg(md->lock, std::adopt_lock);
      considerObject(*md, p, lg);
    }
  }
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
  // std::lock_guard<LL::SimpleLock> lg(md_lock);
  // for (auto& pair : dir.md) {
  //   std::lock_guard<LL::SimpleLock> mdlg(pair.second.lock);
  //   std::cout << "R " << pair.first << ": " << pair.second << "\n";
  // }
  std::cout << "Remote Dir size : " << dir.mapSize() << "\n";
  dir.dump();
}

void RemoteDirectory::dump(fatPointer ptr) {
  // FIXME: write
  // auto pair = dir.md.find(ptr);
  // std::lock_guard<LL::SimpleLock> mdlg(pair.second.lock);
  // std::cout << "R " << pair.first << ": " << pair.second << "\n";
}

void RemoteDirectory::dump(std::ofstream& dumpFileName) {
  dir.dump(dumpFileName);
}

////////////////////////////////////////////////////////////////////////////////
// Local Directory
////////////////////////////////////////////////////////////////////////////////

void LocalDirectory::recvObjectImpl(fatPointer ptr, ResolveFlag flag,
                                    internal::typeHelper* th, RecvBuffer& buf) {
  assert(ptr.isLocal());
  Lockable* obj = ptr.getPtr<Lockable>();
  metadata& md  = dir.getMD(ptr, th);
  std::unique_lock<LL::SimpleLock> lg(md.lock, std::adopt_lock);
  trace("LD::recvObject % flag % md %\n", ptr, flag, md);

  // We can write back the object in place as we know we have the write lock
  assert(dirOwns(obj)); // Check that we have the write lock
  md.th->deserialize(buf, obj);
  md.doWriteback();
  considerObject(md, ptr, lg);
}

void LocalDirectory::clearContended(fatPointer ptr) {
  assert(ptr.isLocal());
  metadata* md = dir.getMD_ifext(ptr);
  if (!md)
    return;
  std::unique_lock<LL::SimpleLock> lg(md->lock, std::adopt_lock);
  trace("LocalDirectory::clearContended % md %\n", ptr, *md);
  md->doClearContended();
  considerObject(*md, ptr, lg);
}

void LocalDirectory::fetchImpl(fatPointer ptr, ResolveFlag flag,
                               internal::typeHelper* th, bool setContended) {
  // FIXME: deal with RO
  assert(ptr.isLocal());
  metadata* md = setContended ? &dir.getMD(ptr, th) : dir.getMD_ifext(ptr);
  if (!md) {
    trace("LocalDirectroy::fetch for % DROPPING\n", ptr);
    return;
  }
  std::unique_lock<LL::SimpleLock> lg(md->lock, std::adopt_lock);
  trace("LocalDirectory::fetch for % flag % setcont % md %\n", ptr, flag,
        setContended, *md);
  md->fetch(flag, setContended);
  considerObject(*md, ptr, lg);
}

void LocalDirectory::recvRequestImpl(fatPointer ptr, uint32_t dest,
                                     ResolveFlag flag,
                                     internal::typeHelper* th) {
  assert(ptr.isLocal());
  metadata& md = dir.getMD(ptr, th);
  std::unique_lock<LL::SimpleLock> lg(md.lock, std::adopt_lock);
  // std::lock_guard<LL::SimpleLock> lg(md.lock, std::adopt_lock);
  trace("LocalDirectory::recvRequest % flag % dest % md %\n", ptr, flag, dest,
        md);
  md.addReq(dest, flag);
  considerObject(md, ptr, lg);
}

void LocalDirectory::sendToReaders(metadata& md, fatPointer ptr) {
  assert(md.locRW == ~0U);
  // assert(obj is RO locked);
  for (auto dest : md.reqsRO) {
    assert(md.locRO.count(dest) == 0);
    md.th->send(dest, ptr, ptr.getPtr<Lockable>(), RO);
    md.locRO.insert(dest);
  }
  md.reqsRO.clear();
}

void LocalDirectory::invalidateReaders(metadata& md, fatPointer ptr,
                                       uint32_t nextDest) {
  assert(md.locRW == ~0U);
  for (auto dest : md.locRO) {
    md.th->request(dest, ptr, nextDest, INV);
    // leave in locRO until ack comes in
  }
}

void LocalDirectory::invalidate(fatPointer ptr) {
  //   assert(ptr.isLocal());
  //   metadata& md = dir.getMD(ptr);
  //   std::lock_guard<LL::SimpleLock> lg(md.lock, std::adopt_lock);

  //   // md.th may be null
  //   assert(md.th || !md.th);

  //   // TODO(ddn): issuing writebacks might also be possible but using INV
  //   seems
  //   // to break less directory invariants
  //   if (md.locRW != ~0)
  //     md.locRO.insert(md.locRW);

  //   for (auto dest : md.locRO)
  //     md.th->request(dest, ptr, NetworkInterface::ID, INV);

  //   if (!md.locRO.empty())
  //     trace("LocalDirectory::invalidate % md %\n", ptr, md);
  abort();
}

void LocalDirectory::considerObject(metadata& md, fatPointer ptr,
                                    std::unique_lock<LL::SimpleLock>& lg) {
  auto p = md.getNextDest();
  trace("LocalDirectory::considerObject % has dest % RW % md %\n", ptr, p.first,
        p.second, md);

  // First check if we need to update a remote host with a new request
  if (md.locRW != ~0U || !md.locRO.empty()) {
    if (md.recalled > p.first) {
      if (md.locRW != ~0U) {
        md.th->request(md.locRW, ptr, p.first,
                       (md.recalled != ~0U) ? UP_RW : RW);
      } else {
        assert(!md.locRO.empty());
        invalidateReaders(md, ptr, p.first);
      }
      md.recalled = p.first;
    }
    return;
  }

  // Object is here

  // contended objects are event driven, so drop request
  if (md.contended && p.first > NetworkInterface::ID)
    return;

  Lockable* obj = ptr.getPtr<Lockable>();

  // Next user is this host
  if (p.first == NetworkInterface::ID) {
    auto nl = std::move(md.notifyList);
    md.notifyList.clear();
    md.doSendTo(p.first, p.second);
    if (p.second) {
      assert(dirOwns(obj));
      dirRelease(obj);
    } else {
      // FIXME: RO locking
      abort();
    }
    for (auto& fn : nl)
      fn(ptr); // FIXME: md lock held

    trace("LocalDirectory::considerObject 2 After fn(ptr) % has dest % RW % md "
          "%\n",
          ptr, p.first, p.second, md);

  } else if (p.first != ~0U) {
    auto foo = NetworkInterface::ID;
    if (!dirOwns(obj)) {
      // Try acquiring lock.
      bool acq = dirAcquire(obj);
      if (!acq) { // Failed?
        addPendingReq(ptr);
        return;
      }
      if (md.contended)
        md.addReq(NetworkInterface::ID, RW);
    }
    md.doSendTo(p.first, p.second);
    if (p.second) { // RW
      trace("LD::sendObject % to % md %\n", ptr, p.first, md);
      md.th->send(p.first, ptr, ptr.getPtr<Lockable>(), RW);
    } else { // RO
      sendToReaders(md, ptr);
      abort();
    }
  }

  p = md.getNextDest();
  trace("LocalDirectory::considerObject 3 % has dest % RW % md.locRW % "
        "md.locRO % md.ishere % md %\n",
        ptr, p.first, p.second, md.locRW, md.locRW, md.isHere(), md);
  if (p.first == ~0U && !md.contended && md.isHere()) {
    trace("LocalDirectory::considerObject 3.5 EraseMD % has dest % RW % "
          "md.locRW % md.locRO % md.ishere % md %\n",
          ptr, p.first, p.second, md.locRW, md.locRW, md.isHere(), md);
    dir.eraseMD(ptr, lg);
  } else if (p.first != ~0U && !md.isHere_contended()) {
    // else if (p.first != ~0 && !md.contended) {
    trace("LocalDirectory::considerObject 4 inside else if % has dest % RW % "
          "md %\n",
          ptr, p.first, p.second, md);
    considerObject(md, ptr, lg);
  }
}

// void LocalDirectory::forwardRequest(metadata& md, fatPointer ptr) {
//   //Check that all needed messages have been sent
//   auto p = md.getNextDest();
//   assert(p.first != ~0);
//   assert(md.locRW != ~0);
//   if (md.recalled > p.first) {
//     if (md.locRW != ~0) { //RW
//       md.th->request(md.locRW, ptr, p.first, (md.recalled != 0 ) ? UP_RW : RW
//       );
//     } else { //RO
//       assert(!md.locRO.empty());
//       invalidateReaders(md, ptr, p.first);
//       abort();
//     }
//     md.recalled = p.first;
//   }
// }

bool LocalDirectory::notify(fatPointer ptr, ResolveFlag flag,
                            std::function<void(fatPointer)> fnotify) {
  metadata* md = dir.getMD_ifext(ptr);
  // metadata& md = dir.getMD(ptr);
  if (md == nullptr)
    return false;

  std::unique_lock<LL::SimpleLock> lg(md->lock, std::adopt_lock);
  assert(flag == RW || flag == RO);
  // check if unnecessary
  // if (md == nullptr || md.isHere()) {
  if (md->isHere()) {
    return false;
  }

  /*  if (md.isHere() && !(md.contended)) {
      dir.eraseMD(ptr, lg);
      return false;
    }
  */

  assert(md->notifyList.empty());
  md->notifyList.push_back(fnotify);
  return true;
}

void LocalDirectory::makeProgress() {
  std::unordered_set<fatPointer> todo;
  {
    std::lock_guard<LL::SimpleLock> lg(pending_lock);
    todo.swap(pending);
  }

  for (fatPointer ptr : todo) {
    metadata* md = dir.getMD_ifext(ptr);
    if (md) {
      std::unique_lock<LL::SimpleLock> obj_lg(md->lock, std::adopt_lock);
      auto p = md->getNextDest();
      if (p.first == ~0U)
        continue;
      else if (md->isHere())
        considerObject(*md, ptr, obj_lg);
      else
        addPendingReq(ptr);
    }
  }
  // trace("LocalDirectory::makeProgress sendBytes: % , DirSize: % , CMsize: %
  // \n", getSystemNetworkInterface().reportSendBytes(), dir.mapSize(),
  // getCacheManager().CM_size());
}

void LocalDirectory::dump() {
  // std::lock_guard<LL::SimpleLock> lg(dir_lock);
  // for (auto& pair : dir) {
  //   std::lock_guard<LL::SimpleLock> mdlg(pair.second.lock);
  //   std::cout << "L " << pair.first << ": " << pair.second << "\n";
  // }
  dir.dump();
}

void LocalDirectory::dump(std::ofstream& dumpFileName) {
  dir.dump(dumpFileName);
}
////////////////////////////////////////////////////////////////////////////////
// Local Directory Metadata
////////////////////////////////////////////////////////////////////////////////

void LocalDirectory::metadata::addReq(uint32_t dest, ResolveFlag flag) {
  switch (flag) {
  case RW:
    reqsRW.insert(dest);
    break;
  case RO:
    reqsRO.insert(dest);
    break;
  default:
    assert(0 && "Unexpected flag");
    abort();
  }
}

bool LocalDirectory::metadata::fetch(ResolveFlag flag, bool setContended) {
  if (!contended && setContended)
    contended = true;
  switch (flag) {
  case RW:
    if (locRW != ~0U) {
      reqsRW.insert(NetworkInterface::ID);
      return true;
    }
    break;
  case RO:
    if (!locRO.count(NetworkInterface::ID)) {
      reqsRO.insert(NetworkInterface::ID);
      return true;
    }
    break;
  default:
    assert(0 && "Unexpected flag");
    abort();
  }
  return false;
}

bool LocalDirectory::metadata::doWriteback() {
  assert(locRW != ~0U);
  assert(locRW != NetworkInterface::ID);
  assert(locRO.empty());
  locRW    = ~0U;
  recalled = ~0U;
  return !reqsRW.empty() || !reqsRO.empty();
}

void LocalDirectory::metadata::doSendTo(uint32_t dest, bool write) {
  assert(locRO.empty() && locRW == ~0U);
  if (write) {
    assert(reqsRW.count(dest));
    reqsRW.erase(dest);
    if (dest == NetworkInterface::ID) {
      locRW = ~0U;
    } else {
      locRW = dest;
    }
  } else {
    assert(reqsRO.count(dest));
    reqsRO.erase(dest);
    locRO.insert(dest);
  }
}

void LocalDirectory::metadata::doClearContended() {
  // bool oldCont = contended;
  contended = false;
}

////////////////////////////////////////////////////////////////////////////////

LocalDirectory& galois::runtime::getLocalDirectory() {
  static LocalDirectory obj;
  return obj;
}

RemoteDirectory& galois::runtime::getRemoteDirectory() {
  static RemoteDirectory obj;
  return obj;
}
