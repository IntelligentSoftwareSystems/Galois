/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

#ifndef GALOIS_SUBSTRATE_PERTHREADSTORAGE_H
#define GALOIS_SUBSTRATE_PERTHREADSTORAGE_H

#include <cassert>
#include <cstddef>
#include <utility>
#include <vector>

#include "galois/config.h"
#include "galois/substrate/HWTopo.h"
#include "galois/substrate/PaddedLock.h"
#include "galois/substrate/ThreadPool.h"

namespace galois {
namespace substrate {

class PerBackend {
  typedef substrate::SimpleLock Lock;

  std::atomic<unsigned int> nextLoc{0};
  std::atomic<char*>* heads{nullptr};
  Lock freeOffsetsLock;
  std::vector<std::vector<unsigned>> freeOffsets;
  /**
   * Guards access to non-POD objects that can be accessed after PerBackend
   * is destroyed. Access can occur through destroying PerThread/PerSocket
   * objects with static storage duration, which have a reference to a
   * PerBackend object, which may have be destroyed before the PerThread
   * object itself.
   */
  bool invalid{false};

  void initCommon(unsigned maxT);
  static unsigned nextLog2(unsigned size);

public:
  PerBackend();

  PerBackend(const PerBackend&)            = delete;
  PerBackend& operator=(const PerBackend&) = delete;

  ~PerBackend() {
    // Intentionally leak heads so that other PerThread operations are
    // still valid after we are gone
    invalid = true;
  }

  char* initPerThread(unsigned maxT);
  char* initPerSocket(unsigned maxT);

  unsigned allocOffset(const unsigned size);
  void deallocOffset(const unsigned offset, const unsigned size);
  void* getRemote(unsigned thread, unsigned offset);
  void* getLocal(unsigned offset, char* base) { return &base[offset]; }
  // faster when (1) you already know the id and (2) shared access to heads is
  // not to expensive; otherwise use getLocal(unsigned,char*)
  void* getLocal(unsigned offset, unsigned id) { return &heads[id][offset]; }
};

extern thread_local char* ptsBase;
PerBackend& getPTSBackend();

extern thread_local char* pssBase;
PerBackend& getPPSBackend();

void initPTS(unsigned maxT);

template <typename T>
class PerThreadStorage {
protected:
  PerBackend* b;
  unsigned offset;

  void destruct() {
    if (offset == ~0U)
      return;

    for (unsigned n = 0; n < getThreadPool().getMaxThreads(); ++n)
      reinterpret_cast<T*>(b->getRemote(n, offset))->~T();
    b->deallocOffset(offset, sizeof(T));
    offset = ~0U;
  }

public:
  // construct on each thread
  template <typename... Args>
  PerThreadStorage(Args&&... args) : b(&getPTSBackend()) {
    // in case we make one of these before initializing the thread pool
    // This will call initPTS for each thread if it hasn't already
    auto& tp = getThreadPool();

    offset = b->allocOffset(sizeof(T));
    for (unsigned n = 0; n < tp.getMaxThreads(); ++n)
      new (b->getRemote(n, offset)) T(std::forward<Args>(args)...);
  }

  PerThreadStorage(PerThreadStorage&& rhs) : b(rhs.b), offset(rhs.offset) {
    rhs.offset = ~0;
  }

  ~PerThreadStorage() { destruct(); }

  PerThreadStorage& operator=(PerThreadStorage&& rhs) {
    std::swap(offset, rhs.offset);
    std::swap(b, rhs.b);
    return *this;
  }

  T* getLocal() {
    void* ditem = b->getLocal(offset, ptsBase);
    return reinterpret_cast<T*>(ditem);
  }

  const T* getLocal() const {
    void* ditem = b->getLocal(offset, ptsBase);
    return reinterpret_cast<T*>(ditem);
  }

  //! Like getLocal() but optimized for when you already know the thread id
  T* getLocal(unsigned int thread) {
    void* ditem = b->getLocal(offset, thread);
    return reinterpret_cast<T*>(ditem);
  }

  const T* getLocal(unsigned int thread) const {
    void* ditem = b->getLocal(offset, thread);
    return reinterpret_cast<T*>(ditem);
  }

  T* getRemote(unsigned int thread) {
    void* ditem = b->getRemote(thread, offset);
    return reinterpret_cast<T*>(ditem);
  }

  const T* getRemote(unsigned int thread) const {
    void* ditem = b->getRemote(thread, offset);
    return reinterpret_cast<T*>(ditem);
  }

  T* get(unsigned int thread) { return getRemote(thread); }

  const T* get(unsigned int thread) const { return getRemote(thread); }

  unsigned size() const { return getThreadPool().getMaxThreads(); }
};

template <typename T>
class PerSocketStorage {
protected:
  unsigned offset;
  PerBackend& b;

  void destruct() {
    auto& tp = getThreadPool();
    for (unsigned n = 0; n < tp.getMaxSockets(); ++n)
      reinterpret_cast<T*>(b.getRemote(tp.getLeaderForSocket(n), offset))->~T();
    b.deallocOffset(offset, sizeof(T));
  }

public:
  template <typename... Args>
  PerSocketStorage(Args&&... args) : b(getPPSBackend()) {
    // in case we make one of these before initializing the thread pool
    // This will call initPTS for each thread if it hasn't already
    getThreadPool();

    offset   = b.allocOffset(sizeof(T));
    auto& tp = getThreadPool();
    for (unsigned n = 0; n < tp.getMaxSockets(); ++n)
      new (b.getRemote(tp.getLeaderForSocket(n), offset))
          T(std::forward<Args>(args)...);
  }

  PerSocketStorage(PerSocketStorage&& o)
      : offset(std::move(o.offset)), b(getPPSBackend()) {}
  PerSocketStorage& operator=(PerSocketStorage&& o) {
    destruct();
    offset = std::move(o.offset);
    return *this;
  }

  PerSocketStorage(const PerSocketStorage&)            = delete;
  PerSocketStorage& operator=(const PerSocketStorage&) = delete;

  ~PerSocketStorage() { destruct(); }

  T* getLocal() {
    void* ditem = b.getLocal(offset, pssBase);
    return reinterpret_cast<T*>(ditem);
  }

  const T* getLocal() const {
    void* ditem = b.getLocal(offset, pssBase);
    return reinterpret_cast<T*>(ditem);
  }

  //! Like getLocal() but optimized for when you already know the thread id
  T* getLocal(unsigned int thread) {
    void* ditem = b.getLocal(offset, thread);
    return reinterpret_cast<T*>(ditem);
  }

  const T* getLocal(unsigned int thread) const {
    void* ditem = b.getLocal(offset, thread);
    return reinterpret_cast<T*>(ditem);
  }

  T* getRemote(unsigned int thread) {
    void* ditem = b.getRemote(thread, offset);
    return reinterpret_cast<T*>(ditem);
  }

  const T* getRemote(unsigned int thread) const {
    void* ditem = b.getRemote(thread, offset);
    return reinterpret_cast<T*>(ditem);
  }

  T* getRemoteByPkg(unsigned int pkg) {
    void* ditem = b.getRemote(getThreadPool().getLeaderForSocket(pkg), offset);
    return reinterpret_cast<T*>(ditem);
  }

  const T* getRemoteByPkg(unsigned int pkg) const {
    void* ditem = b.getRemote(getThreadPool().getLeaderForSocket(pkg), offset);
    return reinterpret_cast<T*>(ditem);
  }

  unsigned size() const { return getThreadPool().getMaxThreads(); }
};

} // namespace substrate
} // end namespace galois
#endif
