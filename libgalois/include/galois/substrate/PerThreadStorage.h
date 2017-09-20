/** Per Thread Storage -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * Dynamic per-thread storage (dPTS).
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#ifndef GALOIS_SUBSTRATE_PERTHREADSTORAGE_H
#define GALOIS_SUBSTRATE_PERTHREADSTORAGE_H

#include "galois/substrate/ThreadPool.h"
#include "galois/substrate/HWTopo.h"
#include "galois/substrate/PaddedLock.h"

#include <cstddef>
//#include <boost/utility.hpp>

#include <cassert>
#include <vector>
#include <utility>

namespace galois {
namespace substrate {

class PerBackend {
  static const unsigned MAX_SIZE = 30;
  // 16 byte alignment so vectorized initialization is easier
  // NB(ddn): llvm seems to assume this under some cases because
  // I've seen weird initialization crashes with MIN_SIZE = 3
  static const unsigned MIN_SIZE = 4;
  typedef substrate::SimpleLock Lock;

  unsigned int nextLoc;
  char** heads;
  Lock freeOffsetsLock;
  std::vector<std::vector<unsigned> > freeOffsets;
  /**
   * Guards access to non-POD objects that can be accessed after PerBackend
   * is destroyed. Access can occur through destroying PerThread/PerPackage
   * objects with static storage duration, which have a reference to a
   * PerBackend object, which may have be destroyed before the PerThread
   * object itself.
   */
  bool invalid; 

  void initCommon(unsigned maxT);
  static unsigned nextLog2(unsigned size);

public:
  PerBackend(): nextLoc(0), heads(0), invalid(false) {
    freeOffsets.resize(MAX_SIZE);
  }

  PerBackend(const PerBackend&) = delete;
  PerBackend& operator=(const PerBackend&) = delete;

  ~PerBackend() {
    // Intentionally leak heads so that other PerThread operations are
    // still valid after we are gone
    invalid = true;
  }

  char* initPerThread(unsigned maxT);
  char* initPerPackage(unsigned maxT);

  unsigned allocOffset(const unsigned size);
  void deallocOffset(const unsigned offset, const unsigned size);
  void* getRemote(unsigned thread, unsigned offset);
  void* getLocal(unsigned offset, char* base) {
    return &base[offset];
  }
  // faster when (1) you already know the id and (2) shared access to heads is
  // not to expensive; otherwise use getLocal(unsigned,char*)
  void* getLocal(unsigned offset, unsigned id) {
    return &heads[id][offset];
  }
};

extern __thread char* ptsBase;
PerBackend& getPTSBackend();

extern __thread char* ppsBase;
PerBackend& getPPSBackend();

void initPTS(unsigned maxT);

template<typename T>
class PerThreadStorage {
protected:
  unsigned offset;
  PerBackend* b;

  void destruct() {
    if (offset == ~0U)
      return;
    
    for (unsigned n = 0; n < getThreadPool().getMaxThreads(); ++n)
      reinterpret_cast<T*>(b->getRemote(n, offset))->~T();
    b->deallocOffset(offset, sizeof(T));
    offset = ~0U;
  }


public:
  //construct on each thread
  template<typename... Args>
  PerThreadStorage(Args&&... args) :b(&getPTSBackend()) {
    //in case we make one of these before initializing the thread pool
    //This will call initPTS for each thread if it hasn't already
    auto& tp = getThreadPool();

    offset = b->allocOffset(sizeof(T));
    for (unsigned n = 0; n < tp.getMaxThreads(); ++n)
      new (b->getRemote(n, offset)) T(std::forward<Args>(args)...);
  }

  PerThreadStorage(PerThreadStorage&& rhs) :b(rhs.b), offset(rhs.offset) {
    rhs.offset = ~0;
  }

  ~PerThreadStorage() {
    destruct();
  }

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

  unsigned size() const {
    return getThreadPool().getMaxThreads();
  }
};




template<typename T>
class PerPackageStorage {
protected:
  unsigned offset;
  PerBackend& b;

  void destruct() {
    auto& tp = getThreadPool();
    for (unsigned n = 0; n < tp.getMaxPackages(); ++n)
      reinterpret_cast<T*>(b.getRemote(tp.getLeaderForPackage(n), offset))->~T();
    b.deallocOffset(offset, sizeof(T));
  }

public:
  
  template<typename... Args>
  PerPackageStorage(Args&&... args) :b(getPPSBackend()) {
    //in case we make one of these before initializing the thread pool
    //This will call initPTS for each thread if it hasn't already
    getThreadPool();

    offset = b.allocOffset(sizeof(T));
    auto& tp = getThreadPool();
    for (unsigned n = 0; n < tp.getMaxPackages(); ++n)
      new (b.getRemote(tp.getLeaderForPackage(n), offset)) T(std::forward<Args>(args)...);
  }

  PerPackageStorage(PerPackageStorage&& o): offset(std::move(o.offset)), b(getPPSBackend()) { }
  PerPackageStorage& operator=(PerPackageStorage&& o) {
    destruct();
    offset = std::move(o.offset);
    return *this;
  }

  PerPackageStorage(const PerPackageStorage&) = delete;
  PerPackageStorage& operator=(const PerPackageStorage&) = delete;

  ~PerPackageStorage() {
    destruct();
  }

  T* getLocal() {
    void* ditem = b.getLocal(offset, ppsBase);
    return reinterpret_cast<T*>(ditem);
  }

  const T* getLocal() const {
    void* ditem = b.getLocal(offset, ppsBase);
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
    void* ditem = b.getRemote(getThreadPool().getLeaderForPackage(pkg), offset);
    return reinterpret_cast<T*>(ditem);
  }

  const T* getRemoteByPkg(unsigned int pkg) const {
    void* ditem = b.getRemote(getThreadPool().getLeaderForPackage(pkg), offset);
    return reinterpret_cast<T*>(ditem);
  }

  unsigned size() const {
    return getThreadPool().getMaxThreads();
  }
};

}
} // end namespace galois
#endif
