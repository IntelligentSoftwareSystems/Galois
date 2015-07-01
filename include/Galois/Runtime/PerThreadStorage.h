/** Per Thread Storage -*- C++ -*-
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

#ifndef GALOIS_RUNTIME_PERTHREADSTORAGE_H
#define GALOIS_RUNTIME_PERTHREADSTORAGE_H

#include "Galois/config.h"
#include "Galois/Substrate/ThreadPool.h"
#include "Galois/Runtime/ll/HWTopo.h"
#include "Galois/Substrate/PaddedLock.h"
#include "Galois/Runtime/ll/TID.h"

#include <cstddef>
#include <boost/utility.hpp>

#include <cassert>
#include <vector>

#include GALOIS_CXX11_STD_HEADER(utility)

namespace Galois {
namespace Runtime {

extern unsigned int activeThreads;

class PerBackend {
  static const unsigned MAX_SIZE = 30;
  // 16 byte alignment so vectorized initialization is easier
  // NB(ddn): llvm seems to assume this under some cases because
  // I've seen weird initialization crashes with MIN_SIZE = 3
  static const unsigned MIN_SIZE = 4;
  typedef Substrate::SimpleLock Lock;

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

  void initCommon();
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

  char* initPerThread();
  char* initPerPackage();

#ifdef GALOIS_USE_EXP
  char* initPerThread_cilk();
  char* initPerPackage_cilk();
#endif // GALOIS_USE_EXP

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

void initPTS();

#ifdef GALOIS_USE_EXP
void initPTS_cilk();
#endif // GALOIS_USE_EXP

template<typename T>
class PerThreadStorage {
protected:
  unsigned offset;
  PerBackend& b;

  void destruct() {
    if (offset == ~0U)
      return;

    for (unsigned n = 0; n < LL::getMaxThreads(); ++n)
      reinterpret_cast<T*>(b.getRemote(n, offset))->~T();
    b.deallocOffset(offset, sizeof(T));
    offset = ~0U;
  }

public:
#if defined(__INTEL_COMPILER) && __INTEL_COMPILER <= 1310
  // ICC 13.1 doesn't detect the other constructor as the default constructor
  PerThreadStorage(): b(getPTSBackend()) {
    //in case we make one of these before initializing the thread pool
    //This will call initPTS for each thread if it hasn't already
    Galois::Runtime::getSystemThreadPool();

    offset = b.allocOffset(sizeof(T));
    for (unsigned n = 0; n < LL::getMaxThreads(); ++n)
      new (b.getRemote(n, offset)) T();
  }
#endif

  template<typename... Args>
  PerThreadStorage(Args&&... args) :b(getPTSBackend()) {
    //in case we make one of these before initializing the thread pool
    //This will call initPTS for each thread if it hasn't already
    Galois::Substrate::getSystemThreadPool();

    offset = b.allocOffset(sizeof(T));
    for (unsigned n = 0; n < LL::getMaxThreads(); ++n)
      new (b.getRemote(n, offset)) T(std::forward<Args>(args)...);
  }

  PerThreadStorage(PerThreadStorage&& o): offset(~0U), b(getPTSBackend()) { 
    std::swap(offset, o.offset);
  }

  PerThreadStorage& operator=(PerThreadStorage&& o) {
    std::swap(offset, o.offset);
    return *this;
  }

  PerThreadStorage(const PerThreadStorage&) = delete;
  PerThreadStorage& operator=(const PerThreadStorage&) = delete;

  ~PerThreadStorage() {
    destruct();
  }

  T* getLocal() {
    void* ditem = b.getLocal(offset, ptsBase);
    return reinterpret_cast<T*>(ditem);
  }

  const T* getLocal() const {
    void* ditem = b.getLocal(offset, ptsBase);
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

  unsigned size() const {
    return LL::getMaxThreads();
  }
};

template<typename T>
class PerPackageStorage {
protected:
  unsigned offset;
  PerBackend& b;

  void destruct() {
    for (unsigned n = 0; n < LL::getMaxPackages(); ++n)
      reinterpret_cast<T*>(b.getRemote(LL::getLeaderForPackage(n), offset))->~T();
    b.deallocOffset(offset, sizeof(T));
  }

public:
#if defined(__INTEL_COMPILER) && __INTEL_COMPILER <= 1310
  // ICC 13.1 doesn't detect the other constructor as the default constructor
  PerPackageStorage(): b(getPPSBackend()) {
    //in case we make one of these before initializing the thread pool
    //This will call initPTS for each thread if it hasn't already
    Galois::Runtime::getSystemThreadPool();

    offset = b.allocOffset(sizeof(T));
    for (unsigned n = 0; n < LL::getMaxPackages(); ++n)
      new (b.getRemote(LL::getLeaderForPackage(n), offset)) T();
  }
#endif
  
  template<typename... Args>
  PerPackageStorage(Args&&... args) :b(getPPSBackend()) {
    //in case we make one of these before initializing the thread pool
    //This will call initPTS for each thread if it hasn't already
    Galois::Substrate::getSystemThreadPool();

    offset = b.allocOffset(sizeof(T));
    for (unsigned n = 0; n < LL::getMaxPackages(); ++n)
      new (b.getRemote(LL::getLeaderForPackage(n), offset)) T(std::forward<Args>(args)...);
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
    void* ditem = b.getRemote(LL::getLeaderForPackage(pkg), offset);
    return reinterpret_cast<T*>(ditem);
  }

  const T* getRemoteByPkg(unsigned int pkg) const {
    void* ditem = b.getRemote(LL::getLeaderForPackage(pkg), offset);
    return reinterpret_cast<T*>(ditem);
  }

  unsigned size() const {
    return LL::getMaxThreads();
  }
};

}
} // end namespace Galois
#endif
