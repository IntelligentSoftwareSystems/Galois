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

#include <cassert>
#include <vector>

#include "ll/TID.h"
#include "ll/HWTopo.h"
#include "ThreadPool.h"

#include <boost/utility.hpp>

namespace GaloisRuntime {

class PerBackend {
  unsigned int nextLoc;
  std::vector<char*> heads;

  void initCommon();

public:
  PerBackend(): nextLoc(0) { }

  char* initPerThread();
  char* initPerPackage();
  unsigned allocOffset(unsigned size);
  void deallocOffset(unsigned offset, unsigned size);
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
extern PerBackend PTSBackend;

extern __thread char* ppsBase;
extern PerBackend PPSBackend;

void initPTS();

template<typename T>
class PerThreadStorage: private boost::noncopyable {
protected:
  unsigned offset;

public:
  PerThreadStorage() {
    //in case we make one of these before initializing the thread pool
    //This will call initPTS for each thread if it hasn't already
    GaloisRuntime::getSystemThreadPool();

    offset = PTSBackend.allocOffset(sizeof(T));
    for (unsigned n = 0; n < LL::getMaxThreads(); ++n)
      new (PTSBackend.getRemote(n, offset)) T();
  }

  ~PerThreadStorage() {
    for (unsigned n = 0; n < LL::getMaxThreads(); ++n)
      reinterpret_cast<T*>(PTSBackend.getRemote(n, offset))->~T();
    PTSBackend.deallocOffset(offset, sizeof(T));
  }

  T* getLocal() const {
    void* ditem = PTSBackend.getLocal(offset, ptsBase);
    return reinterpret_cast<T*>(ditem);
  }

  //! Like getLocal() but optimized for when you already know the thread id
  T* getLocal(unsigned int thread) const {
    void* ditem = PTSBackend.getLocal(offset, thread);
    return reinterpret_cast<T*>(ditem);
  }

  T* getRemote(unsigned int thread) const {
    void* ditem = PTSBackend.getRemote(thread, offset);
    return reinterpret_cast<T*>(ditem);
  }

  unsigned size() const {
    return LL::getMaxThreads();
  }
};

template<typename T>
class PerPackageStorage: private boost::noncopyable {
protected:
  unsigned offset;

public:
  PerPackageStorage() {
    //in case we make one of these before initializing the thread pool
    //This will call initPTS for each thread if it hasn't already
    GaloisRuntime::getSystemThreadPool();

    offset = PPSBackend.allocOffset(sizeof(T));
    for (unsigned n = 0; n < LL::getMaxPackages(); ++n)
      new (PPSBackend.getRemote(LL::getLeaderForPackage(n), offset)) T();
  }

  ~PerPackageStorage() {
    for (unsigned n = 0; n < LL::getMaxPackages(); ++n)
      reinterpret_cast<T*>(PPSBackend.getRemote(LL::getLeaderForPackage(n), offset))->~T();
    PPSBackend.deallocOffset(offset, sizeof(T));
  }

  T* getLocal() const {
    void* ditem = PPSBackend.getLocal(offset, ppsBase);
    return reinterpret_cast<T*>(ditem);
  }

  //! Like getLocal() but optimized for when you already know the thread id
  T* getLocal(unsigned int thread) const {
    void* ditem = PPSBackend.getLocal(offset, thread);
    return reinterpret_cast<T*>(ditem);
  }

  T* getRemote(unsigned int thread) const {
    void* ditem = PPSBackend.getRemote(thread, offset);
    return reinterpret_cast<T*>(ditem);
  }

  T* getRemoteByPkg(unsigned int pkg) const {
    void* ditem = PPSBackend.getRemote(LL::getLeaderForPackage(pkg), offset);
    return reinterpret_cast<T*>(ditem);
  }

  unsigned size() const {
    return LL::getMaxThreads();
  }
};

}

#endif
