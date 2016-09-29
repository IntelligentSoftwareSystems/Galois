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
 * Copyright (C) 2016, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * Dynamic per-thread storage (dPTS).
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#ifndef GALOIS_RUNTIME_PERTHREADSTORAGE_H
#define GALOIS_RUNTIME_PERTHREADSTORAGE_H

#include "Galois/Runtime/ThreadPool.h"
#include "Galois/Runtime/HWTopo.h"
#include "Galois/Runtime/SimpleLock.h"

#include <cstddef>

#include <cassert>
#include <vector>
#include <utility>
#include <bitset>

namespace Galois {
namespace Runtime {

namespace detail {

class PerThreadStorageBase {
  static constexpr unsigned PTSSize = 4*1024*1024;
  static thread_local std::unique_ptr<uint64_t[]> storage;
  static std::vector<uint64_t*> heads;
  static std::bitset<PTSSize> mask;
  static std::atomic<bool> initialized;
  static SimpleLock lock;

  void init_inner();
  void init(ThreadPool& tp);

protected:

  PerThreadStorageBase() : offset(~0) {
    if (!initialized) {
      //in case we make one of these before initializing the thread pool
      //This will call initPTS for each thread if it hasn't already
      auto& tp = ThreadPool::getThreadPool();
    
      init(tp);
    }
  }
  PerThreadStorageBase(PerThreadStorageBase&& rhs) : offset(rhs.offset) {
    rhs.offset = ~0;
  }

  //Fixme: Do we need 16byte alignment (OSX, llvm)?
  static unsigned alloc(unsigned bytes);
  static void dealloc(unsigned offset, unsigned bytes);

  //per-object interface
  unsigned offset;  
  
  void* _getLocal() const {
    return &storage[offset];
  }

  void* _getRemote(unsigned id) const {
    return &heads[id][offset];
  }

  explicit operator bool() const {
    return offset != ~0U;
  }

};
} // namespace PerThreadStorageBase



template<typename T>
class PerThreadStorage : public detail::PerThreadStorageBase {

public:
  typedef T* pointer;
  typedef T element_type;

  //construct on each thread
  template<typename... Args>
  PerThreadStorage(Args&&... args) {
    auto& tp = ThreadPool::getThreadPool();
    offset = alloc(sizeof(T));
    for (unsigned n = 0; n < tp.getMaxThreads(); ++n)
      new (_getRemote(n)) T(std::forward<Args>(args)...);
  }

  PerThreadStorage(PerThreadStorage&& rhs) :PerThreadStorageBase(std::forward(rhs)) {}

  ~PerThreadStorage() {
    for (unsigned n = 0; n < ThreadPool::getThreadPool().getMaxThreads(); ++n)
      get(n)->~T();
    dealloc(offset, sizeof(T));
    offset = ~0U;
  }

  //Standard smart pointer idioms

  pointer get() const noexcept {
    return reinterpret_cast<pointer>(_getLocal());
  }

  pointer get(unsigned tid) const noexcept {
    return reinterpret_cast<pointer>(_getRemote(tid));
  }

  void swap(PerThreadStorage& other) noexcept {
    std::swap(offset, other.offset);
  }

  //Smart Pointer functions

  typename std::add_lvalue_reference<T>::type operator*() const {
    return *get();
  }

  pointer operator->() const noexcept {
    return get();
  }


  PerThreadStorage& operator=(PerThreadStorage&& rhs) {
    std::swap(offset, rhs.offset);
    return *this;
  }

  unsigned size() const {
    return ThreadPool::getThreadPool().getMaxThreads();
  }
};



template<typename T>
class PerPackageStorage : public detail::PerThreadStorageBase {

  void* _getRemotePkg(unsigned n) const {
    return _getRemote(ThreadPool::getThreadPool().getLeader(n));
  }
  void* _getLocalPkg() const {
    return _getRemote(ThreadPool::getLeader());
  }

public:
  typedef T* pointer;
  typedef T element_type;

  //construct on each thread
  template<typename... Args>
  PerPackageStorage(Args&&... args) {
    auto& tp = ThreadPool::getThreadPool();
    offset = alloc(sizeof(T));
    for (unsigned n = 0; n < tp.getMaxPackages(); ++n)
      new (_getRemote(tp.getLeaderForPackage(n))) T(std::forward<Args>(args)...);
  }

  PerPackageStorage(PerPackageStorage&& rhs) :PerThreadStorageBase(std::forward(rhs)) {}

  ~PerPackageStorage() {
    auto& tp = ThreadPool::getThreadPool();
    for (unsigned n = 0; n < tp.getMaxPackages(); ++n)
      get(tp.getLeaderForPackage(n))->~T();
    dealloc(offset, sizeof(T));
    offset = ~0U;
  }

  //Standard smart pointer idioms

  pointer get() const noexcept {
    return reinterpret_cast<pointer>(_getLocalPkg());
  }

  pointer get(unsigned tid) const noexcept {
    return reinterpret_cast<pointer>(_getRemotePkg(tid));
  }

  pointer getByPkg(unsigned pid) const noexcept {
    auto& tp = ThreadPool::getThreadPool();
    return reinterpret_cast<pointer>(_getRemotePkg(tp.getLeaderForPackage(pid)));
  }

  void swap(PerPackageStorage& other) noexcept {
    std::swap(offset, other.offset);
  }

  //Smart Pointer functions

  typename std::add_lvalue_reference<T>::type operator*() const {
    return *get();
  }

  pointer operator->() const noexcept {
    return get();
  }

  PerPackageStorage& operator=(PerPackageStorage&& rhs) {
    std::swap(offset, rhs.offset);
    return *this;
  }

  unsigned size() const {
    return ThreadPool::getThreadPool().getMaxThreads();
  }
};

} // end namespace Runtime
} // end namespace Galois
#endif
