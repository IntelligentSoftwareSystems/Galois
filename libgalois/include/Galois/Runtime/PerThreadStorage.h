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
#include "Galois/Runtime/ErrorFeedBack.h"

#include <boost/core/noncopyable.hpp>

#include <cstddef>
#include <cassert>
#include <vector>
#include <utility>
#include <bitset>

namespace Galois {
namespace Runtime {

namespace detail {

class PerBackend : private boost::noncopyable{
public:
  static constexpr unsigned  PTSSize = 4*1024*1024;
private:
  std::vector<std::unique_ptr<uint64_t[]> > heads;
  std::bitset<PTSSize> mask;
  SimpleLock lock;
public:
  //Fixme: Do we need 16byte alignment (OSX, llvm)?
  unsigned alloc(unsigned bytes);
  void dealloc(unsigned offset, unsigned bytes);

  uint64_t* get(unsigned n) { return heads[n].get(); }
  void set(unsigned n, uint64_t* ptr);
};


class PerThreadStorageBase {
  static thread_local uint64_t* storage;
  static PerBackend perBackend;

  static void init_inner(unsigned max);

  friend void Galois::Runtime::initPTS(unsigned);

protected:
  PerThreadStorageBase() : offset(~0) {
    ThreadPool::getThreadPool();
    if (!storage)
      gDie("Uninitialized PTS");
  }

  PerThreadStorageBase(PerThreadStorageBase&& rhs) : offset(rhs.offset) {
    rhs.offset = ~0;
  }

  //per-object interface
  unsigned offset;  
  
  void* _getLocal() const {
    return &storage[offset];
  }

  void* _getRemote(unsigned id) const {
    return &perBackend.get(id)[offset];
  }

  explicit operator bool() const {
    return offset != ~0U;
  }

  unsigned _alloc(unsigned sT) { return perBackend.alloc(sT); }

  void _dealloc(unsigned offset, unsigned bytes) { perBackend.dealloc(offset, bytes); }

};
} // namespace PerThreadStorageBase



template<typename T>
class PerThreadStorage : public detail::PerThreadStorageBase {

public:
  typedef T* pointer;
  typedef T element_type;

  //construct on each thread
  template<typename... Args>
  PerThreadStorage(Args&&... args) : PerThreadStorageBase() {
    auto& tp = ThreadPool::getThreadPool();
    offset = _alloc(sizeof(T));
    for (unsigned n = 0; n < tp.getMaxThreads(); ++n)
      new (_getRemote(n)) T(std::forward<Args>(args)...);
  }

  PerThreadStorage(PerThreadStorage&& rhs) :PerThreadStorageBase(std::forward(rhs)) {}

  ~PerThreadStorage() {
    for (unsigned n = 0; n < ThreadPool::getThreadPool().getMaxThreads(); ++n)
      get(n)->~T();
    _dealloc(offset, sizeof(T));
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
    offset = _alloc(sizeof(T));
    for (unsigned n = 0; n < tp.getMaxPackages(); ++n)
      new (_getRemote(tp.getLeaderForPackage(n))) T(std::forward<Args>(args)...);
  }

  PerPackageStorage(PerPackageStorage&& rhs) :PerThreadStorageBase(std::forward(rhs)) {}

  ~PerPackageStorage() {
    auto& tp = ThreadPool::getThreadPool();
    for (unsigned n = 0; n < tp.getMaxPackages(); ++n)
      get(tp.getLeaderForPackage(n))->~T();
    _dealloc(offset, sizeof(T));
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
