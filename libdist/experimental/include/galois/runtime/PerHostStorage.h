/**
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of XYZ License (a
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

#ifndef GALOIS_RUNTIME_PERHOSTSTORAGE
#define GALOIS_RUNTIME_PERHOSTSTORAGE

#include "galois/runtime/Serialize.h"
#include "galois/runtime/DistSupport.h"
#include "galois/runtime/ThreadPool.h"
#include <boost/iterator/iterator_facade.hpp>

#include <iostream>
#include <typeinfo>

namespace galois {
namespace runtime {

class PerBackend_v2 {
  std::unordered_map<uint64_t, void*> items;
  std::unordered_map<std::pair<uint64_t, uint32_t>, void*,
                     boost::hash<std::pair<uint64_t, uint32_t>>>
      remoteCache;

  uint32_t nextID;
  LL::SimpleLock lock;

  void* releaseAt_i(uint64_t);
  void* resolve_i(uint64_t);
  void* resolveRemote_i(uint64_t, uint32_t);
  void addRemote(void* ptr, uint32_t srcID, uint64_t off);

  static void pBe2ResolveLP(void* ptr, uint32_t srcID, uint64_t off);
  static void pBe2Resolve(uint32_t dest, uint64_t off);

public:
  PerBackend_v2();

  uint64_t allocateOffset();
  void deallocateOffset(uint64_t);

  void createAt(uint64_t, void*);

  template <typename T>
  T* releaseAt(uint64_t off) {
    return reinterpret_cast<T*>(releaseAt_i(off));
  }

  template <typename T>
  T* resolve(uint64_t off) {
    return reinterpret_cast<T*>(resolve_i(off));
  }

  // returns pointer in remote address space
  template <typename T>
  gptr<T> resolveRemote(uint64_t off, uint32_t hostID) {
    return gptr<T>(hostID, reinterpret_cast<T*>(resolveRemote_i(off, hostID)));
  }
};

PerBackend_v2& getPerHostBackend();

template <typename T>
class PerHost {
  // global name
  uint64_t offset;

  // cached pair
  mutable uint32_t localHost;
  mutable T* localPtr;

  T* resolve() const {
    if (localHost != NetworkInterface::ID || !localPtr) {
      localHost = NetworkInterface::ID;
      localPtr  = getPerHostBackend().resolve<T>(offset);
    }
    return localPtr;
  }

  explicit PerHost(uint64_t off)
      : offset(off), localHost(~0), localPtr(nullptr) {}

  template <typename... Args>
  static void allocOnHost(uint64_t off, Args... args) {
    getPerHostBackend().createAt(off, new T(PerHost(off), args...));
  }

  static void deallocOnHost(uint64_t off) {
    delete getPerHostBackend().releaseAt<T>(off);
  }

public:
  // create a pointer
  template <typename... Args>
  static PerHost allocate(Args... args) {
    uint64_t off = getPerHostBackend().allocateOffset();
    getPerHostBackend().createAt(off, new T(PerHost(off), args...));
    // broadcast may be out of order with other messages
    // getSystemNetworkInterface().broadcastAlt(&allocOnHost<Args...>, off,
    // args...);
    auto& net = getSystemNetworkInterface();
    for (unsigned z = 0; z < NetworkInterface::Num; ++z)
      if (z != NetworkInterface::ID)
        net.sendAlt(z, &allocOnHost<Args...>, off, args...);
    return PerHost(off);
  }
  static void deallocate(PerHost ptr) {
    assert(NetworkInterface::ID == 0);
    getSystemNetworkInterface().broadcastAlt(&deallocOnHost, ptr.offset);
    deallocOnHost(ptr.offset);
    getPerHostBackend().deallocateOffset(ptr.offset);
  }

  PerHost() : offset(0), localHost(~0), localPtr(nullptr) {}

  gptr<T> remote(uint32_t hostID) {
    return getPerHostBackend().resolveRemote<T>(offset, hostID);
  }

  gptr<T> local() { return remote(NetworkInterface::ID); }

  T& operator*() const { return *resolve(); }
  T* operator->() const { return resolve(); }

  bool operator<(const PerHost& rhs) const { return offset < rhs.offset; }
  bool operator>(const PerHost& rhs) const { return offset > rhs.offset; }
  bool operator==(const PerHost& rhs) const { return offset == rhs.offset; }
  bool operator!=(const PerHost& rhs) const { return offset != rhs.offset; }
  explicit operator bool() const { return offset != 0; }

  // serialize
  typedef int tt_has_serialize;
  void serialize(galois::runtime::SerializeBuffer& s) const {
    gSerialize(s, offset);
  }
  void deserialize(galois::runtime::DeSerializeBuffer& s) {
    gDeserialize(s, offset);
    localHost = ~0;
    localPtr  = nullptr;
  }
};

class PerBackend_v3 {
  static const int dynSlots = 1024;
  static __thread void* space[dynSlots];

  std::vector<bool> freelist;
  std::vector<void**> heads;
  std::map<std::tuple<uint64_t, uint32_t, uint32_t>, void*> remoteCache;
  LL::SimpleLock lock;

  void* resolveRemote_i(uint64_t, uint32_t, uint32_t);
  void addRemote(void* ptr, uint32_t srcID, uint64_t off, uint32_t threadID);

  static void pBe2ResolveLP(void* ptr, uint32_t srcID, uint64_t off,
                            uint32_t threadID);
  static void pBe2Resolve(uint32_t dest, uint64_t off, uint32_t threadID);

public:
  PerBackend_v3();

  void initThread();

  uint64_t allocateOffset();
  void deallocateOffset(uint64_t);

  template <typename T>
  T*& resolve(uint64_t off) {
    return *reinterpret_cast<T**>(&space[off]);
  }

  template <typename T>
  T*& resolveThread(uint64_t off, uint32_t tid) {
    return *reinterpret_cast<T**>(&heads.at(tid)[off]);
  }

  // returns pointer in remote address space
  template <typename T>
  gptr<T> resolveRemote(uint64_t off, uint32_t hostID, uint32_t threadID) {
    return gptr<T>(
        hostID, reinterpret_cast<T*>(resolveRemote_i(off, hostID, threadID)));
  }
};

PerBackend_v3& getPerThreadDistBackend();

template <typename T>
class PerThreadDist {
  // global name
  uint64_t offset;

  T* resolve() const {
    assert((offset < (uint64_t)1024) && (offset > 0));
    T* r = getPerThreadDistBackend().resolve<T>(offset);
    assert(r);
    return r;
  }

  explicit PerThreadDist(uint64_t off) : offset(off) {}

  static void allocOnHost(DeSerializeBuffer& buf) {
    uint64_t off;
    gDeserialize(buf, off);
    assert(off < 1024ul);
    for (unsigned x = 0; x < getSystemThreadPool().getMaxThreads(); ++x) {
      if (!getPerThreadDistBackend().resolveThread<T>(off, x)) {
        // DeSerializeBuffer buf2((char*)buf.linearData(),
        // (char*)buf.linearData() + buf.size()); //explicit copy
        auto o = buf.getOffset();
        getPerThreadDistBackend().resolveThread<T>(off, x) =
            new T(PerThreadDist(off), buf);
        buf.setOffset(o);
      }
    }
  }

  static void deallocOnHost(uint64_t off) {
    for (unsigned x = 0; x < getSystemThreadPool().getMaxThreads(); ++x) {
      T*& p = getPerThreadDistBackend().resolveThread<T>(off, x);
      // Invalidate any gptrs we may have generated
      getLocalDirectory().invalidate(
          static_cast<runtime::fatPointer>(gptr<T>(p)));
      delete p;
      p = nullptr;
    }
  }

public:
  // create a pointer
  static PerThreadDist allocate() {
    uint64_t off = getPerThreadDistBackend().allocateOffset();
    getPerThreadDistBackend().resolve<T>(off) = new T(PerThreadDist(off));
    PerThreadDist ptr(off);
    SerializeBuffer buf;
    gSerialize(buf, off);
    ptr->getInitData(buf);
    // std::cout << "Serialize buf :" << buf << "\n";
    SerializeBuffer buf2((char*)buf.linearData(), buf.size()); // explicit copy
    // std::cout <<"Serialize buf2 : " << buf2 << "\n";
    getSystemNetworkInterface().broadcast(&allocOnHost, buf2);
    DeSerializeBuffer dbuf(std::move(buf2));
    allocOnHost(dbuf);
    return ptr;
  }

  static void deallocate(PerThreadDist ptr) {
    getSystemNetworkInterface().broadcastAlt(&deallocOnHost, ptr.offset);
    deallocOnHost(ptr.offset);
    getPerHostBackend().deallocateOffset(ptr.offset);
  }

  PerThreadDist() : offset(~0) {}

  gptr<T> remote(uint32_t hostID, unsigned threadID) const {
    if (hostID == NetworkInterface::ID)
      return gptr<T>(
          getPerThreadDistBackend().resolveThread<T>(offset, threadID));
    else
      return getPerThreadDistBackend().resolveRemote<T>(offset, hostID,
                                                        threadID);
  }

  gptr<T> local() const { return gptr<T>(NetworkInterface::ID, resolve()); }

  T& operator*() const { return *resolve(); }
  T* operator->() const { return resolve(); }

  bool operator<(const PerThreadDist& rhs) const { return offset < rhs.offset; }
  bool operator>(const PerThreadDist& rhs) const { return offset > rhs.offset; }
  bool operator==(const PerThreadDist& rhs) const {
    return offset == rhs.offset;
  }
  bool operator!=(const PerThreadDist& rhs) const {
    return offset != rhs.offset;
  }
  explicit operator bool() const { return offset != 0; }

  class iterator
      : public boost::iterator_facade<iterator, gptr<T>,
                                      std::forward_iterator_tag, gptr<T>> {
    friend class boost::iterator_core_access;
    friend class PerThreadDist;
    uint32_t hostID;
    uint32_t threadID;
    PerThreadDist basePtr;

    gptr<T> dereference() const { return basePtr.remote(hostID, threadID); }
    bool equal(const iterator& rhs) const {
      return hostID == rhs.hostID && threadID == rhs.threadID &&
             basePtr == rhs.basePtr;
    }
    void increment() {
      if (threadID < activeThreads)
        ++threadID;
      if (threadID == activeThreads &&
          hostID < NetworkInterface::Num) { // FIXME: maxthreads on hostID
        ++hostID;
        threadID = 0;
      }
      if (hostID == NetworkInterface::Num) {
        threadID = activeThreads;
        basePtr  = PerThreadDist();
      }
    }
    iterator(uint32_t h, uint32_t t, PerThreadDist p)
        : hostID(h), threadID(t), basePtr(p) {}

  public:
    iterator()
        : hostID(NetworkInterface::Num), threadID(activeThreads), basePtr() {}
  };

  iterator begin() { return iterator(0, 0, *this); }
  iterator end() { return iterator(); }

  // serialize
  typedef int tt_has_serialize;
  void serialize(galois::runtime::SerializeBuffer& s) const {
    gSerialize(s, offset);
  }
  void deserialize(galois::runtime::DeSerializeBuffer& s) {
    gDeserialize(s, offset);
  }

  int getOffset() { return offset; }
};

} // namespace runtime
} // namespace galois

#endif
