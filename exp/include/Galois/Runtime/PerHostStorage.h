/** Galois Per-Topo Storage -*- C++ -*-
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
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */


#ifndef GALOIS_RUNTIME_PERHOSTSTORAGE
#define GALOIS_RUNTIME_PERHOSTSTORAGE

#include "Galois/Runtime/Serialize.h"
#include "Galois/Runtime/DistSupport.h"

namespace Galois {
namespace Runtime {

using namespace Galois::Runtime::Distributed;

class PerBackend_v2 {
  std::unordered_map<uint64_t, void*> items;
  std::unordered_map<std::pair<uint64_t, uint32_t>, void*,
		     pairhash<uint64_t, uint32_t>> remoteCache;

  uint32_t nextID;
  LL::SimpleLock<true> lock;

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

  template<typename T>
  T* releaseAt(uint64_t off) { return reinterpret_cast<T*>(releaseAt_i(off)); }

  template<typename T>
  T* resolve(uint64_t off ) { return reinterpret_cast<T*>(resolve_i(off)); }

  //returns pointer in remote address space
  template<typename T>
  gptr<T> resolveRemote(uint64_t off, uint32_t hostID) {
    return gptr<T>(hostID, reinterpret_cast<T*>(resolveRemote_i(off, hostID)));
  }
};

PerBackend_v2& getPerHostBackend();

template<typename T>
class PerHost {

  //global name
  uint64_t offset;

  //cached pair
  mutable uint32_t localHost;
  mutable T* localPtr;

  T* resolve() const {
    if (localHost != networkHostID || !localPtr) {
      localHost = networkHostID;
      localPtr = getPerHostBackend().resolve<T>(offset);
    }
    return localPtr;  
  }

  explicit PerHost(uint64_t off) :offset(off), localHost(~0), localPtr(nullptr) {}

  static void allocOnHost(DeSerializeBuffer& buf) {
    uint64_t off;
    gDeserialize(buf, off);
    getPerHostBackend().createAt(off, new T(PerHost(off), buf));
  }

  static void deallocOnHost(uint64_t off) {
    delete getPerHostBackend().releaseAt<T>(off);
  }

public:
  //create a pointer
  static PerHost allocate() {
    uint64_t off = getPerHostBackend().allocateOffset();
    getPerHostBackend().createAt(off, new T(PerHost(off)));
    PerHost ptr(off);
    SerializeBuffer buf;
    gSerialize(buf, off);
    ptr->getInitData(buf);
    getSystemNetworkInterface().broadcast(&allocOnHost, buf);
    return ptr;
  }
  static void deallocate(PerHost ptr) {
    getPerHostBackend().deallocateOffset(ptr.offset);
    getSystemNetworkInterface().broadcastAlt(&deallocOnHost, ptr.offset);
    deallocOnHost(ptr.offset);
  }

  PerHost() : offset(0), localHost(~0), localPtr(nullptr) {}

  gptr<T> remote(uint32_t hostID) {
    return getPerHostBackend().resolveRemote<T>(offset, hostID);
  }

  gptr<T> local() {
    return remote(Distributed::networkHostID);
  }

  T& operator*() const { return *resolve(); }
  T* operator->() const { return resolve(); }

  bool operator<(const PerHost& rhs)  const { return offset <  rhs.offset; }
  bool operator>(const PerHost& rhs)  const { return offset >  rhs.offset; }
  bool operator==(const PerHost& rhs) const { return offset == rhs.offset; }
  bool operator!=(const PerHost& rhs) const { return offset != rhs.offset; }
  explicit operator bool() const { return offset != 0; }

  //serialize
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    gSerialize(s,offset);
  }
  void deserialize(Galois::Runtime::Distributed::DeSerializeBuffer& s) {
    gDeserialize(s,offset);
    localHost = ~0;
    localPtr = nullptr;
  }
};

namespace hidden {
using namespace Galois::Runtime::Distributed;

template<typename T>
struct Deallocate {
  gptr<T> p;
  T* rp;
  
  Deallocate(gptr<T> p): p(p), rp(0) { }
  Deallocate() { }

  void operator()(unsigned tid, unsigned) {
    if (tid == 0 && rp) {
      delete rp;
    }
  }

  typedef int tt_has_serialize;
  void serialize(SerializeBuffer& s) const { gSerialize(s, p); }
  void deserialize(DeSerializeBuffer& s) { gDeserialize(s, p); rp = &*p; }
};

template<typename T>
struct AllocateEntry {
  gptr<T> p;
  T* rp;
  
  AllocateEntry(gptr<T> p): p(p), rp(0) { }
  AllocateEntry() { }

  void operator()(unsigned tid, unsigned) {
    if (tid == 0) {
      rp = &*p; // Fault in persistent object
    }
  }

  typedef int tt_has_serialize;
  void serialize(SerializeBuffer& s) const { gSerialize(s, p); }
  void deserialize(DeSerializeBuffer& s) { gDeserialize(s, p); }
};

} // end namespace

template<typename T>
void allocatePerHost(T* orig) {
  //  Galois::on_each(hidden::AllocateEntry<T>(gptr<T>(orig)));
}

/**
 * Deallocate per host copies, but master still has to deallocate original (if necessary).
 */
template<typename T>
void deallocatePerHost(gptr<T> p) {
  //Galois::on_each(hidden::Deallocate<T>(p));
}

template<typename T>
class PerHostStorage : public T {

public:
  typedef int tt_has_serialize;
  typedef int tt_is_persistent;

  PerHostStorage() { 
    //allocatePerHost(this);
  }
  PerHostStorage(DeSerializeBuffer& s) { }

  void serialize(SerializeBuffer& s) const { }
  void deserialize(DeSerializeBuffer& s) { }
};

} // end namespace
} // end namespace

#endif
