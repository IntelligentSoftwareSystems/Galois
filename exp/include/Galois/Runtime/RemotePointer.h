/** Galois Distributed Pointer -*- C++ -*-
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
#ifndef GALOIS_RUNTIME_REMOTEPOINTER_H
#define GALOIS_RUNTIME_REMOTEPOINTER_H

#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/Serialize.h"
#include "Galois/Runtime/Directory.h"

namespace Galois {
namespace Runtime {

typedef std::pair<uint32_t, Lockable*> fatPointer;

template<typename T>
class gptr;

template<typename T>
T* resolve(const gptr<T>& p);

template<typename T>
class gptr {
  fatPointer ptr;

public:
  typedef T element_type;
  
  constexpr gptr() noexcept :ptr({0, nullptr}) {}
  explicit gptr(T* p) noexcept :ptr({NetworkInterface::ID, p}) {}
  gptr(uint32_t o, T* p) :ptr({o, static_cast<Lockable*>(p)}) {}
  
  operator fatPointer() const { return ptr; }

  T& operator*() const {
    return *resolve(*this);
  }
  T* operator->() const {
    return resolve(*this);
  }

  bool operator<(const gptr& rhs) const {
    return ptr < rhs.ptr;
  }
  bool operator>(const gptr& rhs) const {
    return ptr > rhs.ptr;
  }
  bool operator==(const gptr& rhs) const {
    return ptr == rhs.ptr;
  }
  bool operator!=(const gptr& rhs) const {
    return ptr != rhs.ptr;
  }
  explicit operator bool() const { return ptr.second != 0; }

  bool isLocal() const {
    return ptr.first == Galois::Runtime::NetworkInterface::ID;
  }

  bool sameHost(const gptr& rhs) const {
    return ptr.first == rhs.ptr.first;
  }

  void initialize(T* p) {
    ptr.second = p;
    ptr.first = ptr.second ? NetworkInterface::ID : 0;
  }

  //serialize
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::SerializeBuffer& s) const {
    gSerialize(s, ptr);
  }
  void deserialize(Galois::Runtime::DeSerializeBuffer& s) {
    gDeserialize(s, ptr);
  }

  void dump(std::ostream& os) const {
    os << "[" << ptr.first << "," << ptr.second << "]";
  }
};

} //namespace Runtime
} //namespace Galois

#endif //DISTSUPPORT
