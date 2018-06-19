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

#ifndef GALOIS_RUNTIME_FATPOINTER_H
#define GALOIS_RUNTIME_FATPOINTER_H

// std::hash is mixed class and struct in gcc's standard library
#pragma clang diagnostic ignored "-Wmismatched-tags"

#include <boost/functional/hash.hpp>
#include "galois/runtime/Network.h"

#include <ostream>

namespace galois {
namespace runtime {

namespace internal {

// Really fat pointer
class simpleFatPointer {
  uint32_t host;
  uintptr_t ptr;

protected:
  constexpr simpleFatPointer(uint32_t h, uintptr_t p) noexcept
      : host(h), ptr(p) {}

public:
  uint32_t getHost() const { return host; }
  uintptr_t getObj() const { return ptr; }

  void setHost(uint32_t h) { host = h; }
  void setObj(uintptr_t p) { ptr = p; }

  void set(uint32_t h, uintptr_t p) {
    host = h;
    ptr  = p;
  }
  std::pair<uint32_t, uintptr_t> get() const {
    return std::make_pair(host, ptr);
  }
};

// Fat pointer taking advantage of unused bits in AMD64 addresses
class amd64FatPointer {
  uintptr_t val;
  static const uintptr_t ptrMask = 0x80007FFFFFFFFFFFULL;

  constexpr uintptr_t compute(uint32_t h, uintptr_t p) const {
    return ((uintptr_t)h << 47) | (p & ptrMask);
  }

protected:
  constexpr amd64FatPointer(uint32_t h, uintptr_t p) noexcept
      : val(compute(h, p)) {}

public:
  uint32_t getHost() const { return get().first; }
  uintptr_t getObj() const { return get().second; }

  void setHost(uint32_t h) { set(h, get().second); }
  void setObj(uintptr_t p) { set(get().first, p); }

  void set(uint32_t h, uintptr_t p) { val = compute(h, p); }
  std::pair<uint32_t, uintptr_t> get() const {
    uint32_t host = (val >> 47) & 0x0000FFFF;
    uintptr_t ptr = val & ptrMask;
    return std::make_pair(host, ptr);
  }
};

template <typename fatPointerBase>
class fatPointerImpl : public fatPointerBase {
  using fatPointerBase::get;

public:
  using fatPointerBase::getHost;
  using fatPointerBase::getObj;

  struct thisHost_t {};
  // static thisHost_t thisHost;

  constexpr fatPointerImpl() noexcept : fatPointerBase(0, 0) {}
  constexpr fatPointerImpl(uint32_t h, uintptr_t p) noexcept
      : fatPointerBase(h, p) {}
  fatPointerImpl(uintptr_t p, thisHost_t th) noexcept
      : fatPointerBase(NetworkInterface::ID, p) {}

  size_t hash_value() const {
    boost::hash<decltype(get())> ih;
    return ih(get());
  }

  template <typename T>
  T* getPtr() const {
    void* ptr = reinterpret_cast<void*>(getObj());
    return static_cast<T*>(ptr);
  }

  explicit operator bool() const { return fatPointerBase::getObj(); }

  bool operator==(const fatPointerImpl& rhs) const {
    return get() == rhs.get();
  }
  bool operator!=(const fatPointerImpl& rhs) const {
    return get() != rhs.get();
  }
  bool operator<=(const fatPointerImpl& rhs) const {
    return get() <= rhs.get();
  }
  bool operator>=(const fatPointerImpl& rhs) const {
    return get() >= rhs.get();
  }
  bool operator<(const fatPointerImpl& rhs) const { return get() < rhs.get(); }
  bool operator>(const fatPointerImpl& rhs) const { return get() > rhs.get(); }

  bool isLocal() const { return getHost() == NetworkInterface::ID; }

  friend std::ostream& operator<<(std::ostream& os, const fatPointerImpl& v) {
    return os << "[" << v.getHost() << "," << v.getObj() << "]";
  }

  fatPointerImpl arith(int n) {
    return fatPointerImpl(getHost(), getObj() + n);
  }

  // Trivially_copyable
  typedef int tt_is_copyable;
};

template <typename T>
size_t hash_value(const fatPointerImpl<T>& v) {
  return v.hash_value();
}

} // namespace internal

typedef internal::fatPointerImpl<internal::amd64FatPointer> fatPointer;
// typedef internal::fatPointerImpl<internal::simpleFatPointer> fatPointer;

} // namespace runtime
} // namespace galois

namespace std {
template <>
struct hash<galois::runtime::fatPointer> {
  size_t operator()(const galois::runtime::fatPointer& ptr) const {
    return ptr.hash_value();
  }
};
} // namespace std

#endif
