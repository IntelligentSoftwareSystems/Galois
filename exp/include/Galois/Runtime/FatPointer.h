/** Galois Fat Pointer -*- C++ -*-
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
#ifndef GALOIS_RUNTIME_FATPOINTER_H
#define GALOIS_RUNTIME_FATPOINTER_H

//std::hash is mixed class and struct in gcc's standard library
#pragma clang diagnostic ignored "-Wmismatched-tags"

#include <boost/functional/hash.hpp>
#include "Galois/Runtime/Network.h"

#include <ostream>

namespace Galois {
namespace Runtime {

namespace detail {

//Really fat pointer
class simpleFatPointer {
  uint32_t host;
  void* ptr;

  void set(uint32_t h, void* p) {
    host = h;
    ptr = p;
  }

protected:

  constexpr simpleFatPointer() noexcept :host(0), ptr(nullptr) {}

  typedef std::pair<uint32_t, void*> rawType;
  rawType rawCopy() const { return std::make_pair(host,ptr); }

public:

  uint32_t getHost() const { return host; }
  void setHost(uint32_t h) { host = h; }
  void* getObj() const { return ptr; }
  void setObj(void* p) { ptr = p; }
};

//Fat pointer taking advantage of unused bits in AMD64 addresses
class amd64FatPointer {
  uintptr_t val;

  uintptr_t compute(uint32_t h, void* p) const;
  void set(uint32_t h, void* p);

protected:

  amd64FatPointer(uint32_t h, void* p) noexcept :val(compute(h,p)) {}

  typedef uintptr_t rawType;
  rawType rawCopy() const { return val; }

public:
  uint32_t getHost() const {
    uintptr_t hval = val;
    hval >>= 47;
    hval &= 0x0000FFFF;
    return hval;
  }
  void* getObj() const {
    uintptr_t hval = val;
    hval &= 0x80007FFFFFFFFFFF;
    return (void*)hval;
  }
  void setHost(uint32_t h);
  void setObj(void* p);
};


template< typename fatPointerBase>
class fatPointerImpl : public fatPointerBase {
  using fatPointerBase::rawCopy;

public:

  struct thisHost {};

  constexpr fatPointerImpl() noexcept :fatPointerBase(0, nullptr) {}
  constexpr fatPointerImpl(uint32_t h, void* p) noexcept :fatPointerBase(h,p) {}
  fatPointerImpl(void* p, thisHost th) noexcept : fatPointerBase(NetworkInterface::ID,p) {}

  size_t hash_value() const {
    boost::hash<typename fatPointerBase::rawType> ih;
    return ih(rawCopy());
  }

  inline explicit operator bool() const { return fatPointerBase::getObj(); }

  inline bool operator==(const fatPointerImpl& rhs) const { return rawCopy() == rhs.rawCopy(); }
  inline bool operator!=(const fatPointerImpl& rhs) const { return rawCopy() != rhs.rawCopy(); }
  inline bool operator<=(const fatPointerImpl& rhs) const { return rawCopy() <= rhs.rawCopy(); }
  inline bool operator>=(const fatPointerImpl& rhs) const { return rawCopy() >= rhs.rawCopy(); }
  inline bool operator< (const fatPointerImpl& rhs) const { return rawCopy() <  rhs.rawCopy(); }
  inline bool operator> (const fatPointerImpl& rhs) const { return rawCopy() >  rhs.rawCopy(); }

  inline bool isLocal() const {
    return fatPointerBase::getHost() == NetworkInterface::ID;
  }
};

template<typename T>
size_t hash_value(const fatPointerImpl<T>& v) {
  return v.hash_value();
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const fatPointerImpl<T>& v) {
  return os <<  "[" << v.getHost() << "," << v.getObj() << "]";
}


} // namespace detail

typedef detail::fatPointerImpl<detail::amd64FatPointer> fatPointer;

} // namespace Runtime
} // namespace Galois

namespace std {
template<>
struct hash<Galois::Runtime::fatPointer> {
  size_t operator()(const Galois::Runtime::fatPointer& ptr) const {
    return ptr.hash_value();
  }
};
} // namespace std


#endif
