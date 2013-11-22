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

#include  <boost/functional/hash.hpp>

#define LONGPTR 0

namespace Galois {
namespace Runtime {

//forward declaration
class Lockable;

#if LONGPTR

//Really fat pointer
class fatPointer {
  uint32_t host;
  Lockable* ptr;

public:
  constexpr fatPointer() noexcept :host(0), ptr(nullptr) {}
  fatPointer(uint32_t h, Lockable* p) noexcept :host(h), ptr(p) {}
  uint32_t getHost() const { return host; }
  void setHost(uint32_t h) { host = h; }
  Lockable* getObj() const { return ptr; }
  void setObj(Lockable* p) { ptr = p; }
  
  typedef std::pair<uint32_t, Lockable*> rawType;
  rawType rawCopy() const { return std::make_pair(host,ptr); }
};

#else
//Fat pointer taking advantage of unused bits in AMD64 addresses
class fatPointer {
  uintptr_t val;

  uintptr_t compute(uint32_t h, Lockable* p) const {
    return ((uintptr_t)h << 47) | ((uintptr_t)p & 0x80007FFFFFFFFFFF);
  }

  void set(uint32_t h, Lockable* p) {
    val = compute(h,p);
  }

public:
  constexpr fatPointer() noexcept :val(0) {}
  fatPointer(uint32_t h, Lockable* p) noexcept :val(compute(h,p)) {}
  uint32_t getHost() const {
    uintptr_t hval = val;
    hval >>= 47;
    hval &= 0x0000FFFF;
    return hval;
  }
  Lockable* getObj() const {
    uintptr_t hval = val;
    hval &= 0x80007FFFFFFFFFFF;
    return (Lockable*)hval;
  }
  void setHost(uint32_t h) {
    set(h, getObj());
  }
  void setObj(Lockable* p) {
    set(getHost(), p);
  }

  typedef uintptr_t rawType;
  rawType rawCopy() const { return val; }

};
#endif

inline bool operator==(const fatPointer& lhs, const fatPointer& rhs) {
  return lhs.rawCopy() == rhs.rawCopy();
}
inline bool operator!=(const fatPointer& lhs, const fatPointer& rhs) {
  return lhs.rawCopy() != rhs.rawCopy();
}
inline bool operator<=(const fatPointer& lhs, const fatPointer& rhs) {
  return lhs.rawCopy() <= rhs.rawCopy();
}
inline bool operator>=(const fatPointer& lhs, const fatPointer& rhs) {
  return lhs.rawCopy() >= rhs.rawCopy();
}
inline bool operator<(const fatPointer& lhs, const fatPointer& rhs) {
  return lhs.rawCopy() < rhs.rawCopy();
}
inline bool operator>(const fatPointer& lhs, const fatPointer& rhs) {
  return lhs.rawCopy() > rhs.rawCopy();
}

} // namespace Runtime
} // namespace Galois

template<>
struct std::hash<Galois::Runtime::fatPointer> {
  boost::hash<Galois::Runtime::fatPointer::rawType> ih;
  size_t operator()(const Galois::Runtime::fatPointer& ptr) const {
    return ih(ptr.rawCopy());
  }
};


#endif
