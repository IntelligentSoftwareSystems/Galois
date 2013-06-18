/** One element per cache line -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in
 * irregular programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights
 * reserved.  UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES
 * CONCERNING THIS SOFTWARE AND DOCUMENTATION, INCLUDING ANY
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY PARTICULAR PURPOSE,
 * NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY WARRANTY
 * THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF
 * TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO
 * THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect,
 * direct or consequential damages or loss of profits, interruption of
 * business, or related expenses which may arise from use of Software
 * or Documentation, including but not limited to those resulting from
 * defects in Software and/or Documentation, or loss or inaccuracy of
 * data of any kind.  
 *
 * @section Description
 *
 * This wrapper ensures the contents occupy its own cache line(s).
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */
#ifndef GALOIS_RUNTIME_CACHELINESTORAGE_H
#define GALOIS_RUNTIME_CACHELINESTORAGE_H

#include "Galois/config.h"
#include "Galois/Runtime/ll/CompilerSpecific.h"

#include GALOIS_CXX11_STD_HEADER(utility)

namespace Galois {
namespace Runtime {
namespace LL {

template<typename T, int REM>
struct CacheLineImpl {
  GALOIS_ATTRIBUTE_ALIGN_CACHE_LINE T data;
  char pad[REM];
  
  CacheLineImpl() :data() {}

  CacheLineImpl(const T& v) :data(v) {}
  
  template<typename A>
  explicit CacheLineImpl(A&& v) :data(std::forward<A>(v)) {}

  explicit operator T() { return data; }
};

template<typename T>
struct CacheLineImpl<T, 0> {
  GALOIS_ATTRIBUTE_ALIGN_CACHE_LINE T data;
  
  CacheLineImpl() :data() {}

  CacheLineImpl(const T& v) :data(v) {}

  template<typename A>
  explicit CacheLineImpl(A&& v) :data(std::forward<A>(v)) {}

  explicit operator T() { return data; }
};

// Store an item with padding
template<typename T>
struct CacheLineStorage : public CacheLineImpl<T, GALOIS_CACHE_LINE_SIZE % sizeof(T)> {
  typedef CacheLineImpl<T, GALOIS_CACHE_LINE_SIZE % sizeof(T)> PTy;
  
  CacheLineStorage() :PTy() {}

  CacheLineStorage(const T& v) :PTy(v) {}

// XXX(ddn): Forwarding is still wonky in XLC
#if !defined(__IBMCPP__) || __IBMCPP__ > 1210
  template<typename A>
  explicit CacheLineStorage(A&& v) :PTy(std::forward<A>(v)) {}
#endif

  explicit operator T() { return this->data; }

  CacheLineStorage& operator=(const T& v) { this->data = v; return *this; }
};

}
}
} // end namespace Galois

#endif
