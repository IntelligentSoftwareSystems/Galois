/** One element per cache line -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in
 * irregular programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights
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
 * This wrapper ensures the contents occupie its own cache line(s).
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */
#ifndef GALOIS_RUNTIME_CACHE_LINE_STORAGE_H
#define GALOIS_RUNTIME_CACHE_LINE_STORAGE_H

#include "CompilerSpecific.h"

namespace GaloisRuntime {
namespace LL {



template<typename T, int REM>
struct CacheLineImp {
  GALOIS_ATTRIBUTE_ALIGN_CACHE_LINE T data;
  char pad[REM];
  CacheLineImp() :data() {}
  explicit CacheLineImp(const T& v) :data(v) {}
};

template<typename T>
struct CacheLineImp<T, 0> {
  GALOIS_ATTRIBUTE_ALIGN_CACHE_LINE T data;
  CacheLineImp() :data() {}
  explicit CacheLineImp(const T& v) :data(v) {}
};

// Store an item with padding
template<typename T>
struct CacheLineStorage : public CacheLineImp<T, GALOIS_CACHE_LINE_SIZE % sizeof(T)> {
  typedef CacheLineImp<T, GALOIS_CACHE_LINE_SIZE % sizeof(T)> PTy;
  using PTy::data;
  CacheLineStorage() :PTy() {}
  explicit CacheLineStorage(const T& v) :PTy(v) {}
  CacheLineStorage& operator=(const T& v) { data = v; return *this; }
};

}
}

#endif //_CACHE_LINE_STORAGE_H
