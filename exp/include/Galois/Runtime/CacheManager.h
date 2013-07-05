/** Galois Remote Object Store -*- C++ -*-
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
 * @author Manoj Dhanapal <madhanap@cs.utexas.edu>
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_RUNTIME_CACHEMANAGER_H
#define GALOIS_RUNTIME_CACHEMANAGER_H

#include  <boost/functional/hash.hpp>

namespace Galois {
namespace Runtime {

class remoteObj {
public:
  virtual Lockable* getObj() = 0;
  virtual size_t getTypeHash() const = 0;
};

template<typename T>
class remoteObjImpl : public remoteObj {
  T obj;
public:
  virtual size_t getTypeHash() const { return typeid(T).hash_code(); }
  virtual Lockable* getObj()    { return &obj; }
};

typedef std::pair<uint32_t, Lockable*> fatPointer;

class CacheManager {

  std::unordered_map<fatPointer, remoteObj*, boost::hash<fatPointer> > remoteObjects;
  LL::SimpleLock<true> Lock;

public:

  template<typename T>
  remoteObjImpl<T>* resolve(fatPointer ptr);

  remoteObj* weakResolve(fatPointer ptr) {
    assert(ptr.first != NetworkInterface::ID);
    LL::SLguard lgr(Lock);
    return remoteObjects[ptr];
  }
};

CacheManager& getCacheManager();

}
}

#endif
