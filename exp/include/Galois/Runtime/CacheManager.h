/** Galois Remote Object Store -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2014, The University of Texas at Austin. All rights reserved.
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
#ifndef GALOIS_RUNTIME_CACHEMANAGER_H
#define GALOIS_RUNTIME_CACHEMANAGER_H

#include "Galois/Runtime/ll/SimpleLock.h"
#include "Galois/Runtime/FatPointer.h"
//#include "Galois/Runtime/Serialize.h"

#include <unordered_map>
#include <deque>

namespace Galois {
namespace Runtime {

class DeSerializeBuffer;
template<typename T>
T gDeserializeObj(DeSerializeBuffer&);

class CacheManager {

  class remoteObj {
    bool RW;
  public:
    remoteObj(bool rw) :RW(rw) {}
    virtual ~remoteObj();
    virtual void* getObj() = 0;
    bool isRO() const { return !RW; }
    bool isRW() const { return  RW; }
  };
  
  template<typename T>
  class remoteObjImpl : public remoteObj {
    T obj;
  public:
    remoteObjImpl(bool RW, DeSerializeBuffer& buf)
      :remoteObj(RW), obj{std::move(gDeserializeObj<T>(buf))} {}
    remoteObjImpl(bool RW, T&& buf)
      :remoteObj(RW), obj(buf) {}
    virtual ~remoteObjImpl() {}
    virtual void* getObj() { return &obj; }
  };

  std::unordered_map<fatPointer, remoteObj*> remoteObjects;
  std::deque<remoteObj*> garbage;
  LL::SimpleLock Lock;

public:

  template<typename T>
  void create(fatPointer ptr, bool write, DeSerializeBuffer& buf) {
    LL::SLguard lgr(Lock);
    remoteObj*& obj = remoteObjects[ptr];
    if (obj) { // creating can replace RO with RW objects
      assert(obj->isRO() && write);
      garbage.push_back(obj);
      obj = nullptr;
    }
    obj = new remoteObjImpl<T>(write, buf);
  }

  template<typename T>
  void create(fatPointer ptr, bool write, T&& buf) {
    LL::SLguard lgr(Lock);
    remoteObj*& obj = remoteObjects[ptr];
    if (obj) { // creating can replace RO with RW objects
      assert(obj->isRO() && write);
      garbage.push_back(obj);
      obj = nullptr;
    }
    obj = new remoteObjImpl<T>(write, std::forward<T>(buf));
  }

  void* resolve(fatPointer ptr, bool write);
};

CacheManager& getCacheManager();

}
}

#endif
