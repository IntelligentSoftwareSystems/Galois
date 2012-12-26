/** Per CPU/Thread support -*- C++ -*-
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
#ifndef GALOIS_RUNTIME_PERCPU_H
#define GALOIS_RUNTIME_PERCPU_H

#include "Galois/Runtime/ll/TID.h"
#include "Galois/Runtime/ll/HWTopo.h"
#include "Galois/Runtime/ll/CacheLineStorage.h"
#include "Galois/Runtime/ll/CompilerSpecific.h"

#include <boost/utility.hpp>
#include <cassert>

namespace GaloisRuntime {

namespace HIDDEN {
template<typename T, typename BASE>
class PERTHING :private BASE, private boost::noncopyable {
protected:
  LL::CacheLineStorage<T>* datum;

public:
  GALOIS_ATTRIBUTE_DEPRECATED
  PERTHING() {
    datum = new LL::CacheLineStorage<T>[BASE::getMaxSize()];
  }
  explicit PERTHING(const T& v) {
    datum = new LL::CacheLineStorage<T>[BASE::getMaxSize()];
    reset(v);
  }

  ~PERTHING() {
    delete[] datum;
  }

  void reset(const T& d) {
    for (unsigned int i = 0; i < BASE::getMaxSize(); ++i)
      datum[i].data = d;
  }

  T& get(unsigned int i) {
    assert(i < BASE::getMaxSize());
    assert(datum);
    return datum[i].data;
  }
  
  const T& get(unsigned int i) const {
    assert(i < BASE::getMaxSize());
    assert(datum);
    return datum[i].data;
  }

  T& get() {
    return get(BASE::myEID());
  }

  T* getLocal() {
    return &get();
  }

  const T& get() const {
    return get(BASE::myEID());
  }

  T& getNext(unsigned num) {
    return get((BASE::myEID() + 1) % num);
  }
  
  const T& getNext(unsigned num) const {
    return get((BASE::myEID() + 1) % num);
  }

  T& getPrev(unsigned num) {
    return get((BASE::myEID() + num - 1) % num);
  }
  
  const T& getPrev(unsigned num) const {
    return get((BASE::myEID() + num - 1) % num);
  }

  unsigned int size() const {
    return BASE::getMaxSize();
  }

  unsigned myEffectiveID() const {
    return BASE::myEID();
  }

  unsigned effectiveIDFor(unsigned i) const {
    return BASE::otherEID(i);
  }

  bool isLeader() const {
    return BASE::LocalLeader(BASE::myEID());
  }
};

struct H_PERCPU {
  unsigned myEID() const {
    return LL::getTID();
  }
  unsigned getMaxSize() const {
    return LL::getMaxThreads();
  }
  unsigned otherEID(unsigned i) const {
    return i;
  }
  bool LocalLeader(unsigned i) const {
    return true;
  }
};

struct H_PERPACKAGE {
  unsigned myEID() const {
    return LL::getPackageForSelf(LL::getTID());
  }
  unsigned getMaxSize() const {
    return LL::getMaxPackages();
  }
  unsigned otherEID(unsigned i) const {
    return LL::getPackageForThread(i);
  }
  bool LocalLeader(unsigned i) const {
    return LL::isPackageLeaderForSelf(i);
  }
};

struct H_CONSTANT {
  unsigned myEID() const {
    return 0;
  }
  unsigned getMaxSize() const {
    return 1;
  }
  unsigned otherEID(unsigned i) const {
    return i;
  }
  bool LocalLeader(unsigned i) const {
    return true;
  }
};

}

//Stores 1 item per thread
//The master thread is thread 0
//During Parallel regions the threads index
//from 0 -> num - 1 (one thread pool thread shares an index with the user thread)
template<typename T,bool concurrent=true>
class PerCPU;

template<typename T>
class PerCPU<T,true> : public HIDDEN::PERTHING<T, HIDDEN::H_PERCPU> {
public:
  PerCPU() :HIDDEN::PERTHING<T, HIDDEN::H_PERCPU>() {}
  explicit PerCPU(const T& v) :HIDDEN::PERTHING<T, HIDDEN::H_PERCPU>(v) {}
};

template<typename T>
class PerCPU<T,false> : public HIDDEN::PERTHING<T, HIDDEN::H_CONSTANT> {
public:
  PerCPU() :HIDDEN::PERTHING<T, HIDDEN::H_CONSTANT>() {}
  explicit PerCPU(const T& v) :HIDDEN::PERTHING<T, HIDDEN::H_CONSTANT>(v) {}
};

template<typename T,bool concurrent=true>
class PerLevel;

template<typename T>
class PerLevel<T,true> : public HIDDEN::PERTHING<T, HIDDEN::H_PERPACKAGE> {
public:
  PerLevel() :HIDDEN::PERTHING<T, HIDDEN::H_PERPACKAGE>() {}
  explicit PerLevel(const T& v) :HIDDEN::PERTHING<T, HIDDEN::H_PERPACKAGE>(v) {}
};

template<typename T>
class PerLevel<T,false> : public HIDDEN::PERTHING<T, HIDDEN::H_CONSTANT> {
public:
  PerLevel() :HIDDEN::PERTHING<T, HIDDEN::H_CONSTANT>() {}
  explicit PerLevel(const T& v) :HIDDEN::PERTHING<T, HIDDEN::H_CONSTANT>(v) {}
};


}

#endif

