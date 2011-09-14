// Per CPU/Thread data support -*- C++ -*-
/*
Galois, a framework to exploit amorphous data-parallelism in irregular
programs.

Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS SOFTWARE
AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY
PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY
WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF TRADE.
NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO THE USE OF THE
SOFTWARE OR DOCUMENTATION. Under no circumstances shall University be liable
for incidental, special, indirect, direct or consequential damages or loss of
profits, interruption of business, or related expenses which may arise from use
of Software or Documentation, including but not limited to those resulting from
defects in Software and/or Documentation, or loss or inaccuracy of data of any
kind.
*/

#ifndef _GALOIS_RUNTIME_PERCPU_H
#define _GALOIS_RUNTIME_PERCPU_H

#include "Threads.h"
#include "CacheLineStorage.h"

#include <boost/utility.hpp>
#include <cassert>
//#include <iostream>

namespace GaloisRuntime {

namespace HIDDEN {
template<typename T>
class PERTHING :private boost::noncopyable {
protected:
  cache_line_storage<T>* datum;
  unsigned int num;

  int myID() const {
    return ThreadPool::getMyID();
  }

  void create(int n) {
    num = n;
    datum = new cache_line_storage<T>[num];
  }

public:
  ~PERTHING() {
    delete[] datum;
  }

  void reset(const T& d) {
    for (unsigned int i = 0; i < num; ++i)
      datum[i].data = d;
  }

  T& get(unsigned int i) {
    assert(i < num);
    assert(datum);
    return datum[i].data;
  }
  
  const T& get(unsigned int i) const {
    assert(i < num);
    assert(datum);
    return datum[i].data;
  }

  unsigned int size() const {
    return num;
  }

};
}

//Stores 1 item per thread
//The master thread is thread 0
//Durring Parallel regions the threads index
//from 0 -> num - 1 (one thread pool thread shares an index with the user thread)
template<typename T>
class PerCPU : public HIDDEN::PERTHING<T> {
  using HIDDEN::PERTHING<T>::create;
  using HIDDEN::PERTHING<T>::myID;

public:
  PerCPU()
  {
    create(getSystemThreadPolicy().getNumThreads());
  }
  explicit PerCPU(const T& ival)
  {
    create(getSystemThreadPolicy().getNumThreads());
    reset(ival);
  }
  
  unsigned int myEffectiveID() const {
    return myID();
  }
  
  T& get() {
    return get(myID());
  }

  const T& get() const {
    return get(myID());
  }

  // Duplicate superclass functions because superclass is dependent name and
  // thus are difficult to access directly, especially by clients of this
  // class
  T& get(unsigned i) {
    return HIDDEN::PERTHING<T>::get(i);
  }

  const T& get(unsigned i) const {
    return HIDDEN::PERTHING<T>::get(i);
  }

  T& getNext() {
    int n = (myID() + 1) % getSystemThreadPool().getActiveThreads();
    //std::cerr << myID() << " " << n << "\n";
    return get(n);
  }

  const T& getNext() const {
    return get((myID() + 1) % getSystemThreadPool().getActiveThreads());
  }

};

template<typename T>
class PerLevel : public HIDDEN::PERTHING<T> {
  using HIDDEN::PERTHING<T>::create;
  using HIDDEN::PERTHING<T>::myID;

  unsigned int level;
  ThreadPolicy& P;

public:
  PerLevel() :P(getSystemThreadPolicy())
  {
    //last iteresting level (should be package)
    level = P.getNumLevels() - 1;
    create(P.getLevelSize(level));
  }
  explicit PerLevel(const T& ival)
  {
    level = P.getNumLevels() - 1;
    create(P.getLevelSize(level));
    reset(ival);
  }

  unsigned int myEffectiveID() const {
    return P.indexLevelMap(level, myID());
  }

  T& get() {
    return get(myEffectiveID());
  }

  const T& get() const {
    return get(myEffectiveID());
  }

  // Duplicate superclass functions because superclass is dependent name and
  // thus are difficult to access directly, especially by clients of this
  // class
  T& get(unsigned i) {
    return HIDDEN::PERTHING<T>::get(i);
  }

  const T& get(unsigned i) const {
    return HIDDEN::PERTHING<T>::get(i);
  }

  bool isFirstInLevel() const {
    return P.isFirstInLevel(level, myID());
  }

};

}

#endif

