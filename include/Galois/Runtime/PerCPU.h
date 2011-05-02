// Per CPU/Thread data support -*- C++ -*-

#ifndef __GALOIS_PERCPU_H
#define __GALOIS_PERCPU_H

#include "Threads.h"
#include "CacheLineStorage.h"

#include <boost/utility.hpp>
#include <cassert>

namespace GaloisRuntime {

//Stores 1 item per thread
//The master thread is thread 0
//Durring Parallel regions the threads index
//from 0 -> num - 1 (one thread pool thread shares an index with the user thread)
template<typename T>
class PerCPU : boost::noncopyable {
protected:
  cache_line_storage<T>* datum;
  unsigned int num;

  int myID() const {
    int i = ThreadPool::getMyID();
    return std::max(0, i - 1);
  }


public:
  PerCPU()
  {
    num = getSystemThreadPolicy().getNumThreads();
    datum = new cache_line_storage<T>[num];
  }
  explicit PerCPU(const T& ival)
  {
    num = getSystemThreadPolicy().getNumThreads();
    datum = new cache_line_storage<T>[num];
    for (int i = 0; i < num; ++i)
      datum[i] = ival;
  }
  
  virtual ~PerCPU() {
    delete[] datum;
  }

  int myEffectiveID() const {
    return myID();
  }

  T& get(int i) {
    assert(i < num);
    assert(datum);
    return datum[i].data;
  }
  
  const T& get(int i) const {
    assert(i < num);
    assert(datum);
    return datum[i].data;
  }
  
  T& get() {
    return get(myID());
  }

  const T& get() const {
    return get(myID());
  }

  T& getNext() {
    return get((myID() + 1) % getSystemThreadPool().getActiveThreads());
  }

  const T& getNext() const {
    return get((myID() + 1) % getSystemThreadPool().getActiveThreads());
  }

  int size() const {
    return num;
  }
};

template<typename T>
class PerCPU_merge : public PerCPU<T>, public ThreadAware{
  void (*reduce)(T&, T&);

  void __reduce() {
    if (reduce)
      for (int i = 1; i < PerCPU<T>::num; ++i)
	reduce(PerCPU<T>::datum[0].data, PerCPU<T>::datum[i].data);
  }

public:
  explicit PerCPU_merge(void (*func)(T&, T&))
    :reduce(func)
  {}
  virtual void ThreadChange(bool starting) {
    if (!starting)
      __reduce();
  }
};

template<typename T>
class PerLevel {
  cache_line_storage<T>* datum;
  unsigned int num;
  unsigned int level;
  ThreadPolicy& P;

protected:

  int myID() const {
    int i = ThreadPool::getMyID();
    return std::max(0, i - 1);
  }


public:
  PerLevel() :P(getSystemThreadPolicy())
  {
    //last iteresting level (should be package)
    level = P.getNumLevels() - 1;
    num = P.getLevelSize(level);
    datum = new cache_line_storage<T>[num];
  }
  
  virtual ~PerLevel() {
    delete[] datum;
  }

  int myEffectiveID() const {
    return P.indexLevelMap(level, myID());
  }

  T& get(int i) {
    assert(i < num);
    assert(datum);
    return datum[i].data;
  }
  
  const T& get(int i) const {
    assert(i < num);
    assert(datum);
    return datum[i].data;
  }
  
  T& get() {
    return get(myEffectiveID());
  }

  const T& get() const {
    return get(myEffectiveID());
  }

  int size() const {
    return num;
  }
};

}

#endif

