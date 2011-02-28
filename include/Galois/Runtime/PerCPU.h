// Per CPU/Thread data support -*- C++ -*-

#ifndef __GALOIS_PERCPU_H
#define __GALOIS_PERCPU_H

#include "Galois/Runtime/Threads.h"

#include <cassert>

namespace GaloisRuntime {

//xeons have 64 byte cache lines, but will prefetch 2 at a time
#define CACHE_LINE_SIZE 128

// Store an item with padding
template<typename T>
struct cache_line_storage {
  T data __attribute__((aligned(CACHE_LINE_SIZE)));
  char pad[ CACHE_LINE_SIZE % sizeof(T) ?
	    CACHE_LINE_SIZE - (sizeof(T) % CACHE_LINE_SIZE) :
	    0 ];
};


//Stores 1 item per thread
//The master thread is thread 0
//Durring Parallel regions the threads index
//from 0 -> num - 1 (one thread pool thread shares an index with the user thread)
template<typename T>
class PerCPU : public ThreadAware {
  cache_line_storage<T>* datum;
  unsigned int num;
  void (*reduce)(T&, T&);

  void __reduce() {
    if (reduce)
      for (int i = 1; i < num; ++i)
	reduce(datum[0].data, datum[i].data);
  }

protected:

  int myID() const {
    int i = ThreadPool::getMyID();
    return std::max(0, i - 1);
  }


public:
  explicit PerCPU(void (*func)(T&, T&))
    :reduce(func)
  {
    num = getSystemThreadPool().getMaxThreads();
    datum = new cache_line_storage<T>[num];
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

  int size() const {
    return num;
  }

  virtual void ThreadChange(bool starting) {
    if (!starting)
      __reduce();
  }
};

template<typename T>
class PerCPU_ring : public PerCPU<T> {
  using PerCPU<T>::myID;
public:
  using PerCPU<T>::get;
  explicit PerCPU_ring(void (*func)(T&, T&))
    :PerCPU<T>(func)
  {}

  T& getNext() {
    int i = myID() + 1;
    i %= getSystemThreadPool().getActiveThreads();
    return get(i);
  }

  const T& getNext() const {
    int i = myID() + 1;
    i %= getSystemThreadPool().getActiveThreads();
    return get(i);
  }
};

}

#endif

