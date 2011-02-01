// Per CPU/Thread data support -*- C++ -*-

#ifndef __GALOIS_PERCPU_H
#define __GALOIS_PERCPU_H

#include "Galois/Runtime/Threads.h"

#include <cassert>

namespace GaloisRuntime {

//Stores 1 item per thread
//The master thread is thread 0
//Durring Parallel regions the threads index
//from 0 -> num - 1 (one thread pool thread shares an index with the user thread)
template<typename T>
class CPUSpaced : public ThreadAware {
  struct item {
    T data;
    char* padding[64 - (sizeof(T) % 64)];
    item() :data() {}
  };
  item* datum;
  unsigned int num;
  void (*reduce)(T&, T&);
  
  void __reduce() {
    if (reduce)
      for (int i = 1; i < num; ++i)
	reduce(datum[0].data, datum[i].data);
  }

  int myID() const {
    int i = ThreadPool::getMyID();
    return std::max(0, i - 1);
  }

public:
  explicit CPUSpaced(void (*func)(T&, T&))
    :reduce(func)
  {
    num = getSystemThreadPool().getMaxThreads();
    datum = new item[num];
  }
  
  ~CPUSpaced() {
    delete[] datum;
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
    int i = ThreadPool::getMyID();
    i += 1;
    i %= getSystemThreadPool().getActiveThreads();
    return get(i);
  }

  const T& getNext() const {
    int i = ThreadPool::getMyID();
    i += 1;
    i %= getSystemThreadPool().getActiveThreads();
    return get(i);
  }

  int getCount() const {
    return num;
  }

  int size() const {
    return num;
  }

  virtual void ThreadChange(bool starting) {
    if (!starting)
      __reduce();
  }
};

}

#endif

