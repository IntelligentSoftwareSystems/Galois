// Per CPU/Thread data support -*- C++ -*-

#ifndef __GALOIS_PERCPU_H
#define __GALOIS_PERCPU_H

#include "Galois/Runtime/Threads.h"

#include <cassert>

namespace GaloisRuntime {

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
    for (int i = 1; i <= num; ++i)
      reduce(datum[0].data, datum[i].data);
  }

public:
  CPUSpaced(void (*func)(T&, T&))
    :reduce(func)
  {
    num = getSystemThreadPool().size();
    datum = new item[num + 1];
  }
  
  ~CPUSpaced() {
    delete[] datum;
  }

  T& getMaster() {
    return datum[0].data;
  }
  
  T& get() {
    int i = ThreadPool::getMyID();
    assert(i <= num);
    assert(datum);
    return datum[i].data;
  }

  const T& get() const {
    int i = ThreadPool::getMyID();
    assert(i <= num);
    assert(datum);
    return datum[i - 1].data;
  }

  const T& getRemote(int i) const {
    assert(i <= num);
    assert(datum);
    return datum[i - 1].data;
  };

  int getCount() const {
    return num + 1;
  }

  virtual void ThreadChange(bool starting) {
    if (!starting)
      __reduce();
  }
};

}

#endif

