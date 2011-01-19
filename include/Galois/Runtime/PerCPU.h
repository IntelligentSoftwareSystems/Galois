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
  item zero_datum;
  item* datum;
  int num;
  void (*reduce)(T&, T&);
  
  void create(int i) {
    assert(!datum && !num);
    num = i;
    datum = new item[num];
  }

  void reduce_and_reset() {
    for (int i = 0; i < num; ++i)
      reduce(zero_datum.data, datum[i].data);
    delete[] datum;
    datum = 0;
    num = 0;
  }

public:
  CPUSpaced(void (*func)(T&, T&))
    :datum(0), num(0), reduce(func)
  {
    init();
  }
  
  ~CPUSpaced() {
    if (datum) {
      delete[] datum;
    }
  }

  T& getMaster() {
    return zero_datum.data;
  }
  
  T& get() {
    int i = ThreadAware::getMyID();
    if (!i)
      return zero_datum.data;
    assert(i <= num);
    assert(datum);
    return datum[i - 1].data;
  }

  const T& get() const {
    int i = ThreadAware::getMyID();
    if (!i)
      return zero_datum.data;
    assert(i <= num);
    assert(datum);
    return datum[i - 1].data;
  }

  const T& getRemote(int i) const {
    if (!i)
      return zero_datum.data;
    assert(i <= num);
    assert(datum);
    return datum[i - 1].data;
  };

  int getCount() const {
    return num + 1;
  }

  virtual void ThreadChange(int newnum) {
    reduce_and_reset();
    if (newnum)
      create(newnum);
  }
  
};

}

#endif

