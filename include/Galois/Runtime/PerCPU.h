// Per CPU/Thread data support -*- C++ -*-

#ifndef __GALOIS_PERCPU_H
#define __GALOIS_PERCPU_H

#include <vector>

namespace GaloisRuntime {
  //Return an int (dense from 1) labeling this thread
  int getThreadID();

  //Return the largest allocated thread id
  int getMaxThreadID();

  //This only works for primative types that can be converted to and from 0
  template<typename T>
  class PerCPUData {
    int index;
    //Degenerate copy and assignment
    PerCPUData(const PerCPUData&);
    PerCPUData& operator=(const PerCPUData&);
  public:
    PerCPUData();
    ~PerCPUData();

    T get() const;
    void set(const T&);
    void set_assert(const T& val, const T& old);
    
    T getRemote(int cThreadID) const;
  };

  //Hide type casting from users
  template<typename T>
  class PerCPUPtr {
    PerCPUData<void*> backing;
    //Degenerate copy and assignment
    PerCPUPtr(const PerCPUPtr&);
    PerCPUPtr& operator=(const PerCPUPtr&);
  public:
    PerCPUPtr() {}
    ~PerCPUPtr() {}

    T get() const {
      return (T) backing.get();
    }
    void set(const T& d) {
      backing.set((void*)d);
    }
    void set_assert(const T& d, const T& old) {
      backing.set_assert(d, old);
    }

    T getRemote(int cThreadID) const {
      return (T) backing.getRemote(cThreadID);
    }
  };

  template<typename T>
  class CPUSpaced {
    struct item {
      T data;
      char* padding[64 - (sizeof(T) % 64)];
    };
    std::vector<item> datum;

  public:
    explicit CPUSpaced(int i) {
      datum.resize(i);
    }
    CPUSpaced() {}

    T& operator[] (int i) {
      return datum[i].data;
    }

    int size() const {
      return datum.size();
    }

    void resize(int n) {
      datum.resize(n);
    }
  };

}

#endif

