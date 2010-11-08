// Per CPU/Thread data support -*- C++ -*-

#ifndef __GALOIS_PERCPU_H
#define __GALOIS_PERCPU_H

#include <vector>
#include <cassert>

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
    item* datum;
    int num;
    void create(int i) {
      assert(!datum && !num);
      num = i;
      datum = new item[num];
    }
  public:
    explicit CPUSpaced(int i)
      :datum(0), num(0)
    {
      create(i);
    }

    CPUSpaced()
      :datum(0), num(0)
      {}

    ~CPUSpaced() {
      if (datum) {
	delete[] datum;
      }
    }

    T& operator[] (int i) {
      return datum[i].data;
    }

    int size() const {
      return num;
    }

    void late_initialize(int i) {
      create(i);
    }

    template<typename F>
    void reduce_and_reset(F& Func) {
      item* NI = new item[1]; //must use array style
      for (int i = 0; i < num; ++i)
	NI->data = Func(NI->data, datum[i].data);
      delete[] datum;
      datum = NI;
      num = 1;
    }
    
    //You must have reduced the object already
    void grow(int i) {
      assert(i == 1 || i == 0);
      item* OI = datum;
      num = 0;
      datum = 0;
      create(i);
      datum[i].data = OI->data;
      delete[] OI;
    }

  };

}

#endif

