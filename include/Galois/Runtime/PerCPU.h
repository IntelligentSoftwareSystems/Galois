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
    item* datum;
    int num;
    void create(int i) {
      assert(!datum && !num);
      num = i;
      datum = new item[num];
      for (int i = 0; i < num; ++i)
	new (&(datum[i].data)) T(); //in place new
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
	for (int i = 0; i < num; ++i)
	  datum[i].data.~T(); // in place delete
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
  };

}

#endif

