// Insert Bag implementation -*- C++ -*-
// This one is suppose to be scalable and uses Galois PerCPU classes
#ifndef GALOIS_INSERT_BAG_H_
#define GALOIS_INSERT_BAG_H_

#include "Galois/Runtime/PerCPU.h"
#include <list>
#include <ext/malloc_allocator.h>

namespace GaloisRuntime {
  
template< class T>
class galois_insert_bag {
  typedef std::list<T, __gnu_cxx::malloc_allocator<T> > ListTy;
  //typedef std::list<T> ListTy;
  CPUSpaced<ListTy> heads;
  
  static void merge(ListTy& lhs, ListTy& rhs) {
    lhs.splice(lhs.begin(), rhs);
  }

public:
  galois_insert_bag()
    :heads(merge)
  {}

  typedef T        value_type;
  typedef const T& const_reference;
  typedef T&       reference;
  typedef typename std::list<T>::iterator iterator;

  iterator begin() {
    return heads.get().begin();
  }
  
  iterator end() {
    return heads.get().end();
  }

  //Only this is thread safe
  reference push(const T& val) {
    heads.get().push_front(val);
    return heads.get().front();
  }
  
};
}
#endif
