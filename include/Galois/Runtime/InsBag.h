// Insert Bag implementation -*- C++ -*-
// This one is suppose to be scalable and uses Galois PerCPU classes
#ifndef GALOIS_INSERT_BAG_H_
#define GALOIS_INSERT_BAG_H_

#include "Galois/Runtime/PerCPU.h"
#include <list>

namespace GaloisRuntime {
  
template< class T>
class galois_insert_bag {
  CPUSpaced<std::list<T> > heads;
  
  static void merge(std::list<T>& lhs, std::list<T>& rhs) {
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
