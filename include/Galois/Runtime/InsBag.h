/** Insert Bag -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_RUNTIME_INSBAG_H
#define GALOIS_RUNTIME_INSBAG_H

#include "Galois/Runtime/mm/mem.h" 
#include <iterator>

namespace GaloisRuntime {
  
template< class T>
class galois_insert_bag {
  
  struct holder {
    holder* next;
    T data;
  };

  PtrLock<holder*, true> head;
  GaloisRuntime::PerCPU<holder*> heads;
  
  GaloisRuntime::MM::FixedSizeAllocator allocSrc;

public:
  galois_insert_bag()
    :allocSrc(sizeof(holder))
  {}

  ~galois_insert_bag() {
    while (head.getValue()) {
      holder* H = head.getValue();
      head.setValue(H->next);
      allocSrc.deallocate(H);
    }
  }

  typedef T        value_type;
  typedef const T& const_reference;
  typedef T&       reference;

  class iterator : public std::iterator<std::forward_iterator_tag, T>
  {
    holder* p;
  public:
    iterator(holder* x) :p(x) {}
    iterator(const iterator& mit) : p(mit.p) {}
    iterator& operator++() {if (p) p = p->next; return *this;}
    iterator operator++(int) {iterator tmp(*this); operator++(); return tmp;}
    bool operator==(const iterator& rhs) const {return p==rhs.p;}
    bool operator!=(const iterator& rhs) const {return p!=rhs.p;}
    T& operator*() const {return p->data;}
  };

  iterator begin() {
    return iterator(head.getValue());
  }
  iterator end() {
    return iterator((holder*)0);
  }

  //Only this is thread safe
  reference push(const T& val) {
    holder* h = (holder*)allocSrc.allocate(sizeof(holder));
    new ((void *)&h->data) T(val);
    holder* H = heads.get();
    if (!H) { //no thread local head, use the new node as one
      heads.get() = h;
      //splice new list of one onto the head
      head.lock();
      h->next = head.getValue();
      head.unlock_and_set(h);
    } else {
      //existing thread local head, just append
      h->next = H->next;
      H->next = h;
    }
    return h->data;
  }
  
};
}
#endif
