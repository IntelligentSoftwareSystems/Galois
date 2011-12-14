/** Insert Bag v2 -*- C++ -*-
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

#include "Galois/Runtime/mem.h" 
#include <iterator>

namespace GaloisRuntime {

template<class T>
class galois_insert_bag {
  
  struct header {
    header* next;
    T* dbegin; //start of interesting data
    T* dend; //end of valid data
    T* dlast; //end of storage
  };

  PtrLock<header*, true> realHead;
  GaloisRuntime::PerCPU<header*> heads;

  void insHeader(header* h) {
    header* H = heads.get();
    if (!H) { //no thread local head, use the new node as one
      //splice new list of one onto the head
      realHead.lock();
      h->next = realHead.getValue();
      realHead.unlock_and_set(h);
    } else {
      //existing thread local head, just append
      h->next = H->next;
      H->next = h;
    }
    heads.get() = h;
  }

  header* newHeader() {
    void* m = MM::pageAlloc();
    header* H = new (m) header();
    int offset = 1;
    if (sizeof(T) < sizeof(header))
      offset += sizeof(header)/sizeof(T);
    T* a = reinterpret_cast<T*>(m);
    H->dbegin = &a[offset];
    H->dend = H->dbegin;
    H->dlast = &a[(MM::pageSize / sizeof(T))];
    H->next = 0;
    return H;
  }

public:
  galois_insert_bag() {}

  ~galois_insert_bag() {
    while (realHead.getValue()) {
      header* h = realHead.getValue();
      realHead.setValue(h->next);
      for(T* ii = h->dbegin, *ee = h->dend; ii != ee; ++ii) {
	ii->~T();
      }
      MM::pageFree(h);
    }
  }

  typedef T        value_type;
  typedef const T& const_reference;
  typedef T&       reference;

  class iterator : public std::iterator<std::forward_iterator_tag, T> {
    header* p;
    T* v;
  public:
    iterator(header* x) :p(x), v(x ? x->dbegin : 0) {}
    iterator(const iterator& mit) : p(mit.p), v(mit.v) {}
    iterator& operator++() {
      if (p) {
	++v;
	if (v == p->dend) {
	  p = p->next;
	  v = (p ? p->dbegin : 0);
	}
      }
      return *this;
    }
    iterator operator++(int) {iterator tmp(*this); operator++(); return tmp;}
    bool operator==(const iterator& rhs) const {
      return (p==rhs.p && rhs.v == rhs.v);
    }
    bool operator!=(const iterator& rhs) const {
      return !(p==rhs.p);
    }
    T& operator*() const {return *v;}
  };

  iterator begin() {
    return iterator(realHead.getValue());
  }
  iterator end() {
    return iterator((header*)0);
  }

  //Only this is thread safe
  reference push(const T& val) {
    header* H = heads.get();
    T* rv;
    if (!H || H->dend == H->dlast) {
      H = newHeader();
      insHeader(H);
    }
    rv = new (H->dend) T(val);
    H->dend++;
    return *rv;
  }
};
}
#endif
