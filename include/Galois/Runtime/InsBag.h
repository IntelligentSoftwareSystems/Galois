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

#include "Galois/Runtime/mm/Mem.h" 
#include "Galois/Runtime/ll/PtrLock.h"
#include "Galois/Runtime/DualLevelIterator.h"
#include <iterator>

namespace GaloisRuntime {

template<class T>
class galois_insert_bag : private boost::noncopyable {
  
public:
  class gib_Tile : private boost::noncopyable {
    gib_Tile* next;
    T* dbegin;
    T* dend;
    T* dlast;

    friend class galois_insert_bag;

    gib_Tile(T* s, T* e)
      :next(0), dbegin(s), dend(s), dlast(e)
    {}

  public:
    typedef T* iterator;
    iterator begin() { return dbegin; }
    iterator end()  { return dend;   }
    bool full() { return dend == dlast; }
  };

private:
  LL::PtrLock<gib_Tile*, true> realHead;
  GaloisRuntime::PerCPU<gib_Tile*> heads;

  gib_Tile* newHeader() {
    //First Create
    void* m = MM::pageAlloc();
    int offset = 1;
    if (sizeof(T) < sizeof(gib_Tile))
      offset += sizeof(gib_Tile)/sizeof(T);
    T* a = reinterpret_cast<T*>(m);
    gib_Tile* h = new (m) gib_Tile(&a[offset], &a[(MM::pageSize / sizeof(T))]);
    //Then Insert
    gib_Tile*& H = heads.get();
    if (!H) { //no thread local head, use the new node as one
      //splice new list of one onto the head
      realHead.lock();
      h->next = realHead.getValue();
      realHead.unlock_and_set(h);
    } else {
      //existing thread local head, just append
      h->next = H->next;
      asm volatile ("":::"memory");
      H->next = h;
    }
    H = h;
    return h;
  }

  void destruct() {
    while (realHead.getValue()) {
      gib_Tile* h = realHead.getValue();
      realHead.setValue(h->next);
      for (T* ii = h->dbegin, *ee = h->dend; ii != ee; ++ii) {
	ii->~T();
      }
      MM::pageFree(h);
    }
  }

public:
  galois_insert_bag() {}

  ~galois_insert_bag() {
    destruct();
  }

  void clear() {
    destruct();
    for (unsigned i = 0; i < heads.size(); ++i)
      heads.get(i) = 0;
  }

  typedef T        value_type;
  typedef const T& const_reference;
  typedef T&       reference;

  class tile_iterator : public std::iterator<std::forward_iterator_tag, gib_Tile> {
    gib_Tile* p;
    friend class galois_insert_bag;
    tile_iterator(gib_Tile* x) :p(x) {}
  public:
    tile_iterator() :p(0) {}
    tile_iterator(const tile_iterator& mit) : p(mit.p) {}
    tile_iterator& operator++() { p = p->next; return *this; }
    tile_iterator operator++(int) {tile_iterator tmp(*this); operator++(); return tmp;}
    bool operator==(const tile_iterator& rhs) const { return p == rhs.p; }
    bool operator!=(const tile_iterator& rhs) const { return p != rhs.p; }
    gib_Tile& operator*() { return *p; }
    gib_Tile& operator*() const { return *p; }
  };

  tile_iterator tile_begin() {
    return tile_iterator(realHead.getValue());
  }
  tile_iterator tile_end() {
    return tile_iterator();
  }

  typedef GaloisRuntime::DualLevelIterator<tile_iterator> iterator;
  
  iterator begin() {
    return iterator(tile_begin(), tile_end());
  }

  iterator end() {
    return iterator(tile_end(), tile_end());
  }

  //Only this is thread safe
  reference push(const T& val) {
    gib_Tile* H = heads.get();
    T* rv;
    if (!H || H->dend == H->dlast) {
      H = newHeader();
    }
    rv = new (H->dend) T(val);
    H->dend++;
    return *rv;
  }
};
}
#endif
