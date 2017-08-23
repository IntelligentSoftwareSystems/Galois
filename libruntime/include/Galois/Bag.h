/** Bags -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * Large unordered collections of things.
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#ifndef GALOIS_BAG_H
#define GALOIS_BAG_H

#include "Galois/gstl.h"
#include "Galois/Substrate/PerThreadStorage.h"
#include "Galois/gIO.h"
#include "Galois/Runtime/Mem.h"

#include <boost/iterator/iterator_facade.hpp>
#include <stdexcept>
#include <algorithm>

namespace Galois {

/**
 * Unordered collection of elements. This data structure supports scalable
 * concurrent pushes but reading the bag can only be done serially.
 */
template<typename T, unsigned int BlockSize = 0>
class InsertBag {

  struct header {
    header* next;
    T* dbegin; //start of interesting data
    T* dend; //end of valid data
    T* dlast; //end of storage
  };

  typedef std::pair<header*, header*> PerThread;

public:
  template<typename U>
  class Iterator: public boost::iterator_facade<Iterator<U>, U, boost::forward_traversal_tag> {
    friend class boost::iterator_core_access;

    Galois::Substrate::PerThreadStorage<std::pair<header*,header*> >* hd;
    unsigned int thr;
    header* p;
    U* v;

    bool init_thread() {
      p = thr < hd->size() ? hd->getRemote(thr)->first : 0;
      v = p ? p->dbegin : 0;
      return p;
    }

    bool advance_local() {
      if (p) {
        ++v;
        return v != p->dend;
      }
      return false;
    }

    bool advance_chunk() {
      if (p) {
        p = p->next;
        v = p ? p->dbegin : 0;
      }
      return p;
    }

    void advance_thread() {
      while (thr < hd->size()) {
        ++thr;
        if (init_thread())
          return;
      }
    }

    void increment() {
      if (advance_local()) return;
      if (advance_chunk()) return;
      advance_thread();
    }

    template<typename OtherTy>
    bool equal(const Iterator<OtherTy>& o) const {
      return hd == o.hd && thr == o.thr && p == o.p && v == o.v;
    }

    U& dereference() const { return *v; }

  public:
    Iterator(): hd(0), thr(0), p(0), v(0) { }

    template<typename OtherTy>
    Iterator(const Iterator<OtherTy>& o): hd(o.hd), thr(o.thr), p(o.p), v(o.v) { }

    Iterator(Galois::Substrate::PerThreadStorage<std::pair<header*,header*> >* h, unsigned t):
      hd(h), thr(t), p(0), v(0)
    {
      // find first valid item
      if (!init_thread())
        advance_thread();
    }
  };

private:
  Galois::Runtime::FixedSizeHeap heap;
  Galois::Substrate::PerThreadStorage<PerThread> heads;

  void insHeader(header* h) {
    PerThread& hpair = *heads.getLocal();
    if (hpair.second) {
      hpair.second->next = h;
      hpair.second = h;
    } else {
      hpair.first = hpair.second = h;
    }
  }

  header* newHeaderFromHeap(void *m, unsigned size) {
    header* H = new (m) header();
    int offset = 1;
    if (sizeof(T) < sizeof(header))
      offset += sizeof(header)/sizeof(T);
    T* a = reinterpret_cast<T*>(m);
    H->dbegin = &a[offset];
    H->dend = H->dbegin;
    H->dlast = &a[(size / sizeof(T))];
    H->next = 0;
    return H;
  }

  header* newHeader() {
    if (BlockSize) {
      return newHeaderFromHeap(heap.allocate(BlockSize), BlockSize);
    } else {
      return newHeaderFromHeap(Galois::Runtime::pagePoolAlloc(), Galois::Runtime::pagePoolSize());
    }
  }

  void destruct() {
    for (unsigned x = 0; x < heads.size(); ++x) {
      PerThread& hpair = *heads.getRemote(x);
      header*& h = hpair.first;
      while (h) {
        uninitialized_destroy(h->dbegin, h->dend);
        header* h2 = h;
        h = h->next;
        if (BlockSize)
          heap.deallocate(h2);
        else
          Galois::Runtime::pagePoolFree(h2);
      }
      hpair.second = 0;
    }
  }

public:
  // static_assert(BlockSize == 0 || BlockSize >= (2 * sizeof(T) + sizeof(header)),
  //     "BlockSize should larger than sizeof(T) + O(1)");

  InsertBag(): heap(BlockSize) { }
  InsertBag(InsertBag&& o): heap(BlockSize) {
    std::swap(heap, o.heap);
    std::swap(heads, o.heads);
  }

  InsertBag& operator=(InsertBag&& o) {
    std::swap(heap, o.heap);
    std::swap(heads, o.heads);
    return *this;
  }

  InsertBag(const InsertBag&) = delete;
  InsertBag& operator=(const InsertBag&) = delete;

  ~InsertBag() {
    destruct();
  }

  void clear() {
    destruct();
  }

  void swap(InsertBag& o) {
    std::swap(heap, o.heap);
    std::swap(heads, o.heads);
  }

  typedef T value_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef const T& const_reference;
  typedef T& reference;
  typedef Iterator<T> iterator;
  typedef Iterator<const T> const_iterator;
  typedef iterator local_iterator;

  iterator begin() { return iterator(&heads, 0); }
  iterator end() { return iterator(&heads, heads.size()); }
  const_iterator begin() const { return const_iterator(&heads, 0); }
  const_iterator end() const { return const_iterator(&heads, heads.size()); }
  
  local_iterator local_begin() { return local_iterator(&heads, Galois::Substrate::ThreadPool::getTID()); }
  local_iterator local_end() { return local_iterator(&heads, Galois::Substrate::ThreadPool::getTID() + 1); }

  bool empty() const {
    for (unsigned x = 0; x < heads.size(); ++x) {
      header* h = heads.getRemote(x)->first;
      if (h)
        return false;
    }
    return true;
  }

  //! Thread safe bag insertion
  template<typename... Args>
  reference emplace(Args&&... args) {
    header* H = heads.getLocal()->second;
    T* rv;
    if (!H || H->dend == H->dlast) {
      H = newHeader();
      insHeader(H);
    }
    rv = new (H->dend) T(std::forward<Args>(args)...);
    ++H->dend;
    return *rv;
  }

  template<typename... Args>
  reference emplace_back(Args&&... args) {
    return emplace(std::forward<Args>(args)...);
  }

  /**
   * Pop the last element pushed by this thread. The number of consecutive
   * pops supported without intevening pushes is implementation dependent. 
   */
  void pop() {
    header* H = heads.getLocal()->second;
    if (H->dbegin == H->dend) {
      throw std::out_of_range("InsertBag::pop");
    }
    uninitialized_destroy(H->dend - 1, H->dend);
    --H->dend;
  }

  //! Thread safe bag insertion
  template<typename ItemTy>
  reference push(ItemTy&& val) { return emplace(std::forward<ItemTy>(val)); }

  //! Thread safe bag insertion
  template<typename ItemTy>
  reference push_back(ItemTy&& val) { return emplace(std::forward<ItemTy>(val)); }
};

}

#endif
