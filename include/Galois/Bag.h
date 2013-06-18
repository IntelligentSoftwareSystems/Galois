/** Bags -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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
 * Large unordered collections of things.
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_BAG_H
#define GALOIS_BAG_H

#include "Galois/config.h"
#include "Galois/gstl.h"
#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Runtime/ll/gio.h"
#include "Galois/Runtime/mm/Mem.h"

#include <boost/iterator/iterator_facade.hpp>

#include GALOIS_CXX11_STD_HEADER(algorithm)

namespace Galois {

/**
 * Bag for only concurrent insertions. This data structure
 * supports scalable concurrent pushes but reading the bag
 * can only be done serially.
 */
template<typename T, unsigned int BlockSize = 0>
class InsertBag: private boost::noncopyable {

  struct header {
    header* next;
    T* dbegin; //start of interesting data
    T* dend; //end of valid data
    T* dlast; //end of storage
  };

  template<typename U>
  class Iterator: public boost::iterator_facade<Iterator<U>, U, boost::forward_traversal_tag> {
    friend class boost::iterator_core_access;

    Galois::Runtime::PerThreadStorage<header*>* hd;
    unsigned int thr;
    header* p;
    U* v;

    bool init_thread() {
      p = thr < hd->size() ? *hd->getRemote(thr) : 0;
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

    Iterator(Galois::Runtime::PerThreadStorage<header*>* h, unsigned t):
      hd(h), thr(t), p(0), v(0)
    {
      // find first valid item
      if (!init_thread())
        advance_thread();
    }
  };
  
  Galois::Runtime::MM::FixedSizeAllocator heap;
  Galois::Runtime::PerThreadStorage<header*> heads;

  void insHeader(header* h) {
    header*& H = *heads.getLocal();
    h->next = H;
    H = h;
  }

  header* newHeaderFromAllocator(void *m, unsigned size) {
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
      return newHeaderFromAllocator(heap.allocate(BlockSize), BlockSize);
    } else {
      return newHeaderFromAllocator(Galois::Runtime::MM::pageAlloc(), Galois::Runtime::MM::pageSize);
    }
  }

  void destruct() {
    for (unsigned x = 0; x < heads.size(); ++x) {
      header*& h = *heads.getRemote(x);
      while (h) {
        uninitialized_destroy(h->dbegin, h->dend);
        header* h2 = h;
        h = h->next;
        if (BlockSize)
          heap.deallocate(h2);
        else
          Galois::Runtime::MM::pageFree(h2);
      }
    }
  }

public:
  // static_assert(BlockSize == 0 || BlockSize >= (2 * sizeof(T) + sizeof(header)),
  //     "BlockSize should larger than sizeof(T) + O(1)");

  InsertBag(): heap(BlockSize) { }

  ~InsertBag() {
    destruct();
  }

  void clear() {
    destruct();
  }

  typedef T        value_type;
  typedef const T& const_reference;
  typedef T&       reference;
  typedef Iterator<T> iterator;
  typedef Iterator<const T> const_iterator;
  typedef iterator local_iterator;

  iterator begin() { return iterator(&heads, 0); }
  iterator end() { return iterator(&heads, heads.size()); }
  const_iterator begin() const { return const_iterator(&heads, 0); }
  const_iterator end() const { return const_iterator(&heads, heads.size()); }
  
  local_iterator local_begin() { return local_iterator(&heads, Galois::Runtime::LL::getTID()); }
  local_iterator local_end() { return local_iterator(&heads, Galois::Runtime::LL::getTID() + 1); }

  bool empty() const {
    for (unsigned x = 0; x < heads.size(); ++x) {
      header* h = *heads.getRemote(x);
      if (h)
        return false;
    }
    return true;
  }

  //! Thread safe bag insertion
  template<typename... Args>
  reference emplace(Args&&... args) {
    header* H = *heads.getLocal();
    T* rv;
    if (!H || H->dend == H->dlast) {
      H = newHeader();
      insHeader(H);
    }
    rv = new (H->dend) T(std::forward<Args>(args)...);
    H->dend++;
    return *rv;
  }

  //! Thread safe bag insertion
  reference push(const T& val) { return emplace(val); }
  //! Thread safe bag insertion
  reference push(T&& val) { return emplace(std::move(val)); }

  //! Thread safe bag insertion
  reference push_back(const T& val) { return emplace(val); }
  //! Thread safe bag insertion
  reference push_back(T&& val) { return emplace(std::move(val)); }
};

}

#endif
