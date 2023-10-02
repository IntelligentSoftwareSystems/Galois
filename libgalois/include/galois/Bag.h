/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#ifndef GALOIS_BAG_H
#define GALOIS_BAG_H

#include <algorithm>
#include <stdexcept>

#include <boost/iterator/iterator_facade.hpp>

#include "galois/config.h"
#include "galois/gstl.h"
#include "galois/runtime/Executor_OnEach.h"
#include "galois/substrate/PerThreadStorage.h"
#include "galois/gIO.h"
#include "galois/runtime/Mem.h"

namespace galois {

/**
 * Unordered collection of elements. This data structure supports scalable
 * concurrent pushes but reading the bag can only be done serially.
 */
template <typename T, unsigned int BlockSize = 0>
class InsertBag {

  struct header {
    header* next;
    T* dbegin; // start of interesting data
    T* dend;   // end of valid data
    T* dlast;  // end of storage
  };

  typedef std::pair<header*, header*> PerThread;

public:
  template <typename U>
  class Iterator : public boost::iterator_facade<Iterator<U>, U,
                                                 boost::forward_traversal_tag> {
    friend class boost::iterator_core_access;

    galois::substrate::PerThreadStorage<std::pair<header*, header*>>* hd;
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
      if (advance_local())
        return;
      if (advance_chunk())
        return;
      advance_thread();
    }

    template <typename OtherTy>
    bool equal(const Iterator<OtherTy>& o) const {
      return hd == o.hd && thr == o.thr && p == o.p && v == o.v;
    }

    U& dereference() const { return *v; }

  public:
    Iterator() : hd(0), thr(0), p(0), v(0) {}

    template <typename OtherTy>
    Iterator(const Iterator<OtherTy>& o)
        : hd(o.hd), thr(o.thr), p(o.p), v(o.v) {}

    Iterator(
        galois::substrate::PerThreadStorage<std::pair<header*, header*>>* h,
        unsigned t)
        : hd(h), thr(t), p(0), v(0) {
      // find first valid item
      if (!init_thread())
        advance_thread();
    }
  };

private:
  galois::runtime::FixedSizeHeap heap;
  galois::substrate::PerThreadStorage<PerThread> heads;

  void insHeader(header* h) {
    PerThread& hpair = *heads.getLocal();
    if (hpair.second) {
      hpair.second->next = h;
      hpair.second       = h;
    } else {
      hpair.first = hpair.second = h;
    }
  }

  header* newHeaderFromHeap(void* m, unsigned size) {
    header* H  = new (m) header();
    int offset = 1;
    if (sizeof(T) < sizeof(header))
      offset += sizeof(header) / sizeof(T);
    T* a      = reinterpret_cast<T*>(m);
    H->dbegin = &a[offset];
    H->dend   = H->dbegin;
    H->dlast  = &a[(size / sizeof(T))];
    H->next   = 0;
    return H;
  }

  header* newHeader() {
    if (BlockSize) {
      return newHeaderFromHeap(heap.allocate(BlockSize), BlockSize);
    } else {
      return newHeaderFromHeap(galois::runtime::pagePoolAlloc(),
                               galois::runtime::pagePoolSize());
    }
  }

  void destruct_serial() {
    for (unsigned x = 0; x < heads.size(); ++x) {
      PerThread& hpair = *heads.getRemote(x);
      header*& h       = hpair.first;
      while (h) {
        uninitialized_destroy(h->dbegin, h->dend);
        header* h2 = h;
        h          = h->next;
        if (BlockSize)
          heap.deallocate(h2);
        else
          galois::runtime::pagePoolFree(h2);
      }
      hpair.second = 0;
    }
  }

  void destruct_parallel(void) {
    galois::runtime::on_each_gen(
        [this](const unsigned int tid, const unsigned int) {
          PerThread& hpair = *heads.getLocal(tid);
          header*& h       = hpair.first;
          while (h) {
            uninitialized_destroy(h->dbegin, h->dend);
            header* h2 = h;
            h          = h->next;
            if (BlockSize)
              heap.deallocate(h2);
            else
              galois::runtime::pagePoolFree(h2);
          }
          hpair.second = 0;
        },
        std::make_tuple(galois::no_stats()));
  }

public:
  // static_assert(BlockSize == 0 || BlockSize >= (2 * sizeof(T) +
  // sizeof(header)),
  //     "BlockSize should larger than sizeof(T) + O(1)");

  InsertBag() : heap(BlockSize) {}
  InsertBag(InsertBag&& o) : heap(BlockSize) {
    std::swap(heap, o.heap);
    std::swap(heads, o.heads);
  }

  InsertBag& operator=(InsertBag&& o) {
    std::swap(heap, o.heap);
    std::swap(heads, o.heads);
    return *this;
  }

  InsertBag(const InsertBag&)            = delete;
  InsertBag& operator=(const InsertBag&) = delete;

  ~InsertBag() { destruct_parallel(); }

  void clear() { destruct_parallel(); }

  void clear_serial() { destruct_serial(); }

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

  local_iterator local_begin() {
    return local_iterator(&heads, galois::substrate::ThreadPool::getTID());
  }
  local_iterator local_end() {
    return local_iterator(&heads, galois::substrate::ThreadPool::getTID() + 1);
  }

  bool empty() const {
    for (unsigned x = 0; x < heads.size(); ++x) {
      header* h = heads.getRemote(x)->first;
      if (h)
        return false;
    }
    return true;
  }
  //! Thread safe bag insertion
  template <typename... Args>
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

  template <typename... Args>
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
  template <typename ItemTy>
  reference push(ItemTy&& val) {
    return emplace(std::forward<ItemTy>(val));
  }

  //! Thread safe bag insertion
  template <typename ItemTy>
  reference push_back(ItemTy&& val) {
    return emplace(std::forward<ItemTy>(val));
  }
};

} // namespace galois

#endif
