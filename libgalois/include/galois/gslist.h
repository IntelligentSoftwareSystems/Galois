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

#ifndef GALOIS_GSLIST_H
#define GALOIS_GSLIST_H

#include <type_traits>

#include <boost/iterator/iterator_facade.hpp>
#include <boost/mpl/if.hpp>

#include "galois/config.h"
#include "galois/FixedSizeRing.h"
#include "galois/TwoLevelIteratorA.h"

namespace galois {

template <typename T, int ChunkSize, bool Concurrent>
class gslist_base {
public:
  //! Tag for methods that depend on user to deallocate memory, although gslist
  //! will destroy elements
  struct promise_to_dealloc {};

private:
  typedef typename boost::mpl::if_c<Concurrent,
                                    ConcurrentFixedSizeBag<T, ChunkSize>,
                                    FixedSizeBag<T, ChunkSize>>::type Ring;

  struct Block : public Ring {
    Block* next;
    Block() : next() {}
  };

  template <typename U>
  class outer_iterator
      : public boost::iterator_facade<outer_iterator<U>, U,
                                      boost::forward_traversal_tag> {
    friend class boost::iterator_core_access;
    U* cur;

    void increment() { cur = cur->next; }

    template <typename OtherTy>
    bool equal(const outer_iterator<OtherTy>& o) const {
      return cur == o.cur;
    }

    U& dereference() const { return *cur; }

  public:
    outer_iterator(U* c = 0) : cur(c) {}

    template <typename OtherTy>
    outer_iterator(const outer_iterator<OtherTy>& o) : cur(o.cur) {}
  };

  typedef
      typename boost::mpl::if_c<Concurrent, std::atomic<Block*>, Block*>::type
          First;

  First first;

  template <typename HeapTy>
  Block* alloc_block(HeapTy& heap) {
    return new (heap.allocate(sizeof(Block))) Block();
  }

  template <typename HeapTy>
  void free_block(HeapTy& heap, Block* b) {
    b->~Block();
    heap.deallocate(b);
  }

  void free_block(promise_to_dealloc, Block* b) { b->~Block(); }

  template <typename HeapTy, bool C = Concurrent>
  auto extend_first(HeapTy& heap) -> typename std::enable_if<C>::type {
    Block* b = alloc_block(heap);
    while (true) {
      Block* f = first.load(std::memory_order_relaxed);
      b->next  = f;
      if (first.compare_exchange_weak(f, b))
        return;
    }
  }

  template <typename HeapTy, bool C = Concurrent>
  auto extend_first(HeapTy& heap) -> typename std::enable_if<!C>::type {
    Block* b = alloc_block(heap);
    b->next  = first;
    first    = b;
  }

  Block* get_first() {
    Block* b = first;
    return b;
  }

  const Block* get_first() const {
    Block* b = first;
    return b;
  }

  template <typename U, bool C = Concurrent>
  auto shrink_first(Block* old_first, U&& arg) ->
      typename std::enable_if<C>::type {
    if (first.compare_exchange_strong(old_first, old_first->next)) {
      // old_first->clear();
      free_block(std::forward<U>(arg), old_first);
    }
  }

  template <typename U, bool C = Concurrent>
  auto shrink_first(Block* old_first, U&& arg) ->
      typename std::enable_if<!C>::type {
    if (first != old_first)
      return;
    first = old_first->next;
    // old_first->clear();
    free_block(std::forward<U>(arg), old_first);
  }

  template <typename U>
  void _clear(U&& arg) {
    Block* b = get_first();
    while (b) {
      shrink_first(b, std::forward<U>(arg));
      b = get_first();
    }
  }

  template <typename U>
  bool _pop_front(U&& arg) {
    while (true) {
      Block* b = get_first();
      if (!b)
        return false;
      if (b->pop_front())
        return true;

      shrink_first(b, std::forward<U>(arg));
    }
  }

public:
  //! External allocator must be able to allocate this type
  typedef Block block_type;
  typedef T value_type;
  typedef galois::TwoLevelIteratorA<outer_iterator<Block>,
                                    typename Block::iterator,
                                    std::forward_iterator_tag, GetBegin, GetEnd>
      iterator;
  typedef galois::TwoLevelIteratorA<outer_iterator<const Block>,
                                    typename Block::const_iterator,
                                    std::forward_iterator_tag, GetBegin, GetEnd>
      const_iterator;

  gslist_base() : first(0) {}

  gslist_base(const gslist_base&)            = delete;
  gslist_base& operator=(const gslist_base&) = delete;

  gslist_base(gslist_base&& other) : first(0) { *this = std::move(other); }

  gslist_base& operator=(gslist_base&& o) {
    Block* m_first = first;
    Block* o_first = o.first;
    first          = o_first;
    o.first        = m_first;
    return *this;
  }

  ~gslist_base() {
    _clear(promise_to_dealloc());
    // assert(empty() && "Memory leak if gslist is not empty before
    // destruction");
  }

  iterator begin() {
    return galois::make_two_level_iterator(outer_iterator<Block>(get_first()),
                                           outer_iterator<Block>(nullptr))
        .first;
  }

  iterator end() {
    return galois::make_two_level_iterator(outer_iterator<Block>(get_first()),
                                           outer_iterator<Block>(nullptr))
        .second;
  }

  const_iterator begin() const {
    return galois::make_two_level_iterator(
               outer_iterator<const Block>(get_first()),
               outer_iterator<const Block>(nullptr))
        .first;
  }

  const_iterator end() const {
    return galois::make_two_level_iterator(
               outer_iterator<const Block>(get_first()),
               outer_iterator<const Block>(nullptr))
        .second;
  }

  bool empty() const {
    return first == NULL || (get_first()->empty() && get_first()->next == NULL);
  }

  value_type& front() { return get_first()->front(); }

  const value_type& front() const { return get_first()->front(); }

  template <typename HeapTy, typename... Args, bool C = Concurrent>
  auto emplace_front(HeapTy& heap, Args&&... args) ->
      typename std::enable_if<!C>::type {
    if (!first || first->full())
      extend_first(heap);
    first->emplace_front(std::forward<Args>(args)...);
  }

  template <typename HeapTy, bool C = Concurrent>
  auto push_front(HeapTy& heap, const value_type& v) ->
      typename std::enable_if<C>::type {
    while (true) {
      Block* b = get_first();
      if (b && b->push_front(v))
        return;
      extend_first(heap);
    }
  }

  template <typename HeapTy, typename ValueTy, bool C = Concurrent>
  auto push_front(HeapTy& heap, ValueTy&& v) ->
      typename std::enable_if<!C>::type {
    emplace_front(heap, std::forward<ValueTy>(v));
  }

  //! Returns true if something was popped
  template <typename HeapTy>
  bool pop_front(HeapTy& heap) {
    return _pop_front(heap);
  }

  //! Returns true if something was popped
  bool pop_front(promise_to_dealloc) {
    return _pop_front(promise_to_dealloc());
  }

  template <typename HeapTy>
  void clear(HeapTy& heap) {
    _clear(heap);
  }

  void clear(promise_to_dealloc) { _clear(promise_to_dealloc()); }
};

/**
 * Singly linked list. To conserve space, allocator is maintained external to
 * the list.
 */
template <typename T, unsigned chunksize = 16>
using gslist = gslist_base<T, chunksize, false>;

/**
 * Concurrent linked list. To conserve space, allocator is maintained external
 * to the list. Iteration order is unspecified.
 */
template <typename T, unsigned chunksize = 16>
using concurrent_gslist = gslist_base<T, chunksize, true>;

} // namespace galois
#endif
