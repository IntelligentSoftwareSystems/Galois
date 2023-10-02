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

#ifndef GALOIS_GDEQUE_H
#define GALOIS_GDEQUE_H

#include "galois/config.h"
#include "galois/FixedSizeRing.h"
#include "galois/Mem.h"
#include "galois/TwoLevelIteratorA.h"

#include <boost/iterator/iterator_facade.hpp>
#include <boost/iterator/reverse_iterator.hpp>

#include <algorithm>
#include <utility>

namespace galois {

// Experimental random access iterator. Slower than old iterator for simple
// traversals, so disable for now
// #define _NEW_ITERATOR

//! Like std::deque but use Galois memory management functionality
template <typename T, unsigned ChunkSize = 64,
          typename ContainerTy = FixedSizeRing<T, ChunkSize>>
class gdeque {

protected:
  struct Block : ContainerTy {
    Block* next;
    Block* prev;

    Block() : next(), prev() {}

    template <typename InputIterator>
    Block(InputIterator first, InputIterator second)
        : ContainerTy(first, second), next(), prev() {}
  };

#ifdef _NEW_ITERATOR
  template <typename U>
  class outer_iterator
      : public boost::iterator_facade<outer_iterator<U>, U,
                                      boost::bidirectional_traversal_tag> {
    friend class boost::iterator_core_access;
    template <typename, unsigned, typename>
    friend class gdeque;
    Block* cur;
    Block* last;

    void increment() { cur = cur->next; }
    void decrement() {
      if (cur) {
        cur = cur->prev;
      } else {
        cur = last;
      }
    }

    template <typename OtherTy>
    bool equal(const outer_iterator<OtherTy>& o) const {
      return cur == o.cur;
    }

    U& dereference() const { return *cur; }

  public:
    outer_iterator(Block* b = 0, Block* l = 0) : cur(b), last(l) {}

    template <typename OtherTy>
    outer_iterator(const outer_iterator<OtherTy>& o)
        : cur(o.cur), last(o.last) {}
  };

  typedef typename Block::iterator inner_iterator;
  typedef typename Block::const_iterator const_inner_iterator;
#endif

  Block* first;

private:
  Block* last;
  unsigned num;

  //! [Example Fixed Size Allocator]
  galois::FixedSizeAllocator<Block> heap;

  template <typename... Args>
  Block* alloc_block(Args&&... args) {
    // Fixed size allocator can only allocate 1 object at a time of size
    // sizeof(Block). Argument to allocate is always 1.
    Block* b = heap.allocate(1);
    return new (b) Block(std::forward<Args>(args)...);
  }

  void free_block(Block* b) {
    b->~Block();
    heap.deallocate(b, 1);
  }
  //! [Example Fixed Size Allocator]

  bool precondition() const {
    return (num == 0 && first == NULL && last == NULL) ||
           (num > 0 && first != NULL && last != NULL);
  }

  Block* extend_first() {
    Block* b = alloc_block();
    b->next  = first;
    if (b->next)
      b->next->prev = b;
    first = b;
    if (!last)
      last = b;
    return b;
  }

  Block* extend_last() {
    Block* b = alloc_block();
    b->prev  = last;
    if (b->prev)
      b->prev->next = b;
    last = b;
    if (!first)
      first = b;
    return b;
  }

  void shrink(Block* b) {
    if (b->next)
      b->next->prev = b->prev;
    if (b->prev)
      b->prev->next = b->next;
    if (b == first)
      first = b->next;
    if (b == last)
      last = b->prev;
    free_block(b);
  }

  template <typename... Args>
  std::pair<Block*, typename Block::iterator>
  emplace(Block* b, typename Block::iterator ii, Args&&... args) {
    ++num;
    if (!b) {
      // gdeque is empty or iteration == end
      b = last;
      if (!b || b->full())
        b = extend_last();
      ii = b->end();
    } else if (b == first && ii == b->begin()) {
      // iteration == begin
      b = first;
      if (!b || b->full())
        b = extend_first();
      ii = b->begin();
    } else if (b->full()) {
      auto d   = std::distance(ii, b->end());
      Block* n = alloc_block(std::make_move_iterator(ii),
                             std::make_move_iterator(b->end()));
      for (; d > 0; --d)
        b->pop_back();
      ii      = b->end();
      n->next = b->next;
      n->prev = b;
      b->next = n;
      if (b == last)
        last = n;
    }
    unsigned boff = std::distance(b->begin(), ii);
    b->emplace(ii, std::forward<Args>(args)...);
    return std::make_pair(b, b->begin() + boff);
  }

public:
#ifdef _NEW_ITERATOR
  typedef galois::TwoLevelIteratorA<outer_iterator<Block>, inner_iterator,
                                    std::random_access_iterator_tag,
                                    GetBegin<Block>, GetEnd<Block>>
      iterator;
  typedef galois::TwoLevelIteratorA<outer_iterator<const Block>,
                                    const_inner_iterator,
                                    std::random_access_iterator_tag,
                                    GetBegin<const Block>, GetEnd<const Block>>
      const_iterator;
#endif
#ifndef _NEW_ITERATOR
  template <typename U>
  struct Iterator
      : public boost::iterator_facade<Iterator<U>, U,
                                      boost::bidirectional_traversal_tag> {
    friend class boost::iterator_core_access;

    Block* b;
    Block* last;
    unsigned offset;

  private:
    void increment() {
      ++offset;
      if (offset == b->size()) {
        b      = b->next;
        offset = 0;
      }
    }

    void decrement() {
      if (!b) {
        b      = last;
        offset = b->size() - 1;
        return;
      } else if (offset == 0) {
        b      = b->prev;
        offset = b->size() - 1;
      } else {
        --offset;
      }
    }

    template <typename OtherTy>
    bool equal(const Iterator<OtherTy>& o) const {
      return b == o.b && offset == o.offset;
    }

    U& dereference() const { return b->getAt(offset); }

  public:
    Iterator(Block* _b = 0, Block* _l = 0, unsigned _off = 0)
        : b(_b), last(_l), offset(_off) {}

    template <typename OtherTy>
    Iterator(const Iterator<OtherTy>& o)
        : b(o.b), last(o.last), offset(o.offset) {}
  };
  typedef Iterator<T> iterator;
  typedef Iterator<const T> const_iterator;
#endif

  typedef boost::reverse_iterator<iterator> reverse_iterator;
  typedef boost::reverse_iterator<const_iterator> const_reverse_iterator;
  typedef typename iterator::value_type value_type;
  typedef typename iterator::pointer pointer;
  typedef typename iterator::reference reference;
  typedef typename const_iterator::reference const_reference;
  typedef typename iterator::difference_type difference_type;
  typedef size_t size_type;

  gdeque() : first(), last(), num(), heap() {}

  gdeque(gdeque&& o) : first(), last(), num(), heap() {
    std::swap(first, o.first);
    std::swap(last, o.last);
    std::swap(num, o.num);
  }

  gdeque& operator=(gdeque&& o) {
    std::swap(first, o.first);
    std::swap(last, o.last);
    std::swap(num, o.num);
    return *this;
  }

  gdeque(const gdeque&)            = delete;
  gdeque& operator=(const gdeque&) = delete;

  ~gdeque() { clear(); }

  iterator begin() {
    assert(precondition());

#ifdef _NEW_ITERATOR
    return iterator{outer_iterator<Block>{first, last},
                    outer_iterator<Block>{nullptr, last},
                    outer_iterator<Block>{first, last}, GetBegin<Block>{},
                    GetEnd<Block>{}};
#else
    return iterator{first, last, 0};
#endif
  }

  iterator end() {
    assert(precondition());
#ifdef _NEW_ITERATOR
    return iterator{outer_iterator<Block>{first, last},
                    outer_iterator<Block>{nullptr, last},
                    outer_iterator<Block>{nullptr, last}, GetBegin<Block>{},
                    GetEnd<Block>{}};
#else
    return iterator{nullptr, last, 0};
#endif
  }

  const_iterator begin() const {
    assert(precondition());

#ifdef _NEW_ITERATOR
    return const_iterator{outer_iterator<const Block>{first, last},
                          outer_iterator<const Block>{nullptr, last},
                          outer_iterator<const Block>{first, last},
                          GetBegin<const Block>{},
                          GetEnd<const Block, const_inner_iterator>{}};
#else
    return const_iterator{first, last, 0};
#endif
  }

  const_iterator end() const {
#ifdef _NEW_ITERATOR
    return const_iterator{outer_iterator<const Block>{first, last},
                          outer_iterator<const Block>{nullptr, last},
                          outer_iterator<const Block>{nullptr, last},
                          GetBegin<const Block>{},
                          GetEnd<const Block, const_inner_iterator>{}};
#else
    return const_iterator{nullptr, last, 0};
#endif
  }

  reverse_iterator rbegin() { return reverse_iterator{end()}; }

  reverse_iterator rend() { return reverse_iterator{begin()}; }

  const_reverse_iterator rbegin() const {
    return const_reverse_iterator{end()};
  }

  const_reverse_iterator rend() const {
    return const_reverse_iterator{begin()};
  }

  size_t size() const {
    assert(precondition());
    return num;
  }

  bool empty() const {
    assert(precondition());
    return num == 0;
  }

  reference front() {
    assert(!empty());
    return first->front();
  }

  const_reference front() const {
    assert(!empty());
    return first->front();
  }

  reference back() {
    assert(!empty());
    return last->back();
  }

  const_reference back() const {
    assert(!empty());
    return last->back();
  }

  void pop_back() {
    assert(!empty());
    --num;
    last->pop_back();
    if (last->empty())
      shrink(last);
  }

  void pop_front() {
    assert(!empty());
    --num;
    first->pop_front();
    if (first->empty())
      shrink(first);
  }

  void clear() {
    assert(precondition());
    Block* b = first;
    while (b) {
      b->clear();
      Block* old = b;
      b          = b->next;
      free_block(old);
    }
    first = last = NULL;
    num          = 0;
  }

  //! Invalidates pointers
  template <typename... Args>
  iterator emplace(iterator pos, Args&&... args) {
#ifdef _NEW_ITERATOR
    Block* b          = pos.get_outer_reference().cur;
    inner_iterator ii = pos.get_inner_reference();
#else
    Block* b = pos.b;
    typename Block::iterator ii;
    if (b)
      ii = b->begin() + pos.offset;
#endif
    auto p = emplace(b, ii, std::forward<Args>(args)...);
#ifdef _NEW_ITERATOR
    return iterator{outer_iterator<Block>{first, last},
                    outer_iterator<Block>{nullptr, last},
                    outer_iterator<Block>{p.first, last},
                    p.second,
                    GetBegin<Block>{},
                    GetEnd<Block>{}};
#else
    return iterator(p.first, last, std::distance(p.first->begin(), p.second));
#endif
  }

  iterator erase(iterator pos) {
    GALOIS_DIE("not yet implemented");
    return pos;
  }

#ifdef _NEW_ITERATOR
  //! Not truly constant time
  reference operator[](size_t x) {
    if (x == 0)
      return front();
    else if (x == num)
      return back();
    auto ii = begin();
    std::advance(ii, x);
    return *ii;
  }

  //! Not truly constant time
  const_reference operator[](size_t x) const {
    if (x == 0)
      return front();
    else if (x == num)
      return back();
    auto ii = begin();
    std::advance(ii, x);
    return *ii;
  }
#endif

  template <typename... Args>
  void emplace_back(Args&&... args) {
    assert(precondition());
    ++num;
    if (!last || last->full())
      extend_last();
#ifndef NDEBUG
    pointer p = last->emplace_back(std::forward<Args>(args)...);
    assert(p);
#else
    last->emplace_back(std::forward<Args>(args)...);
#endif
  }

  template <typename ValueTy>
  void push_back(ValueTy&& v) {
    emplace_back(std::forward<ValueTy>(v));
  }

  template <typename... Args>
  void emplace_front(Args&&... args) {
    assert(precondition());
    ++num;
    if (!first || first->full())
      extend_first();
    pointer p = first->emplace_front(std::forward<Args>(args)...);
    assert(p);
  }

  template <typename ValueTy>
  void push_front(ValueTy&& v) {
    emplace_front(std::forward<ValueTy>(v));
  }
};

#undef _NEW_ITERATOR
} // namespace galois
#endif
