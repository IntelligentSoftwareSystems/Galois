/** deque like structure with scalable allocator usage -*- C++ -*-
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
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#ifndef GALOIS_GDEQUE_H
#define GALOIS_GDEQUE_H

#include "Galois/Runtime/mm/Mem.h"

#include "Galois/FixedSizeRing.h"

#include <iterator>

namespace Galois {

template < class T > 
class gdeque {

  struct Block : public FixedSizeRing<T> {
    Block* next;
    Block* prev;
    Block() :next(), prev() {}
  };

  Block* first;
  Block* last;
  unsigned num;

  GaloisRuntime::MM::FixedSizeAllocator heap;
  
  Block* alloc_block() {
    return new (heap.allocate(sizeof(Block))) Block();
  }

  void free_block(Block* b) {
    b->~Block();
    heap.deallocate(b);
  }

  void extend_first() {
    Block* b = alloc_block();
    b->next = first;
    if (b->next)
      b->next->prev = b;
    first = b;
    if (!last)
      last = b;
  }

  void extend_last() {
    Block* b = alloc_block();
    b->prev = last;
    if (b->prev)
      b->prev->next = b;
    last = b;
    if (!first)
      first = b;
  }

  void shrink_first() {
    Block* b = first;
    first = b->next;
    if (b->next)
      b->prev = 0;
    if (last == b)
      last = first;
    free_block(b);
  }

  void shrink_last() {
    Block* b = last;
    last = b->prev;
    if (b->prev)
      b->next = 0;
    if (first == b)
      first = last;
    free_block(b);
  }

public:
  typedef T value_type;

  gdeque() :first(), last(), num(), heap(sizeof(Block)) { }

  class iterator : public std::iterator<std::forward_iterator_tag, T> {
    Block* b;
    unsigned offset;

    void advance() {
      if (!b) return;
      ++offset;
      if (offset == b->size()) {
	b = b->next;
	offset = 0;
      }
    }

  public:
    iterator(Block* _b = 0, unsigned _off = 0) :b(_b), offset(_off) {}

    bool operator==(const iterator& rhs) const {
      return b == rhs.b && offset == rhs.offset;
    }

    bool operator!=(const iterator& rhs) const {
      return b != rhs.b || offset != rhs.offset;
    }

    T& operator*() const {
      return b->getAt(offset);
    }

    iterator& operator++() {
      advance();
      return *this;
    }

    iterator operator++(int) {
      iterator tmp(*this);
      advance();
      return tmp;
    }

    // iterator& operator--() {
    //   regress();
    //   return *this;
    // }

    // iterator operator--(int) {
    //   iterator tmp(*this);
    //   regress();
    //   return tmp;
    // }
  };

  iterator begin() const {
    return iterator(first);
  }

  iterator end() const {
    return iterator();
  }

  size_t size() const {
    return num;
  }

  bool empty() const {
    return num == 0;
  }

  value_type& front() {
    return first->getAt(0);
  }

  value_type& back() {
    return last->getAt(last->size() - 1);
  }

  void push_back(const value_type& v) {
    ++num;
    if (last && last->push_back(v))
      return;
    extend_last();
    last->push_back(v);
  }

  void push_front(const value_type& v) {
    ++num;
    if (first && first->push_front(v))
      return;
    extend_first();
    first->push_front(v);
  }

  void pop_back() {
    --num;
    last->pop_back();
    if (last->empty())
      shrink_last();
  }

  void pop_front() {
    --num;
    first->pop_front();
    if (first->empty())
      shrink_first();
  }

  void clear() {
    while (first) {
      first->clear();
      shrink_first();
    }
  }
};

}

#endif
