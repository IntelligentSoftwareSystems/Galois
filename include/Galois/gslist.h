/** Low-space overhead list -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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
 * @section Description
 *
 * Container for when you want to minimize meta-data overhead but still
 * want a custom allocator.
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#ifndef GALOIS_GSLIST_H
#define GALOIS_GSLIST_H

#include "Galois/Runtime/mm/Mem.h"

#include "Galois/FixedSizeRing.h"

#include <iterator>

namespace Galois {

//! Singly linked list. To conserve space, allocator is maintained
//! external to the list. 
template<typename T, int ChunkSize=16> 
class gslist {

  struct Block: public FixedSizeRing<T,ChunkSize> {
    Block* next;
    Block(): next() {}
  };

  Block* first;
  
  template<typename HeapTy>
  Block* alloc_block(HeapTy& heap) {
    return new (heap.allocate(sizeof(Block))) Block();
  }

  template<typename HeapTy>
  void free_block(HeapTy& heap, Block* b) {
    b->~Block();
    heap.deallocate(b);
  }

  template<typename HeapTy>
  void extend_first(HeapTy& heap) {
    Block* b = alloc_block(heap);
    b->next = first;
    first = b;
  }

  template<typename HeapTy>
  void shrink_first(HeapTy& heap) {
    Block* b = first;
    first = b->next;
    free_block(heap, b);
  }

public:
  //! External allocator must be able to allocate this type
  typedef Block block_type;
  typedef T value_type;

  gslist(): first() { }

  ~gslist() {
    assert(empty() && "Memory leak if gslist is not empty before destruction");
  }

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
    iterator(Block* _b = 0, unsigned _off = 0): b(_b), offset(_off) {}

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
  };

  iterator begin() const {
    return iterator(first);
  }

  iterator end() const {
    return iterator();
  }

  bool empty() const {
    return first == NULL;
  }

  value_type& front() {
    return first->front();
  }

  template<typename HeapTy>
  void push_front(HeapTy& heap, const value_type& v) {
    if (first && first->push_front(v))
      return;
    extend_first(heap);
    first->push_front(v);
  }

  template<typename HeapTy>
  void pop_front(HeapTy& heap) {
    first->pop_front();
    if (first->empty())
      shrink_first(heap);
  }

  template<typename HeapTy>
  void clear(HeapTy& heap) {
    while (first) {
      first->clear();
      shrink_first(heap);
    }
  }
};

}

#endif
