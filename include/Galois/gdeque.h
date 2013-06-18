/** deque like structure with scalable allocator usage -*- C++ -*-
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
 * @author Andrew Lenharth <andrew@lenharth.org>
 */
#ifndef GALOIS_GDEQUE_H
#define GALOIS_GDEQUE_H

#include "Galois/config.h"
#include "Galois/FixedSizeRing.h"
#include "Galois/Runtime/mm/Mem.h"

#include <boost/iterator/iterator_facade.hpp>

#include GALOIS_CXX11_STD_HEADER(algorithm)
#include GALOIS_CXX11_STD_HEADER(utility)

namespace Galois {

//! Like std::deque but use Galois memory management functionality
template<typename T, unsigned ChunkSize=64, typename ContainerTy=FixedSizeRing<T, ChunkSize> > 
class gdeque: private boost::noncopyable {
protected:
  struct Block: ContainerTy {
    Block* next;
    Block* prev;
    Block(): next(), prev() {}
  };

  Block* first;

private:
  Block* last;
  unsigned num;

  Galois::Runtime::MM::FixedSizeAllocator heap;
  
  Block* alloc_block() {
    return new (heap.allocate(sizeof(Block))) Block();
  }

  bool precondition() const {
    return (num == 0 && first == NULL && last == NULL)
      || (num > 0 && first != NULL && last != NULL);
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
      b->next->prev = 0;
    else
      last = 0;
    free_block(b);
  }

  void shrink_last() {
    Block* b = last;
    last = b->prev;
    if (b->prev)
      b->prev->next = 0;
    else
      first = 0;
    free_block(b);
  }

public:
  template<typename U>
  struct Iterator: public boost::iterator_facade<Iterator<U>, U, boost::forward_traversal_tag> {
    friend class boost::iterator_core_access;

    Block* b;
    unsigned offset;

  private:
    void increment() {
      if (!b) return;
      ++offset;
      if (offset == b->size()) {
	b = b->next;
	offset = 0;
      }
    }

    template<typename OtherTy>
    bool equal(const Iterator<OtherTy>& o) const { return b == o.b && offset == o.offset; }

    U& dereference() const { return b->getAt(offset); }

  public:
    Iterator(Block* _b = 0, unsigned _off = 0) :b(_b), offset(_off) { }
    
    template<typename OtherTy>
    Iterator(const Iterator<OtherTy>& o): b(o.b), offset(o.offset) { }
  };

  typedef T value_type;
  typedef T* pointer;
  typedef T& reference;
  typedef const T& const_reference;
  typedef Iterator<T> iterator;
  typedef Iterator<const T> const_iterator;

  gdeque(): first(), last(), num(), heap(sizeof(Block)) { }

  ~gdeque() { clear(); }

  iterator begin() { assert(precondition()); return iterator(first); }
  iterator end() { assert(precondition()); return iterator(); }
  const_iterator begin() const { assert(precondition()); return const_iterator(first); }
  const_iterator end() const { assert(precondition()); return const_iterator(); }

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
      shrink_last();
  }

  void pop_front() {
    assert(!empty());
    --num;
    first->pop_front();
    if (first->empty())
      shrink_first();
  }

  void clear() {
    assert(precondition());
    Block* b = first;
    while (b) {
      b->clear();
      Block* old = b;
      b = b->next;
      free_block(old);
    }
    first = last = NULL;
    num = 0;
  }

  //FIXME: support alternate insert locations
  iterator insert(iterator position, size_t n, const value_type& val) {
    assert(position == end());
    if (!n)
      return end();

    push_back(val);
    iterator retval = iterator(last, last->size()-1);
    for (size_t x = 1; x < n; ++x)
      push_back(val);
    return retval;
  }

  template<typename... Args>
  void emplace_back(Args&&... args) {
    assert(precondition());
    ++num;
    if (last && last->emplace_back(std::forward<Args>(args)...))
      return;
    extend_last();
    pointer p = last->emplace_back(std::forward<Args>(args)...);
    assert(p);
  }

  void push_back(value_type&& v) {
    emplace_back(std::move(v));
  }

  void push_back(const value_type& v) {
    emplace_back(v);
  }

  template<typename... Args>
  void emplace_front(Args&&... args) {
    assert(precondition());
    ++num;
    if (first && first->emplace_front(std::forward<Args>(args)...))
      return;
    extend_first();
    pointer p = first->emplace_front(std::forward<Args>(args)...);
    assert(p);
  }

  void push_front(value_type&& v) {
    emplace_front(std::move(v));
  }

  void push_front(const value_type& v) {
    emplace_front(v);
  }
};

}
#endif
