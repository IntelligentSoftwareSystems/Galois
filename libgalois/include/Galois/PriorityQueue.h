/** TODO -*- C++ -*-
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
 * @author <ahassaan@ices.utexas.edu>
 */

#ifndef GALOIS_PRIORITYQUEUE_H
#define GALOIS_PRIORITYQUEUE_H

#include "Galois/Substrate/PaddedLock.h"
#include "Galois/Substrate/CompilerSpecific.h"

#include <vector>
#include <algorithm>
#include <set>

#include "Galois/Mem.h"

namespace galois {

/**
 * Thread-safe ordered set. Faster than STL heap operations (about 10%-15% faster on serially) and
 * can use scalable allocation, e.g., {@link FixedSizeAllocator}.
 */
template <typename T, typename Cmp=std::less<T>, typename Alloc=galois::FixedSizeAllocator<T> >
class ThreadSafeOrderedSet {
  typedef std::set<T, Cmp, Alloc> Set;

public:
  typedef Set container_type;
  typedef typename container_type::value_type value_type;
  typedef typename container_type::reference reference;
  typedef typename container_type::const_reference const_reference;
  typedef typename container_type::pointer pointer;
  typedef typename container_type::size_type size_type;
  typedef typename container_type::const_iterator iterator;
  typedef typename container_type::const_iterator const_iterator;
  typedef typename container_type::const_reverse_iterator reverse_iterator;
  typedef typename container_type::const_reverse_iterator const_reverse_iterator;
  typedef galois::Substrate::SimpleLock Lock_ty;

private:
  GALOIS_ATTRIBUTE_ALIGN_CACHE_LINE Lock_ty mutex;
  Set orderedSet;

public:
  template<typename _T, typename _Cmp=std::less<_T>, typename _Alloc=galois::FixedSizeAllocator<_T> >
  using retype = ThreadSafeOrderedSet<_T, _Cmp, _Alloc>; // FIXME: loses Alloc and Cmp types

  explicit ThreadSafeOrderedSet(const Cmp& cmp=Cmp(), const Alloc& alloc=Alloc()):
      orderedSet(cmp, alloc)
  {}

  template <typename Iter>
  ThreadSafeOrderedSet(Iter b, Iter e, const Cmp& cmp=Cmp(), const Alloc& alloc=Alloc())
    : orderedSet(cmp, alloc)
  {
    for (; b != e; ++b) {
      orderedSet.insert(*b);
    }
  }

  bool empty() const {
    mutex.lock();
    bool ret = orderedSet.empty();
    mutex.unlock();

    return ret;
  }

  size_type size() const {
    mutex.lock();
    size_type sz =  orderedSet.size();
    mutex.unlock();

    return sz;
  }

  value_type top() const {
    mutex.lock();
    value_type x = *orderedSet.begin();
    mutex.unlock();
    return x;
  }

  bool find(const value_type& x) const {
    mutex.lock();
    bool ret = (orderedSet.find(x) != orderedSet.end());
    mutex.unlock();
    return ret;
  }

  bool push(const value_type& x) {
    mutex.lock();
    auto p = orderedSet.insert(x);
    mutex.unlock();
    return p.second;
  }

  value_type pop() {
    mutex.lock();
    value_type x = *orderedSet.begin();
    orderedSet.erase(orderedSet.begin());
    mutex.unlock();
    return x;
  }

  bool remove(const value_type& x) {
    mutex.lock();
    bool ret = false;

    if (x == *orderedSet.begin()) {
      orderedSet.erase(orderedSet.begin());
      ret = true;
    } else {
      size_type s = orderedSet.erase(x);
      ret = (s > 0);
    }
    mutex.unlock();

    return ret;
  }

  void clear () {
    mutex.lock ();
    orderedSet.clear ();
    mutex.unlock ();
  }

  const_iterator begin() const { return orderedSet.begin(); }
  const_iterator end() const { return orderedSet.end(); }
};

template <typename T, typename Cmp=std::less<T>, typename Cont=std::vector<T, Runtime::Pow_2_BlockAllocator<T> > >
class MinHeap {
public:
  typedef Runtime::Pow_2_BlockAllocator<T> alloc_type;
  typedef Cont container_type;

  typedef typename container_type::value_type value_type;
  typedef typename container_type::reference reference;
  typedef typename container_type::const_reference const_reference;
  typedef typename container_type::pointer pointer;
  typedef typename container_type::size_type size_type;
  typedef typename container_type::const_iterator iterator;
  typedef typename container_type::const_iterator const_iterator;
  typedef typename container_type::const_reverse_iterator reverse_iterator;
  typedef typename container_type::const_reverse_iterator const_reverse_iterator;
  // typedef typename container_type::const_iterator iterator;

protected:
  struct RevCmp {
    Cmp cmp;

    explicit RevCmp(const Cmp& cmp): cmp(cmp) {}

    bool operator()(const T& left, const T& right) const {
      return cmp(right, left);
    }
  };

  Cont container;
  RevCmp revCmp;

  const_reference top_internal() const {
    assert(!container.empty());
    return container.front();
  }
  
  value_type pop_internal() {
    assert(!container.empty());
    std::pop_heap(container.begin(), container.end(), revCmp);

    value_type x = container.back();
    container.pop_back();

    return x;
  }

public:
  explicit MinHeap(const Cmp& cmp=Cmp(), const Cont& container=Cont())
    : container(container), revCmp(cmp)
  {}

  template <typename Iter>
  MinHeap(Iter b, Iter e, const Cmp& cmp=Cmp())
    : container(b, e), revCmp(cmp)
  {
    std::make_heap(container.begin(), container.end());
  }
  
  bool empty() const {
    return container.empty();
  }

  size_type size() const {
   return container.size();
  }

  const_reference top() const { 
    return container.front();
  }

  void push(const value_type& x) {
    container.push_back(x);
    std::push_heap(container.begin(), container.end(), revCmp);
  }

  value_type pop() {
    assert(!container.empty());
    std::pop_heap(container.begin(), container.end(), revCmp);

    value_type x = container.back();
    container.pop_back();
    return x;
  }

  bool remove(const value_type& x) {
    bool ret = false;

    // TODO: write a better remove method
    if (x == top()) {
      pop();
      ret = true;
    } else {
      typename container_type::iterator nend = 
        std::remove(container.begin(), container.end(), x);

      ret = (nend != container.end());
      container.erase(nend, container.end());

      std::make_heap(container.begin(), container.end(), revCmp);
    }

    return ret;
  }

  bool find(const value_type& x) const {
    return (std::find(begin(), end(), x) != end());
  }

  void clear () { 
    container.clear ();
  }

  const_iterator begin() const { return container.begin(); }
  const_iterator end() const { return container.end(); }

  void reserve(size_type s) {
    container.reserve(s);
  }
};

/**
 * Thread-safe min heap.
 */
template <typename T, typename Cmp=std::less<T> >
class ThreadSafeMinHeap {
public:
  typedef MinHeap<T, Cmp> container_type;

  typedef typename container_type::value_type value_type;
  typedef typename container_type::reference reference;
  typedef typename container_type::const_reference const_reference;
  typedef typename container_type::pointer pointer;
  typedef typename container_type::size_type size_type;
  typedef typename container_type::const_iterator iterator;
  typedef typename container_type::const_iterator const_iterator;
  typedef typename container_type::const_reverse_iterator reverse_iterator;
  typedef typename container_type::const_reverse_iterator const_reverse_iterator;

protected:
  typedef galois::Substrate::SimpleLock Lock_ty;

  GALOIS_ATTRIBUTE_ALIGN_CACHE_LINE Lock_ty mutex;
  container_type heap;

public:
  explicit ThreadSafeMinHeap(const Cmp& cmp=Cmp())
    : heap(cmp)
  {}

  template <typename Iter>
  ThreadSafeMinHeap(Iter b, Iter e, const Cmp& cmp=Cmp())
    : heap(b, e, cmp)
  {}
  
  
  bool empty() const {
    mutex.lock();
    bool ret = heap.empty();
    mutex.unlock();

    return ret;
  }

  size_type size() const {
    mutex.lock();
    size_type sz =  heap.size();
    mutex.unlock();

    return sz;
  }

  // can't return a reference, because the reference may not be pointing
  // to a valid location due to vector doubling in size and moving to 
  // another memory location
  value_type top() const {
    mutex.lock();
    value_type x = heap.top();
    mutex.unlock();

    return x;
  }

  void push(const value_type& x) {
    mutex.lock();
    heap.push(x);
    mutex.unlock();
  }

  value_type pop() {
    mutex.lock();
    value_type x = heap.pop();
    mutex.unlock();
    return x;
  }

  bool remove(const value_type& x) {
    // TODO: write a better remove method
    mutex.lock();
    bool ret = heap.remove(x);
    mutex.unlock();

    return ret;
  }

  bool find(const value_type& x) const {
    mutex.lock();
    bool ret = heap.find(x);
    mutex.unlock();

    return ret;
  }

  void clear () { 
    mutex.lock ();
    heap.clear ();
    mutex.unlock ();
  }

  // TODO: can't use in parallel context
  const_iterator begin() const { return heap.begin(); }
  const_iterator end() const { return heap.end(); }

  void reserve(size_type s) {
    heap.reserve(s);
  }
};

}

#endif
