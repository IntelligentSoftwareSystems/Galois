/** TODO -*- C++ -*-
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
 * @section Description
 *
 * TODO 
 *
 * @author <ahassaan@ices.utexas.edu>
 */

#ifndef GALOIS_PRIORITY_QUEUE
#define GALOIS_PRIORITY_QUEUE

#include "Galois/Runtime/ll/PaddedLock.h"
#include "Galois/Runtime/ll/CompilerSpecific.h"

#include <vector>
#include <algorithm>
#include <set>

#include "Galois/Mem.h"

namespace Galois {

/**
 * Thread-safe ordered set. Faster than STL heap operations (about 10%-15% faster on serially) and
 * can use scalable allocation, e.g., {@link GFixedAllocator}.
 */
template <typename T, typename Cmp=std::less<T>, typename Alloc=Galois::GFixedAllocator<T> >
class ThreadSafeOrderedSet {
  using Set = std::set<T, Cmp, Alloc>;

public:
  using container_type = Set;
  using value_type = typename container_type::value_type;
  using reference = typename container_type::reference;
  using const_reference = typename container_type::const_reference;
  using pointer = typename container_type::pointer;
  using size_type = typename container_type::size_type;
  using iterator = typename container_type::const_iterator;
  using const_iterator = typename container_type::const_iterator;
  using reverse_iterator = typename container_type::const_reverse_iterator;
  using const_reverse_iterator = typename container_type::const_reverse_iterator;
  using Lock_ty = Galois::Runtime::LL::SimpleLock<true>;

private:
  GALOIS_ATTRIBUTE_ALIGN_CACHE_LINE Lock_ty mutex;
  Set orderedSet;

public:
  explicit ThreadSafeOrderedSet (const Cmp& cmp=Cmp (), const Alloc& alloc=Alloc ())
    :
      mutex (),
      orderedSet (cmp, alloc)
  {}

  template <typename Iter>
  ThreadSafeOrderedSet (Iter b, Iter e, const Cmp& cmp=Cmp (), const Alloc& alloc=Alloc ())
    :
      mutex (),
      orderedSet (cmp, alloc)
  {
    for (; b != e; ++b) {
      orderedSet.insert (*b);
    }
  }

  bool empty () const {
    mutex.lock ();
      bool ret = orderedSet.empty ();
    mutex.unlock ();

    return ret;
  }

  size_type size () const {
    mutex.lock ();
      size_type sz =  orderedSet.size ();
    mutex.unlock ();

    return sz;
  }

  value_type top () const {
    mutex.lock ();
      value_type x = *orderedSet.begin ();
    mutex.unlock ();
    return x;
  }

  bool find (const value_type& x) const {
    mutex.lock ();
      bool ret = (orderedSet.find (x) != orderedSet.end ());
    mutex.unlock ();
    return ret;
  }

  void push (const value_type& x) {
    mutex.lock ();
      orderedSet.insert (x);
    mutex.unlock ();
  }

  value_type pop () {
    mutex.lock ();
      value_type x = *orderedSet.begin ();
      orderedSet.erase (orderedSet.begin ());
    mutex.unlock ();
    return x;
  }

  bool remove (const value_type& x) {
    mutex.lock ();
      bool ret = false;

      if (x == *orderedSet.begin ()) {
        orderedSet.erase (orderedSet.begin ());
        ret = true;

      } else {
        size_type s = orderedSet.erase (x);
        ret = (s > 0);
      }
    mutex.unlock ();

    return ret;
  }

  const_iterator begin () const { return orderedSet.begin (); }
  const_iterator end () const { return orderedSet.end (); }

  
};

template <typename T, typename Cmp=std::less<T>, typename Cont=std::vector<T> >
class MinHeap {
public:
  using container_type = Cont;

  using value_type = typename container_type::value_type;
  using reference = typename container_type::reference;
  using const_reference = typename container_type::const_reference;
  using pointer = typename container_type::pointer;
  using size_type = typename container_type::size_type;
  using iterator = typename container_type::const_iterator;
  using const_iterator = typename container_type::const_iterator;
  using reverse_iterator = typename container_type::const_reverse_iterator;
  using const_reverse_iterator = typename container_type::const_reverse_iterator;
  // typedef typename container_type::const_iterator iterator;

protected:
  struct RevCmp {
    Cmp cmp;

    explicit RevCmp (const Cmp& cmp): cmp (cmp) {}

    bool operator () (const T& left, const T& right) const {
      return !cmp (left, right);
    }
  };

  Cont container;
  RevCmp revCmp;

  const_reference top_internal () const {
    assert (!container.empty ());
    return container.front ();
  }
  
  value_type pop_internal () {
    assert (!container.empty ());
    std::pop_heap (container.begin (), container.end (), revCmp);

    value_type x = container.back ();
    container.pop_back ();

    return x;
  }

public:

  explicit MinHeap (const Cmp& cmp=Cmp (), const Cont& container=Cont ())
    : container (container), revCmp (cmp)
  {}

  template <typename Iter>
  MinHeap (Iter b, Iter e, const Cmp& cmp=Cmp ())
    : container (b, e), revCmp (cmp)
  {
    std::make_heap (container.begin (), container.end ());
  }
  
  
  bool empty () const {
    return container.empty ();
  }

  size_type size () const {
   return container.size ();
  }

  const_reference top () const { 
    return container.front ();
  }

  void push (const value_type& x) {
    container.push_back (x);
    std::push_heap (container.begin (), container.end (), revCmp);
  }

  value_type pop () {
    assert (!container.empty ());
    std::pop_heap (container.begin (), container.end (), revCmp);

    value_type x = container.back ();
    container.pop_back ();
    return x;
  }

  bool remove (const value_type& x) {
    bool ret = false;

    // TODO: write a better remove method
    if (x == top ()) {
      pop ();
      ret = true;

    } else {
      typename container_type::iterator nend = 
        std::remove (container.begin (), container.end (), x);

      ret = (nend != container.end ());
      container.erase (nend, container.end ());

      std::make_heap (container.begin (), container.end (), revCmp);

    }

    return ret;
  }

  bool find (const value_type& x) const {
    return (std::find (begin (), end (), x) != end ());
  }

  const_iterator begin () const { return container.begin (); }
  const_iterator end () const { return container.end (); }

  void reserve (size_type s) {
    container.reserve (s);
  }
};

/**
 * Thread-safe min heap.
 */
template <typename T, typename Cmp=std::less<T> >
class ThreadSafeMinHeap {

public:
  using container_type = MinHeap<T, Cmp>;

  using value_type = typename container_type::value_type;
  using reference = typename container_type::reference;
  using const_reference = typename container_type::const_reference;
  using pointer = typename container_type::pointer;
  using size_type = typename container_type::size_type;
  using iterator = typename container_type::const_iterator;
  using const_iterator = typename container_type::const_iterator;
  using reverse_iterator = typename container_type::const_reverse_iterator;
  using const_reverse_iterator = typename container_type::const_reverse_iterator;

protected:

  using Lock_ty = Galois::Runtime::LL::SimpleLock<true>;

  GALOIS_ATTRIBUTE_ALIGN_CACHE_LINE Lock_ty mutex;
  container_type heap;

public:

  explicit ThreadSafeMinHeap (const Cmp& cmp=Cmp ())
    : mutex (), heap (cmp)
  {}

  template <typename Iter>
  ThreadSafeMinHeap (Iter b, Iter e, const Cmp& cmp=Cmp ())
    : mutex (), heap (b, e, cmp)
  {}
  
  
  bool empty () const {
    mutex.lock ();
      bool ret = heap.empty ();
    mutex.unlock ();

    return ret;
  }

  size_type size () const {
    mutex.lock ();
      size_type sz =  heap.size ();
    mutex.unlock ();

    return sz;
  }

  // can't return a reference, because the reference may not be pointing
  // to a valid location due to vector doubling in size and moving to 
  // another memory location
  value_type top () const {
    mutex.lock ();
      value_type x = heap.top ();
    mutex.unlock ();

    return x;
  }

  void push (const value_type& x) {
    mutex.lock ();
      heap.push (x);
    mutex.unlock ();
  }

  value_type pop () {
    mutex.lock ();
      value_type x = heap.pop ();
    mutex.unlock ();
    return x;
  }

  bool remove (const value_type& x) {
    bool ret = false;

    // TODO: write a better remove method
    mutex.lock ();
      ret = heap.remove (x);
    mutex.unlock ();

    return ret;
  }

  bool find (const value_type& x) const {
    mutex.lock ();
      bool ret = heap.find (x);
    mutex.unlock ();

    return ret;
  }

  // TODO: can't use in parallel context
  const_iterator begin () const { return heap.begin (); }
  const_iterator end () const { return heap.end (); }

  void reserve (size_type s) {
    heap.reserve (s);
  }
};

}

#endif // GALOIS_PRIORITY_QUEUE

