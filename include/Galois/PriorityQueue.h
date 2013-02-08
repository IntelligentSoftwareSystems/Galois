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

namespace Galois {

/**
 * Thread-safe ordered set. Faster than STL heap operations (about 10%-15% faster on serially) and
 * can use scalable allocation, e.g., {@link GFixedAllocator}.
 */
template <typename T, typename Cmp=std::less<T>, typename Alloc=Galois::GFixedAllocator<T> >
class ThreadSafeOrderedSet {
  typedef std::set<T, Cmp, Alloc> Set;

public:
  typedef Set container_type;
  typedef typename Set::value_type value_type;
  typedef typename Set::reference reference;
  typedef typename Set::const_reference const_reference;
  typedef typename Set::size_type size_type;
  typedef typename Set::const_iterator const_iterator;
  typedef Galois::Runtime::LL::SimpleLock<true> Lock_ty;

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


/**
 * Thread-safe min heap.
 */
template <typename T, typename Cmp=std::less<T>, typename Cont=std::vector<T> >
class ThreadSafeMinHeap {

public:
  typedef Cont container_type;

  typedef typename container_type::value_type value_type;
  typedef typename container_type::reference reference;
  typedef typename container_type::const_reference const_reference;
  typedef typename container_type::size_type size_type;
  typedef typename container_type::const_iterator const_iterator;

protected:
  struct RevCmp {
    Cmp cmp;

    explicit RevCmp (Cmp cmp): cmp (cmp) {}

    bool operator () (const T& left, const T& right) const {
      return !cmp (left, right);
    }
  };

  typedef Galois::Runtime::LL::SimpleLock<true> Lock_ty;

  GALOIS_ATTRIBUTE_ALIGN_CACHE_LINE Lock_ty mutex;
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

  explicit ThreadSafeMinHeap (const Cmp& cmp=Cmp (), const Cont& container=Cont ())
    : mutex (), container (container), revCmp (cmp)
  {}

  template <typename Iter>
  ThreadSafeMinHeap (Iter b, Iter e, const Cmp& cmp=Cmp ())
    : mutex (), container (b, e), revCmp (cmp)
  {
    std::make_heap (container.begin (), container.end ());
  }
  
  
  bool empty () const {
    mutex.lock ();
      bool ret = container.empty ();
    mutex.unlock ();

    return ret;
  }

  size_type size () const {
    mutex.lock ();
      size_type sz =  container.size ();
    mutex.unlock ();

    return sz;
  }

  // can't return a reference, because the reference may not be pointing
  // to a valid location due to vector doubling in size and moving to 
  // another memory location
  value_type top () const {
    mutex.lock ();
      value_type x = top_internal ();
    mutex.unlock ();

    return x;
  }

  void push (const value_type& x) {
    mutex.lock ();
      container.push_back (x);
      std::push_heap (container.begin (), container.end (), revCmp);
    mutex.unlock ();
  }

  value_type pop () {
    mutex.lock ();
      value_type x = pop_internal ();
    mutex.unlock ();
    return x;
  }

  bool remove (const value_type& x) {
    bool ret = false;

    // TODO: write a better remove method
    mutex.lock ();
      if (x == top_internal ()) {
        pop_internal ();
        ret = true;

      } else {
        typename container_type::iterator nend = 
            std::remove (container.begin (), container.end (), x);

        ret = nend != container.end ();
        container.erase (nend, container.end ());

        std::make_heap (container.begin (), container.end (), revCmp);

      }
    mutex.unlock ();

    return ret;
  }

  bool find (const value_type& x) const {
    mutex.lock ();
      bool ret = (std::find (begin (), end (), x) != end ());
    mutex.unlock ();

    return ret;
  }

  // TODO: can't use in parallel context
  const_iterator begin () const { return container.begin (); }
  const_iterator end () const { return container.end (); }

  // void reserve (size_type s) {
    // container.reserve (s);
  // }
};

}

#endif // GALOIS_PRIORITY_QUEUE

