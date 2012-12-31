/** worklists building blocks -*- C++ -*-
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_RUNTIME_WORKLISTHELPERS_H
#define GALOIS_RUNTIME_WORKLISTHELPERS_H

#ifndef GALOIS_WLCOMPILECHECK
#define GALOIS_WLCOMPILECHECK(name) //
#endif

#include "ll/PtrLock.h"

#include <boost/iterator/iterator_facade.hpp>
#include <boost/optional.hpp>

namespace GaloisRuntime {
namespace WorkList {

template<typename T>
class ConExtListNode {
  T* NextPtr;
public:
  ConExtListNode() :NextPtr(0) {}
  T*& getNextPtr() { return NextPtr; }
  T*const& getNextPtr() const { return NextPtr; }
};

template<typename T>
class ConExtIterator: public boost::iterator_facade<
                      ConExtIterator<T>, T, boost::forward_traversal_tag> {
  friend class boost::iterator_core_access;
  T* at;

  template<typename OtherTy>
  bool equal(const ConExtIterator<OtherTy>& o) const { return at == o.at; }

  T& dereference() const { return *at; }
  void increment() { at = at->getNextPtr(); }

public:
  ConExtIterator(): at(0) { }
  
  template<typename OtherTy>
  ConExtIterator(const ConExtIterator<OtherTy>& o): at(o.at) { }
  
  explicit ConExtIterator(T* x): at(x) { }
};

template<typename T, bool concurrent>
class ConExtLinkedStack {
  LL::PtrLock<T, concurrent> head;
  
public:
  typedef ConExtListNode<T> ListNode;

  bool empty() const {
    return !head.getValue();
  }

  void push(T* C) {
    T* oldhead(0);
    do {
      oldhead = head.getValue();
      C->getNextPtr() = oldhead;
    } while (!head.CAS(oldhead, C));
  }

  T* pop() {
    //lock free Fast path (empty)
    if (empty()) return 0;
    
    //Disable CAS
    head.lock();
    T* C = head.getValue();
    if (!C) {
      head.unlock();
      return 0;
    }
    head.unlock_and_set(C->getNextPtr());
    C->getNextPtr() = 0;
    return C;
  }

  //! iterators not safe with concurrent modifications
  typedef T value_type;
  typedef T& reference;
  typedef ConExtIterator<T> iterator;
  typedef ConExtIterator<const T> const_iterator;

  iterator begin() { return iterator(head.getValue()); }
  iterator end() { return iterator(); }

  const_iterator begin() const { return const_iterator(head.getValue()); }
  const_iterator end() const { return const_iterator(); }
};


template<typename T, bool concurrent>
class ConExtLinkedQueue {
  LL::PtrLock<T,concurrent> head;
  T* tail;
  
public:
  class ListNode {
    T* NextPtr;
  public:
    ListNode() :NextPtr(0) {}
    T*& getNextPtr() { return NextPtr; }
  };
  
  ConExtLinkedQueue() :tail(0) { }

  bool empty() const {
    return !tail;
  }

  void push(T* C) {
    head.lock();
    //std::cerr << "in(" << C << ") ";
    C->getNextPtr() = 0;
    if (tail) {
      tail->getNextPtr() = C;
      tail = C;
      head.unlock();
    } else {
      assert(!head.getValue());
      tail = C;
      head.unlock_and_set(C);
    }
  }

  T* pop() {
    //lock free Fast path empty case
    if (empty()) return 0;

    head.lock();
    T* C = head.getValue();
    if (!C) {
      head.unlock();
      return 0;
    }
    if (tail == C) {
      tail = 0;
      assert(!C->getNextPtr());
      head.unlock_and_clear();
    } else {
      head.unlock_and_set(C->getNextPtr());
      C->getNextPtr() = 0;
    }
    return C;
  }

  //! iterators not safe with concurrent modifications
  typedef T value_type;
  typedef T& reference;
  typedef ConExtIterator<T> iterator;
  typedef ConExtIterator<const T> const_iterator;

  iterator begin() { return iterator(head.getValue()); }
  iterator end() { return iterator(); }

  const_iterator begin() const { return const_iterator(head.getValue()); }
  const_iterator end() const { return const_iterator(); }
};

template<typename T>
struct DummyIndexer: public std::unary_function<const T&,unsigned> {
  unsigned operator()(const T& x) { return 0; }
};

}
}

#endif
