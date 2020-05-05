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

#ifndef GALOIS_WORKLIST_WORKLISTHELPERS_H
#define GALOIS_WORKLIST_WORKLISTHELPERS_H

#include <boost/iterator/iterator_facade.hpp>

#include "galois/config.h"
#include "galois/substrate/PtrLock.h"
#include "galois/worklists/WLCompileCheck.h"

namespace galois {
namespace worklists {

template <typename T>
class ConExtListNode {
  T* next;

public:
  ConExtListNode() : next(0) {}
  T*& getNext() { return next; }
  T* const& getNext() const { return next; }
};

template <typename T>
class ConExtIterator
    : public boost::iterator_facade<ConExtIterator<T>, T,
                                    boost::forward_traversal_tag> {
  friend class boost::iterator_core_access;
  T* at;

  template <typename OtherTy>
  bool equal(const ConExtIterator<OtherTy>& o) const {
    return at == o.at;
  }

  T& dereference() const { return *at; }
  void increment() { at = at->getNext(); }

public:
  ConExtIterator() : at(0) {}

  template <typename OtherTy>
  ConExtIterator(const ConExtIterator<OtherTy>& o) : at(o.at) {}

  explicit ConExtIterator(T* x) : at(x) {}
};

template <typename T, bool concurrent>
class ConExtLinkedStack {
  // fixme: deal with concurrent
  substrate::PtrLock<T> head;

public:
  typedef ConExtListNode<T> ListNode;

  bool empty() const { return !head.getValue(); }

  void push(T* C) {
    T* oldhead(0);
    do {
      oldhead      = head.getValue();
      C->getNext() = oldhead;
    } while (!head.CAS(oldhead, C));
  }

  T* pop() {
    // lock free Fast path (empty)
    if (empty())
      return 0;

    // Disable CAS
    head.lock();
    T* C = head.getValue();
    if (!C) {
      head.unlock();
      return 0;
    }
    head.unlock_and_set(C->getNext());
    C->getNext() = 0;
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

template <typename T, bool concurrent>
class ConExtLinkedQueue {
  // Fixme: deal with concurrent
  substrate::PtrLock<T> head;
  T* tail;

public:
  typedef ConExtListNode<T> ListNode;

  ConExtLinkedQueue() : tail(0) {}

  bool empty() const { return !tail; }

  void push(T* C) {
    head.lock();
    // std::cerr << "in(" << C << ") ";
    C->getNext() = 0;
    if (tail) {
      tail->getNext() = C;
      tail            = C;
      head.unlock();
    } else {
      assert(!head.getValue());
      tail = C;
      head.unlock_and_set(C);
    }
  }

  T* pop() {
    // lock free Fast path empty case
    if (empty())
      return 0;

    head.lock();
    T* C = head.getValue();
    if (!C) {
      head.unlock();
      return 0;
    }
    if (tail == C) {
      tail = 0;
      assert(!C->getNext());
      head.unlock_and_clear();
    } else {
      head.unlock_and_set(C->getNext());
      C->getNext() = 0;
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

template <typename T>
struct DummyIndexer : public std::unary_function<const T&, unsigned> {
  unsigned operator()(const T&) { return 0; }
};

} // namespace worklists
} // end namespace galois

#endif
