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

#include "Galois/Runtime/ll/PaddedLock.h"

#include <boost/optional.hpp>

namespace GaloisRuntime {
namespace WorkList {

template<typename T, unsigned chunksize = 64>
class FixedSizeRing :private boost::noncopyable {
  unsigned start;
  unsigned count;
  char datac[sizeof(T[chunksize])] __attribute__ ((aligned (__alignof__(T))));

  T* data() {
    return reinterpret_cast<T*>(&datac[0]);
  }

  T* at(unsigned i) {
    return &data()[i];
  }

  void destroy(unsigned i) {
    (at(i))->~T();
  }

  T* create(unsigned i, const T& val) {
    return new (at(i)) T(val);
  }

public:

  typedef T value_type;

  FixedSizeRing() :start(0), count(0) {}
  ~FixedSizeRing() {
    clear();
  }

  unsigned size() const {
    return count;
  }

  bool empty() const {
    return count == 0;
  }

  bool full() const {
    return count == chunksize;
  }

  T& getAt(unsigned x) {
    return *at((start + x) % chunksize);
  }

  void clear() {
    for (unsigned x = 0; x < count; ++x)
      destroy((start + count) % chunksize);
    count = 0;
    start = 0;
  }

  value_type* push_front(const value_type& val) {
    if (full()) return 0;
    start = (start + chunksize - 1) % chunksize;
    ++count;
    return create(start, val);
  }

  value_type* push_back(const value_type& val) {
    if (full()) return 0;
    int end = (start + count) % chunksize;
    ++count;
    return create(end, val);
  }

  template<typename Iter>
  Iter push_back(Iter b, Iter e) {
    while (push_back(*b)) { ++b; }
    return b;
  }

  template<typename Iter>
  Iter push_front(Iter b, Iter e) {
    while (push_front(*b)) { ++b; }
    return b;
  }

  boost::optional<value_type> pop_front() {
    boost::optional<value_type> retval;
    if (!empty()) {
      retval = *at(start);
      destroy(start);
      start = (start + 1) % chunksize;
      --count;
    }
    return retval;
  }
  
  boost::optional<value_type> pop_back() {
    boost::optional<value_type> retval;
    if (!empty()) {
      int end = (start + count - 1) % chunksize;
      retval = *at(end);
      destroy(end);
      --count;
    }
    return retval;
  }
};
  
template<typename T, bool concurrent>
class ConExtLinkedStack {
  LL::PtrLock<T, concurrent> head;
  
public:
  class ListNode {
    T* NextPtr;
  public:
    ListNode() :NextPtr(0) {}
    T*& getNextPtr() { return NextPtr; }
  };

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
};

template<typename T>
struct DummyIndexer: public std::unary_function<const T&,unsigned> {
  unsigned operator()(const T& x) { return 0; }
};

}
}


#endif

