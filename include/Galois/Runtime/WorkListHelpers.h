// worklists building blocks -*- C++ -*-

#ifndef __WORKLISTHELPERS_H_
#define __WORKLISTHELPERS_H_

#include <iostream>

namespace GaloisRuntime {
namespace WorkList {

template<typename T, unsigned chunksize = 64, bool concurrent = true>
class FixedSizeRing :private boost::noncopyable, private PaddedLock<concurrent> {
  using PaddedLock<concurrent>::lock;
  using PaddedLock<concurrent>::unlock;
  unsigned start;
  unsigned end;
  T data[chunksize];

  bool _i_empty() {
    return start == end;
  }

  bool _i_full() {
    return (end + 1) % chunksize == start;
  }

  inline void assertSE() {
    assert(start <= chunksize);
    assert(end <= chunksize);
  }

public:
  
  template<bool newconcurrent>
  struct rethread {
    typedef FixedSizeRing<T, chunksize, newconcurrent> WL;
  };

  typedef T value_type;

  FixedSizeRing() :start(0), end(0) { assertSE(); }

  bool empty() {
    lock();
    assertSE();
    bool retval = _i_empty();
    assertSE();
    unlock();
    return retval;
  }

  bool full() {
    lock();
    assertSE();
    bool retval = _i_full();
    assertSE();
    unlock();
    return retval;
  }

  bool push_front(value_type val) {
    lock();
    assertSE();
    if (_i_full()) {
      unlock();
      return false;
    }
    start += chunksize - 1;
    start %= chunksize;
    data[start] = val;
    assertSE();
    unlock();
    return true;
  }

  bool push_back(value_type val) {
    lock();
    assertSE();
    if (_i_full()) {
      unlock();
      return false;
    }
    data[end] = val;
    end += 1;
    end %= chunksize;
    assertSE();
    unlock();
    return true;
  }

  std::pair<bool, value_type> pop_front() {
    lock();
    assertSE();
    if (_i_empty()) {
      unlock();
      return std::make_pair(false, value_type());
    }
    value_type retval = data[start];
    ++start;
    start %= chunksize;
    assertSE();
    unlock();
    return std::make_pair(true, retval);
  }

  std::pair<bool, value_type> pop_back() {
    lock();
    assertSE();
    if (_i_empty()) {
      unlock();
      return std::make_pair(false, value_type());
    }
    end += chunksize - 1;
    end %= chunksize;
    value_type retval = data[end];
    assertSE();
    unlock();
    return std::make_pair(true, retval);
  }
};


template<typename T, bool concurrent>
class ConExtLinkedStack {
  PtrLock<T*, concurrent> head;

public:
  struct ListNode {
    T* NextPtrLock;
    T*& getNextPtr() {
      return NextPtrLock;
    }
  };
  
  bool empty() {
    return !head.getValue();
  }

  void push(T* C) {
    head.lock();
    C->getNextPtr() = head.getValue();
    head.unlock_and_set(C);
  }

  T* pop() {
    //lock free Fast path (empty)
    if (empty()) return 0;

    head.lock();
    T* C = head.getValue();
    if (C) {
      head.unlock_and_set(C->getNextPtr());
      C->getNextPtr() = 0;
    } else {
      head.unlock();
    }
    return C;
  }
};


template<typename T, bool concurrent>
class ConExtLinkedQueue {
  PtrLock<T*, concurrent> head;
  T* tail;

public:
  struct ListNode {
    T* NextPtrLock;
    T*& getNextPtr() {
      return NextPtrLock;
    }
  };

  ConExtLinkedQueue() :tail(0) {}
  
  bool empty() {
    return !head.getValue();
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
    if (C) {
      if (tail == C)
	tail = 0;
      head.unlock_and_set(C->getNextPtr());
      C->getNextPtr() = 0;
      //std::cerr << "pop(" << C << ") ";
    } else {
      //std::cerr << "pop(" << C << ") ";
      head.unlock();
    }
    return C;
  }
};


}
}


#endif
