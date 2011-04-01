// worklists building blocks -*- C++ -*-

#ifndef __WORKLISTHELPERS_H_
#define __WORKLISTHELPERS_H_

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

  //externally managed locking access methods

  void external_lock() {
    lock();
  }

  void external_unlock() {
    unlock();
  }

  bool external_empty() {
    return _i_empty();
  }

  bool external_full() {
    return _i_full();
  }

  bool external_push_front(value_type val) {
    if (_i_full()) {
      return false;
    }
    start += chunksize - 1;
    start %= chunksize;
    data[start] = val;
    return true;
  }

  bool external_push_back(value_type val) {
    if (_i_full()) {
      return false;
    }
    data[end] = val;
    end += 1;
    end %= chunksize;
    return true;
  }

  std::pair<bool, value_type> external_pop_front() {
    if (_i_empty()) {
      return std::make_pair(false, value_type());
    }
    value_type retval = data[start];
    ++start;
    start %= chunksize;
    return std::make_pair(true, retval);
  }

  std::pair<bool, value_type> external_pop_back() {
    if (_i_empty()) {
      return std::make_pair(false, value_type());
    }
    end += chunksize - 1;
    end %= chunksize;
    value_type retval = data[end];
    return std::make_pair(true, retval);
  }

  //"Normal" access methods

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


}
}


#endif
