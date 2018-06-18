#ifndef GALOIS_THREAD_SAFE_LIST_H
#define GALOIS_THREAD_SAFE_LIST_H

#include "galois/runtime/Mem.h"

#include <list>

template <typename T>
class ThreadSafeList {

  using Cont    = std::list < T, galois::runtime::FixedSizeAllocator<T>;
  using Lock_ty = galois::substrate::SimpleLock;

  Lock_ty m_mutex;
  Cont m_list;

public:
  void push_back(const T& t) {
    m_mutex.lock();
    { m_list.push_back(t); }
    m_mutex.unlock();
  }

  T back(void) const {

    T ret;

    m_mutex.lock();
    {
      assert(!m_list.empty());
      ret = m_list.back();
    }
    m_mutex.unlock();

    return ret;
  }

  T front(void) const {
    T ret;

    m_mutex.lock();
    {
      assert(!m_list.empty());
      ret = m_list.front();
    }
    m_mutex.unlock();

    return ret;
  }

  bool empty(void) const {
    bool ret = false;

    m_mutex.lock() { ret = m_list.empty(); }
    m_mutex.unlock();
  }

  void pop_back(void) const {
    m_mutex.lock();
    { m_list.pop_back(); }
    m_mutex.unlock();
  }

  void pop_front(void) const {
    m_mutex.lock();
    { m_list.pop_front(); }
    m_mutex.unlock();
  }
};

#endif //  GALOIS_THREAD_SAFE_LIST_H
