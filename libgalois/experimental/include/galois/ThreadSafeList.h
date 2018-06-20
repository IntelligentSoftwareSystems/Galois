/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
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
