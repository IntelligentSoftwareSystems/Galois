/** FIFO worklist -*- C++ -*-
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#ifndef GALOIS_WORKLIST_FIFO_H
#define GALOIS_WORKLIST_FIFO_H

#include "Galois/Substrate/PaddedLock.h"
#include "Galois/gdeque.h"
#include "WLCompileCheck.h"

#include <deque>
#include <mutex>

namespace galois {
namespace worklists {

//! Simple Container Wrapper worklist (not scalable).
template<typename T, typename container = std::deque<T>, bool popBack = true>
class Wrapper : private boost::noncopyable {
  substrate::PaddedLock<true> lock;
  container wl;

public:
  template<typename _T>
  using retype = Wrapper<_T>;

  template<bool b>
  using rethread = Wrapper;

  typedef T value_type;
  
  void push(const value_type& val) {
    std::lock_guard<substrate::PaddedLock<true> > lg(lock);
    wl.push_back(val);
  }
  
  template<typename Iter>
  void push(Iter b, Iter e) {
    std::lock_guard<substrate::PaddedLock<true> > lg(lock);
    wl.insert(wl.end(),b,e);
  }

  template<typename RangeTy>
  void push_initial(const RangeTy& range) {
    if (substrate::ThreadPool::getTID() == 0)
      push(range.begin(), range.end());
  }

  galois::optional<value_type> pop() {
    galois::optional<value_type> retval;
    std::lock_guard<substrate::PaddedLock<true> > lg(lock);
    if (!wl.empty()) {
      if (popBack) {
        retval = wl.back();
        wl.pop_back();
      } else {
        retval = wl.front();
        wl.pop_front();
      }
    }
    return retval;
  }
};

template<typename T = int>
using FIFO = Wrapper<T, std::deque<T>, false >;

template<typename T = int>
using GFIFO = Wrapper<T, galois::gdeque<T>, false >;

template<typename T = int>
using LIFO = Wrapper<T, std::deque<T>, true >;

template<typename T = int>
using GLIFO = Wrapper<T, galois::gdeque<T>, true >;


GALOIS_WLCOMPILECHECK(FIFO)
GALOIS_WLCOMPILECHECK(GFIFO)
GALOIS_WLCOMPILECHECK(LIFO)
GALOIS_WLCOMPILECHECK(GLIFO)


} // end namespace worklists
} // end namespace galois

#endif
