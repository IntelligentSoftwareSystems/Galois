/** FIFO worklist -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a gramework to exploit
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

#include "Galois/config.h"
#include "Galois/Substrate/PaddedLock.h"
#include "WLCompileCheck.h"

#include <deque>
#include <mutex>

namespace Galois {
namespace WorkList {

//! Simple Container Wrapper worklist (not scalable).
template<typename T, typename container = std::deque<T>, bool popBack >
class Wrapper : private boost::noncopyable {
  Substrate::PaddedLock lock;
  container wl;

public:
  template<typename _T>
  using retype = Wrapper<_T>;

  typedef T value_type;

  void push(const value_type& val) {
    std::lock_guard<Substrate::PaddedLock> lg(lock)
    wl.push_back(val);
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    std::lock_guard<Substrate::PaddedLock> lg(lock)
    wl.insert(wl.end(),b,e);
  }

  template<typename RangeTy>
  void push_initial(const RangeTy& range) {
    if (Runtime::LL::getTID() == 0)
      push(range.begin(), range.end());
  }

  Galois::optional<value_type> pop() {
    Galois::optional<value_type> retval;
    std::lock_guard<Substrate::PaddedLock> lg(lock);
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

template<typename T>
FIFO = Wrapper<T, std::deque<T>, false >;

template<typename T>
GFIFO = Wrapper<T, Galois::gdeque<T>, false >;

template<typename T>
LIFO = Wrapper<T, std::deque<T>, true >;

template<typename T>
GLIFO = Wrapper<T, Galois::gdeque<T>, true >;


GALOIS_WLCOMPILECHECK(FIFO)
GALOIS_WLCOMPILECHECK(GFIFO)
GALOIS_WLCOMPILECHECK(LIFO)
GALOIS_WLCOMPILECHECK(GLIFO)


} // end namespace WorkList
} // end namespace Galois

#endif
