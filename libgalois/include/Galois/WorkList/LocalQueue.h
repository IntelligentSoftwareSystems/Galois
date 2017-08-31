/** LocalQueues worklist -*- C++ -*-
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
#ifndef GALOIS_WORKLIST_LOCALQUEUE_H
#define GALOIS_WORKLIST_LOCALQUEUE_H

#include <boost/mpl/if.hpp>
#include "Galois/WorkList/Simple.h"

#include <type_traits>

namespace Galois {
namespace WorkList {

template<typename T = int>
struct NoGlobalQueue {
  template<bool _concurrent>
  using rethread = NoGlobalQueue<T>;

  template<typename _T>
  using retype = NoGlobalQueue<_T>;
};

template<typename Global = NoGlobalQueue<>, typename Local = GFIFO<int>, typename T = int>
struct LocalQueue : private boost::noncopyable {
  template<bool _concurrent>
  using rethread = LocalQueue<Global, Local, T>;

  template<typename _T>
  using retype = LocalQueue<typename Global::template retype<_T>, typename Local::template retype<_T>, _T>;

  template<typename _global>
  using with_global = LocalQueue<_global, Local, T>;

  template<typename _local>
  using with_local = LocalQueue<Global, _local, T>;

private:
  typedef typename Local::template rethread<false> lWLTy;
  Substrate::PerThreadStorage<lWLTy> local;
  Global global;

  template<typename RangeTy, bool Enable = std::is_same<Global,NoGlobalQueue<T> >::value>
  void pushGlobal(const RangeTy& range, typename std::enable_if<Enable>::type* = 0) {
    auto rp = range.local_pair();
    local.getLocal()->push(rp.first, rp.second);
  }

  template<typename RangeTy, bool Enable = std::is_same<Global,NoGlobalQueue<T> >::value>
  void pushGlobal(const RangeTy& range, typename std::enable_if<!Enable>::type* = 0) {
    global.push_initial(range);
  }

  template<bool Enable = std::is_same<Global,NoGlobalQueue<T> >::value>
  Galois::optional<T> popGlobal(typename std::enable_if<Enable>::type* = 0) {
    return Galois::optional<value_type>();
  }

  template<bool Enable = std::is_same<Global,NoGlobalQueue<T> >::value>
  Galois::optional<T> popGlobal(typename std::enable_if<!Enable>::type* = 0) {
    return global.pop();
  }

public:
  typedef T value_type;

  void push(const value_type& val) {
    local.getLocal()->push(val);
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    local.getLocal()->push(b,e);
  }

  template<typename RangeTy>
  void push_initial(const RangeTy& range) {
    pushGlobal(range);
  }

  Galois::optional<value_type> pop() {
    Galois::optional<value_type> ret = local.getLocal()->pop();
    if (ret)
      return ret;
    return popGlobal();
  }
};
GALOIS_WLCOMPILECHECK(LocalQueue)

} // end namespace WorkList
} // end namespace Galois

#endif
