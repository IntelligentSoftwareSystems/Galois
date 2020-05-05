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

#ifndef GALOIS_WORKLIST_LOCALQUEUE_H
#define GALOIS_WORKLIST_LOCALQUEUE_H

#include <type_traits>

#include <boost/mpl/if.hpp>

#include "galois/config.h"
#include "galois/worklists/Simple.h"

namespace galois {
namespace worklists {

template <typename T = int>
struct NoGlobalQueue {
  template <bool _concurrent>
  using rethread = NoGlobalQueue<T>;

  template <typename _T>
  using retype = NoGlobalQueue<_T>;
};

template <typename Global = NoGlobalQueue<>, typename Local = GFIFO<int>,
          typename T = int>
struct LocalQueue : private boost::noncopyable {
  template <bool _concurrent>
  using rethread = LocalQueue<Global, Local, T>;

  template <typename _T>
  using retype = LocalQueue<typename Global::template retype<_T>,
                            typename Local::template retype<_T>, _T>;

  template <typename _global>
  using with_global = LocalQueue<_global, Local, T>;

  template <typename _local>
  using with_local = LocalQueue<Global, _local, T>;

private:
  typedef typename Local::template rethread<false> lWLTy;
  substrate::PerThreadStorage<lWLTy> local;
  Global global;

  template <typename RangeTy,
            bool Enable = std::is_same<Global, NoGlobalQueue<T>>::value>
  void pushGlobal(const RangeTy& range,
                  typename std::enable_if<Enable>::type* = 0) {
    auto rp = range.local_pair();
    local.getLocal()->push(rp.first, rp.second);
  }

  template <typename RangeTy,
            bool Enable = std::is_same<Global, NoGlobalQueue<T>>::value>
  void pushGlobal(const RangeTy& range,
                  typename std::enable_if<!Enable>::type* = 0) {
    global.push_initial(range);
  }

  template <bool Enable = std::is_same<Global, NoGlobalQueue<T>>::value>
  galois::optional<T> popGlobal(typename std::enable_if<Enable>::type* = 0) {
    return galois::optional<value_type>();
  }

  template <bool Enable = std::is_same<Global, NoGlobalQueue<T>>::value>
  galois::optional<T> popGlobal(typename std::enable_if<!Enable>::type* = 0) {
    return global.pop();
  }

public:
  typedef T value_type;

  void push(const value_type& val) { local.getLocal()->push(val); }

  template <typename Iter>
  void push(Iter b, Iter e) {
    local.getLocal()->push(b, e);
  }

  template <typename RangeTy>
  void push_initial(const RangeTy& range) {
    pushGlobal(range);
  }

  galois::optional<value_type> pop() {
    galois::optional<value_type> ret = local.getLocal()->pop();
    if (ret)
      return ret;
    return popGlobal();
  }
};
GALOIS_WLCOMPILECHECK(LocalQueue)

} // end namespace worklists
} // end namespace galois

#endif
