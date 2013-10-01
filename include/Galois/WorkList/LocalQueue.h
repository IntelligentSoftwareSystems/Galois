/** LocalQueues worklist -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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
#ifndef GALOIS_WORKLIST_LOCALQUEUE_H
#define GALOIS_WORKLIST_LOCALQUEUE_H

#include "Galois/config.h"
#include <boost/mpl/if.hpp>
#include GALOIS_CXX11_STD_HEADER(type_traits)

namespace Galois {
namespace WorkList {

template<typename T = int>
struct NoGlobalQueue {
  template<bool _concurrent>
  struct rethread { typedef NoGlobalQueue<T> type; };

  template<typename _T>
  struct retype { typedef NoGlobalQueue<_T> type; };
};

template<typename Global = NoGlobalQueue<>, typename Local = FIFO<>, typename T = int>
struct LocalQueue : private boost::noncopyable {
  template<bool _concurrent>
  struct rethread { typedef LocalQueue<Global, Local, T> type; };

  template<typename _T>
  struct retype { typedef LocalQueue<typename Global::template retype<_T>::type, typename Local::template retype<_T>::type, _T> type; };

  template<typename _global>
  struct with_global { typedef LocalQueue<_global, Local, T> type; };

  template<typename _local>
  struct with_local { typedef LocalQueue<Global, _local, T> type; };

private:
  typedef typename Local::template rethread<false>::type lWLTy;
  Runtime::PerThreadStorage<lWLTy> local;
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
