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

namespace Galois {
namespace WorkList {

template<typename Global = FIFO<>, typename Local = FIFO<>, typename T = int>
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
  Runtime::PerThreadStorage<lWLTy> local;
  Global global;

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
    global.push_initial(range);
  }

  boost::optional<value_type> pop() {
    boost::optional<value_type> ret = local.getLocal()->pop();
    if (ret)
      return ret;
    return global.pop();
  }
};
GALOIS_WLCOMPILECHECK(LocalQueue)

} // end namespace WorkList
} // end namespace Galois

#endif
