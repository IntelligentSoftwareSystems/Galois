/** BulkSynchronous worklist -*- C++ -*-
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
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#ifndef GALOIS_WORKLIST_BULKSYNCHRONOUS_H
#define GALOIS_WORKLIST_BULKSYNCHRONOUS_H

#include "galois/Runtime/Substrate.h"
#include "Chunked.h"
#include "WLCompileCheck.h"

#include <atomic>

namespace galois {
namespace worklists {

/**
 * Bulk-synchronous scheduling. Work is processed in rounds, and all newly
 * created work is processed after all the current work in a round is
 * completed.
 */
template<class Container=dChunkedFIFO<>, class T=int, bool Concurrent = true>
class BulkSynchronous : private boost::noncopyable {
public:
  template<bool _concurrent>
  using rethread = BulkSynchronous<Container, T, _concurrent>;

  template<typename _T>
  using retype = BulkSynchronous<typename Container::template retype<_T>, _T, Concurrent>;

  template<typename _container>
  using with_container = BulkSynchronous<_container, T, Concurrent>;

private:
  typedef typename Container::template rethread<Concurrent> CTy;

  struct TLD {
    unsigned round;
    TLD(): round(0) { }
  };

  CTy wls[2];
  substrate::PerThreadStorage<TLD> tlds;
  substrate::Barrier& barrier;
  substrate::CacheLineStorage<std::atomic<bool>> some;
  std::atomic<bool> isEmpty;

 public:
  typedef T value_type;

  BulkSynchronous(): barrier(runtime::getBarrier(runtime::activeThreads)), some(false), isEmpty(false) { }

  void push(const value_type& val) {
    wls[(tlds.getLocal()->round + 1) & 1].push(val);
  }

  template<typename ItTy>
  void push(ItTy b, ItTy e) {
    while (b != e)
      push(*b++);
  }

  template<typename RangeTy>
  void push_initial(const RangeTy& range) {
    auto rp = range.local_pair();
    push(rp.first, rp.second);
    tlds.getLocal()->round = 1;
    some.get() = true;
  }

  galois::optional<value_type> pop() {
    TLD& tld = *tlds.getLocal();
    galois::optional<value_type> r;
    
    while (true) {
      if (isEmpty)
        return r; // empty

      r = wls[tld.round].pop();
      if (r)
        return r;

      barrier.wait();
      if (substrate::ThreadPool::getTID() == 0) {
        if (!some.get())
          isEmpty = true;
        some.get() = false; 
      }
      tld.round = (tld.round + 1) & 1;
      barrier.wait();

      r = wls[tld.round].pop();
      if (r) {
        some.get() = true;
        return r;
      }
    }
  }
};
GALOIS_WLCOMPILECHECK(BulkSynchronous)

} // end namespace worklists
} // end namespace galois

#endif
