/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
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

#ifndef GALOIS_WORKLIST_OWNERCOMPUTES_H
#define GALOIS_WORKLIST_OWNERCOMPUTES_H

#include "WLCompileCheck.h"

namespace galois {
namespace worklists {

template<typename OwnerFn=DummyIndexer<int>, typename Container=ChunkedLIFO<>, typename T = int>
struct OwnerComputes : private boost::noncopyable {
  template<typename _T>
  using retype = OwnerComputes<OwnerFn, typename Container::template retype<_T>, _T>;

  template<bool b>
  using rethread = OwnerComputes<OwnerFn, typename Container::template rethread<b>, T>;

  template<typename _container>
  struct with_container { typedef OwnerComputes<OwnerFn, _container, T> type; };

  template<typename _indexer>
  struct with_indexer { typedef OwnerComputes<_indexer, Container, T> type; };

private:
  typedef typename Container::template retype<T> lWLTy;

  typedef lWLTy cWL;
  typedef lWLTy pWL;

  OwnerFn Fn;
  substrate::PerSocketStorage<cWL> items;
  substrate::PerSocketStorage<pWL> pushBuffer;

public:
  typedef T value_type;

  void push(const value_type& val)  {
    unsigned int index = Fn(val);
    auto& tp = substrate::getThreadPool();
    unsigned int mindex = tp.getSocket(index);
    //std::cerr << "[" << index << "," << index % active << "]\n";
    if (mindex == substrate::ThreadPool::getSocket())
      items.getLocal()->push(val);
    else
      pushBuffer.getRemote(mindex)->push(val);
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
    for (unsigned int x = 0; x < pushBuffer.size(); ++x)
      pushBuffer.getRemote(x)->flush();
  }

  galois::optional<value_type> pop() {
    cWL& wl = *items.getLocal();
    galois::optional<value_type> retval = wl.pop();
    if (retval)
      return retval;
    pWL& p = *pushBuffer.getLocal();
    while ((retval = p.pop()))
      wl.push(*retval);
    return wl.pop();
  }
};
GALOIS_WLCOMPILECHECK(OwnerComputes)


} // end namespace worklists
} // end namespace galois

#endif
