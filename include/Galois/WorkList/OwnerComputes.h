/** Owner Computes worklist -*- C++ -*-
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
#ifndef GALOIS_WORKLIST_OWNERCOMPUTES_H
#define GALOIS_WORKLIST_OWNERCOMPUTES_H

#include "WLCompileCheck.h"

namespace Galois {
namespace WorkList {

template<typename OwnerFn=DummyIndexer<int>, typename Container=ChunkedLIFO<>, typename T = int>
struct OwnerComputes : private boost::noncopyable {
  template<bool _concurrent>
  struct rethread { typedef OwnerComputes<OwnerFn, typename Container::template rethread<_concurrent>::type, T> type; };

  template<typename _T>
  struct retype { typedef OwnerComputes<OwnerFn, typename Container::template retype<_T>::type, _T> type; };

  template<typename _container>
  struct with_container { typedef OwnerComputes<OwnerFn, _container, T> type; };

  template<typename _indexer>
  struct with_indexer { typedef OwnerComputes<_indexer, Container, T> type; };

private:
  typedef typename Container::template retype<T>::type lWLTy;

  typedef lWLTy cWL;
  typedef lWLTy pWL;

  OwnerFn Fn;
  Runtime::PerPackageStorage<cWL> items;
  Runtime::PerPackageStorage<pWL> pushBuffer;

public:
  typedef T value_type;

  void push(const value_type& val)  {
    unsigned int index = Fn(val);
    unsigned int tid = Runtime::LL::getTID();
    unsigned int mindex = Runtime::LL::getPackageForThread(index);
    //std::cerr << "[" << index << "," << index % active << "]\n";
    if (mindex == Runtime::LL::getPackageForSelf(tid))
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

  Galois::optional<value_type> pop() {
    cWL& wl = *items.getLocal();
    Galois::optional<value_type> retval = wl.pop();
    if (retval)
      return retval;
    pWL& p = *pushBuffer.getLocal();
    while ((retval = p.pop()))
      wl.push(*retval);
    return wl.pop();
  }
};
GALOIS_WLCOMPILECHECK(OwnerComputes)


} // end namespace WorkList
} // end namespace Galois

#endif
