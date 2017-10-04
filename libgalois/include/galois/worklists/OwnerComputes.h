/** Owner Computes worklist -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
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
  substrate::PerPackageStorage<cWL> items;
  substrate::PerPackageStorage<pWL> pushBuffer;

public:
  typedef T value_type;

  void push(const value_type& val)  {
    unsigned int index = Fn(val);
    auto& tp = substrate::getThreadPool();
    unsigned int mindex = tp.getPackage(index);
    //std::cerr << "[" << index << "," << index % active << "]\n";
    if (mindex == substrate::ThreadPool::getPackage())
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
