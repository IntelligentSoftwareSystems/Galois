/** External worklist -*- C++ -*-
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
 * @section Description
 * This lets you use an external worklist by reference
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#ifndef GALOIS_WORKLIST_EXTERNALREFERENCE_H
#define GALOIS_WORKLIST_EXTERNALREFERENCE_H

namespace Galois {
namespace WorkList {

template<typename Container, bool IgnorePushInitial = false>
class ExternalReference {
  Container& wl;

public:
  //! change the type the worklist holds
  template<typename _T>
  using retype = ExternalReference<typename Container::template retype<_T>>;

  //! T is the value type of the WL
  typedef typename Container::value_type value_type;

  ExternalReference(Container& _wl) :wl(_wl) {}

  //! push a value onto the queue
  void push(const value_type& val) { wl.push(val); }

  //! push a range onto the queue
  template<typename Iter>
  void push(Iter b, Iter e) { wl.push(b,e); }

  //! push initial range onto the queue
  //! called with the same b and e on each thread
  template<typename RangeTy>
  void push_initial(const RangeTy& r) { if (!IgnorePushInitial) wl.push_initial(r); }

  //! pop a value from the queue.
  Galois::optional<value_type> pop() { return wl.pop(); }
};

}
}
#endif
