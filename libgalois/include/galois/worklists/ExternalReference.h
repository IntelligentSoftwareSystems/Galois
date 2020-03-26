/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

#ifndef GALOIS_WORKLIST_EXTERNALREFERENCE_H
#define GALOIS_WORKLIST_EXTERNALREFERENCE_H

namespace galois {
namespace worklists {

template <typename Container, bool IgnorePushInitial = false>
class ExternalReference {
  Container& wl;

public:
  //! change the type the worklist holds
  template <typename _T>
  using retype = ExternalReference<typename Container::template retype<_T>>;

  //! T is the value type of the WL
  typedef typename Container::value_type value_type;

  ExternalReference(Container& _wl) : wl(_wl) {}

  //! push a value onto the queue
  void push(const value_type& val) { wl.push(val); }

  //! push a range onto the queue
  template <typename Iter>
  void push(Iter b, Iter e) {
    wl.push(b, e);
  }

  //! push initial range onto the queue
  //! called with the same b and e on each thread
  template <typename RangeTy>
  void push_initial(const RangeTy& r) {
    if (!IgnorePushInitial)
      wl.push_initial(r);
  }

  //! pop a value from the queue.
  galois::optional<value_type> pop() { return wl.pop(); }
};

} // namespace worklists
} // namespace galois
#endif
