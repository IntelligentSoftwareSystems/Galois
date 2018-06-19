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

#ifndef GALOIS_RUNTIME_WORK_LIST_WRAPPER_H
#define GALOIS_RUNTIME_WORK_LIST_WRAPPER_H
namespace galois {
namespace worklists {

template <typename WL>
class WLsizeWrapper : public WL {

  substrate::PerThreadStorage<size_t> size_cntr;

public:
  template <typename _T>
  using retype = WLsizeWrapper<typename WL::template retype<_T>>;

  WLsizeWrapper() : WL() {
    for (unsigned i = 0; i < size_cntr.size(); ++i) {
      *(size_cntr.getRemote(i)) = 0;
    }
  }

  void push(const typename WL::value_type& v) {
    WL::push(v);
    *(size_cntr.getLocal()) += 1;
  }

  template <typename I>
  void push(I b, I e) {
    for (I i = b; i != e; ++i) {
      push(*i);
    }
  }

  template <typename R>
  void push_initial(const R& range) {
    auto rp = range.local_pair();
    push(rp.first, rp.second);
  }

  size_t size(void) const {
    size_t s = 0;
    for (unsigned i = 0; i < size_cntr.size(); ++i) {
      s += *(size_cntr.getRemote(i));
    }
    return s;
  }

  // parallel
  void reset(void) { *(size_cntr.getLocal()) = 0; }

  // sequential
  void reset_all(void) {
    for (unsigned i = 0; i < size_cntr.size(); ++i) {
      *(size_cntr.getRemote(i)) = 0;
    }
  }
};

} // end namespace worklists
} // end namespace galois

#endif // GALOIS_RUNTIME_WORK_LIST_WRAPPER_H
