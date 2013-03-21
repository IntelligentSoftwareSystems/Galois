/** External worklist -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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
 * @description
 * This let's you use an external worklist by reference
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

namespace Galois {
namespace WorkList {

template<typename eWLTy, bool pushinit = true>
class ExternRef {
  eWLTy& wl;
public:
  ExternRef(eWLTy& _wl) :wl(_wl) {}

  //! T is the value type of the WL
  typedef typename eWLTy::value_type value_type;

  //! change the concurrency flag
  template<bool newconcurrent>
  using rethread = ExternRef<typename eWLTy::template rethread<newconcurrent> >;
  
  //! change the type the worklist holds
  template<typename Tnew>
  using retype = ExternRef<typename eWLTy::template retype<Tnew> >;

  //! push a value onto the queue
  void push(const value_type& val) { wl.push(val); }

  //! push a range onto the queue
  template<typename Iter>
  void push(Iter b, Iter e) { wl.push(b,e); }

  //! push initial range onto the queue
  //! called with the same b and e on each thread
  template<typename RangeTy>
  void push_initial(const RangeTy& r) { if (pushinit) wl.push_initial(r); }

  //! pop a value from the queue.
  boost::optional<value_type> pop() { return wl.pop(); }
};


}
}
