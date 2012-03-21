/** Simple STL style algorithms -*- C++ -*-
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

namespace Galois {


template<typename IterTy, class Distance>
IterTy safe_advance_impl(IterTy b, IterTy e, Distance n, std::random_access_iterator_tag) {
  if (std::distance(b,e) <= n)
    return b + n;
  else
    return e;
}

template<typename IterTy, class Distance>
IterTy safe_advance_impl(IterTy b, IterTy e, Distance n, std::forward_iterator_tag) {
  while (b != e && n--)
    ++b;
  return b;
}

template<typename IterTy, class Distance>
IterTy safe_advance(IterTy b, IterTy e, Distance n) {
  typename std::iterator_traits<IterTy>::iterator_category category;
  safe_advance_impl(b,e,n,category);
}

}
