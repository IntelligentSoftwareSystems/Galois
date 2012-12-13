/** LocalIterator Wrapper -*- C++ -*-
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
 * @author Andrew Lenharth <andrew@lenharth.org>
 */
#ifndef GALOIS_LOCALITERATOR_H
#define GALOIS_LOCALITERATOR_H

namespace Galois {
namespace Runtime {

//! Iterator over items in containers of containers
template<typename ContainerTy>
class LocalBounce : public std::iterator_traits<typename ContainerTy::local_iterator> {
  ContainerTy* C;
  bool isBegin;

public:
  LocalBounce(ContainerTy* c = 0, bool b = false) :C(c), isBegin(b) {}

  typename ContainerTy::local_iterator resolve() {
    if (isBegin)
      return C->local_begin();
    return C->local_end();
  }
};

}
}

#endif
