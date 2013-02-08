/** Ranges -*- C++ -*-
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
 * @section Description
 *
 * @author Donald Nguyen <ddn@cs.texas.edu>
 */
#ifndef GALOIS_RUNTIME_RANGE_H
#define GALOIS_RUNTIME_RANGE_H

#include "Galois/gstl.h"

#include "Galois/Runtime/ActiveThreads.h"
#include "Galois/Runtime/ll/TID.h"

#include <iterator>

namespace Galois {
namespace Runtime {

// TODO(ddn): update to have better forward iterator behavor for blocked/local iteration

template<typename T>
class LocalRange {
  T* container;

public:
  typedef typename T::iterator iterator;
  typedef typename T::local_iterator local_iterator;
  typedef iterator block_iterator;
  typedef typename std::iterator_traits<iterator>::value_type value_type;
  
  LocalRange(T& c): container(&c) { }

  iterator begin() const { return container->begin(); }
  iterator end() const { return container->end(); }

  std::pair<block_iterator, block_iterator> block_pair() const {
    return Galois::block_range(begin(), end(), LL::getTID(), activeThreads);
  }

  std::pair<local_iterator, local_iterator> local_pair() const {
    return std::make_pair(container->local_begin(), container->local_end());
  }

  local_iterator local_begin() const { return container->local_begin(); }
  local_iterator local_end() const { return container->local_end(); }

  block_iterator block_begin() const { return block_pair().first; }
  block_iterator block_end() const { return block_pair().second; }
};

template<typename T>
inline LocalRange<T> makeLocalRange(T& obj) { return LocalRange<T>(obj); }

template<typename IterTy>
class StandardRange {
  IterTy ii, ei;
public:
  typedef IterTy iterator;
  typedef iterator local_iterator;
  typedef iterator block_iterator;

  typedef typename std::iterator_traits<IterTy>::value_type value_type;

  StandardRange(IterTy b, IterTy e): ii(b), ei(e) { }

  iterator begin() const { return ii; }
  iterator end() const { return ei; }

  std::pair<block_iterator, block_iterator> block_pair() const {
    return Galois::block_range(ii, ei, LL::getTID(), activeThreads);
  }

  std::pair<local_iterator, local_iterator> local_pair() const {
    return block_pair();
  }

  local_iterator local_begin() const { return block_begin(); }
  local_iterator local_end() const { return block_end(); }

  block_iterator block_begin() const { return block_pair().first; }

  block_iterator block_end() const { return block_pair().second; }
};

template<typename IterTy>
inline StandardRange<IterTy> makeStandardRange(IterTy begin, IterTy end) {
  return StandardRange<IterTy>(begin, end);
}

}
} // end namespace Galois

#endif
