/** Ranges -*- C++ -*-
 * @file
 * This is the only file to include for basic Galois functionality.
 *
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
 * Copyright (C) 2016, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * @author Donald Nguyen <ddn@cs.texas.edu>
 */
#ifndef GALOIS_RUNTIME_RANGE_H
#define GALOIS_RUNTIME_RANGE_H

#include "Galois/Threads.h"

#include "Galois/Runtime/Blocking.h"
#include "Galois/Runtime/ThreadPool.h"

#include <iterator>

namespace Galois {
namespace Runtime {

// TODO(ddn): update to have better forward iterator behavor for blocked/local iteration

template<typename T>
class LocalRange {
  T* container;

public:
  typedef T container_type;
  typedef typename T::iterator iterator;
  typedef typename T::local_iterator local_iterator;
  typedef iterator block_iterator;
  typedef typename std::iterator_traits<iterator>::value_type value_type;
  
  LocalRange(T& c): container(&c) { }

  iterator begin() const { return container->begin(); }
  iterator end() const { return container->end(); }

  // TODO fix constness of local containers
  /* const */ T& get_container() const { return *container; }

  std::pair<block_iterator, block_iterator> block_pair() const {
    return block_range(begin(), end(), ThreadPool::getTID(), getActiveThreads());
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
    return block_range(ii, ei, ThreadPool::getTID(), getActiveThreads());
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
