/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
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

#ifndef GALOIS_GRAPHNODEBAG_H
#define GALOIS_GRAPHNODEBAG_H

#include "galois/Reduction.h"
#include "galois/LargeArray.h"
#include "galois/Bag.h"

namespace galois {

/**
 * Stores graph nodes to execute for {@link Ligra} executor.
 */
template<unsigned int BlockSize = 0>
class GraphNodeBag {
  typedef galois::InsertBag<size_t, BlockSize> Bag;
  typedef galois::LargeArray<bool> Bitmask;

  Bag bag;
  galois::GAccumulator<size_t> counts;
  galois::GAccumulator<size_t> numNodes;
  Bitmask bitmask;
  size_t size;
  bool isDense;

  struct InitializeSmall {
    GraphNodeBag* self;
    void operator()(size_t n) const {
      self->bitmask[n] = 0;
    }
  };

  struct InitializeBig {
    GraphNodeBag* self;
    void operator()(unsigned id, unsigned total) {
      typedef typename Bitmask::iterator It;
      std::pair<It,It> p = galois::block_range(self->bitmask.begin(), self->bitmask.end(), id, total);
      std::fill(p.first, p.second, 0);
    }
  };

  struct Densify {
    GraphNodeBag* self;
    void operator()(size_t n) const {
      self->bitmask[n] = true;
    }
  };
public:
  GraphNodeBag(size_t n): size(n), isDense(false) { }

  typedef typename Bag::iterator iterator;
  typedef typename Bag::local_iterator local_iterator;

  iterator begin() { return bag.begin(); }
  iterator end() { return bag.end(); }
  local_iterator local_begin() { return bag.local_begin(); }
  local_iterator local_end() { return bag.local_end(); }

  void pushDense(size_t n, size_t numEdges) {
    assert(isDense);

    if (!bitmask[n]) {
      bitmask[n] = true;
      push(n, numEdges);
    }
  }

  void push(size_t n, size_t numEdges) {
    bag.push(n);
    numNodes += 1;
    counts += 1 + numEdges;
  }

  size_t getCount() { return counts.reduce(); }

  size_t getSize() { return numNodes.reduce(); }

  void clear() {
    if (isDense) {
      if (numNodes.reduce() < bitmask.size() / 4) {
        InitializeSmall fn = { this };
        galois::do_all(galois::iterate(bag), fn);
      } else {
        InitializeBig fn = { this };
        galois::on_each(fn);
      }
    }
    bag.clear();
    counts.reset();
    numNodes.reset();

    isDense = false;
  }

  bool contains(size_t n) {
    assert(isDense);
    return bitmask[n];
  }

  bool empty() const { return bag.empty(); }

  void densify() {
    isDense = true;
    if (bitmask.size() == 0) {
      bitmask.create(size);
    }

    Densify fn = { this };
    galois::do_all(galois::iterate(bag), fn);
  }
};

/**
 * Stores graph nodes to execute for {@link Ligra} executor. Unlike {@link
 * GraphNodeBag}, this class stores two bags to facilitate bulk-synchronous
 * processing.
 */
template<unsigned int BlockSize = 0>
class GraphNodeBagPair {
  GraphNodeBag<BlockSize> bag1;
  GraphNodeBag<BlockSize> bag2;
  int curp;
public:
  typedef GraphNodeBag<BlockSize> bag_type;

  GraphNodeBagPair(size_t n): bag1(n), bag2(n), curp(0) { }

  GraphNodeBag<BlockSize>& cur() { return (*this)[curp]; }
  GraphNodeBag<BlockSize>& next() { return (*this)[(curp+1) & 1]; }
  void swap() {
    curp = (curp + 1) & 1;
    next().clear();
  }
  GraphNodeBag<BlockSize>& operator[](int i) {
    if (i == 0)
      return bag1;
    else
      return bag2;
  }
};

}
#endif
