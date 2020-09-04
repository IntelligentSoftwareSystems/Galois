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

#ifndef GALOIS_WORKLIST_WORKLIST_H
#define GALOIS_WORKLIST_WORKLIST_H

#include "galois/config.h"
#include "galois/optional.h"
#include "galois/worklists/AdaptiveObim.h"
#include "galois/worklists/PerThreadChunk.h"
#include "galois/worklists/BulkSynchronous.h"
#include "galois/worklists/Chunk.h"
#include "galois/worklists/Simple.h"
#include "galois/worklists/LocalQueue.h"
#include "galois/worklists/Obim.h"
#include "galois/worklists/OrderedList.h"
#include "galois/worklists/OwnerComputes.h"
#include "galois/worklists/StableIterator.h"

namespace galois {
/**
 * Scheduling policies for Galois iterators. Unless you have very specific
 * scheduling requirement, {@link PerSocketChunkLIFO} or {@link
 * PerSocketChunkFIFO} is a reasonable scheduling policy. If you need
 * approximate priority scheduling, use {@link OrderedByIntegerMetric}. For
 * debugging, you may be interested in {@link FIFO} or {@link LIFO}, which try
 * to follow serial order exactly.
 *
 * The way to use a worklist is to pass it as a template parameter to
 * {@link for_each()}. For example,
 *
 * \code
 * galois::for_each(galois::iterate(beg,end), fn,
 * galois::wl<galois::worklists::PerSocketChunkFIFO<32>>()); \endcode
 */
namespace worklists {
namespace { // don't pollute the symbol table with the example

// Worklists may not be copied.
// All classes (should) conform to:
template <typename T>
class AbstractWorkList {
  AbstractWorkList(const AbstractWorkList&) = delete;
  const AbstractWorkList& operator=(const AbstractWorkList&) = delete;

public:
  AbstractWorkList();

  //! Optional paramaterized Constructor
  //! parameters can be whatever
  AbstractWorkList(int, double, char*);

  //! T is the value type of the WL
  typedef T value_type;

  //! Changes the type the worklist holds
  template <typename _T>
  using retype = AbstractWorkList<_T>;

  //! Pushes a value onto the queue
  void push(const value_type& val);

  //! Pushes a range onto the queue
  template <typename Iter>
  void push(Iter b, Iter e);

  /**
   * Pushes initial range onto the queue. Called with the same b and e on each
   * thread
   */
  template <typename RangeTy>
  void push_initial(const RangeTy&);

  //! Pops a value from the queue.
  galois::optional<value_type> pop();

  /**
   * (optional) Returns true if the worklist is empty. Called infrequently
   * by scheduler after pop has failed. Good way to split retrieving work
   * into pop (fast path) and empty (slow path).
   */
  bool empty();
};

} // namespace
} // end namespace worklists
} // end namespace galois

#endif
