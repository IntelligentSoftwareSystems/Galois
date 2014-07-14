/** Worklists -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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
#ifndef GALOIS_RUNTIME_WORKLIST_H
#define GALOIS_RUNTIME_WORKLIST_H

#include "Galois/optional.h"

#include "AltChunked.h"
#include "BulkSynchronous.h"
#include "Chunked.h"
#include "Fifo.h"
#include "GFifo.h"
#include "Lifo.h"
#include "LocalQueue.h"
#include "Obim.h"
#include "OrderedList.h"
#include "OwnerComputes.h"
#include "StableIterator.h"

namespace Galois {
/**
 * Scheduling policies for Galois iterators. Unless you have very specific
 * scheduling requirement, {@link dChunkedLIFO} or {@link dChunkedFIFO} is a
 * reasonable scheduling policy. If you need approximate priority scheduling,
 * use {@link OrderedByIntegerMetric}. For debugging, you may be interested
 * in {@link FIFO} or {@link LIFO}, which try to follow serial order exactly.
 *
 * The way to use a worklist is to pass it as a template parameter to
 * {@link for_each()}. For example,
 *
 * \code
 * Galois::for_each(begin, end, fn, Galois::wl<Galois::WorkList::dChunkedFIFO<32>>());
 * \endcode
 */
namespace WorkList {
namespace { // don't pollute the symbol table with the example

// Worklists may not be copied.
// Worklists should be default instantiatable
// All classes (should) conform to:
template<typename T, bool Concurrent>
class AbstractWorkList {
  AbstractWorkList(const AbstractWorkList&) = delete;
  const AbstractWorkList& operator=(const AbstractWorkList&) = delete;

public:
  AbstractWorkList() { }

  //! T is the value type of the WL
  typedef T value_type;

  //! Changes the concurrency flag
  template<bool _concurrent>
  struct rethread { typedef AbstractWorkList<T, _concurrent> type; };

  //! Changes the type the worklist holds
  template<typename _T>
  struct retype { typedef AbstractWorkList<_T, Concurrent> type; };

  //! Pushes a value onto the queue
  void push(const value_type& val);

  //! Pushes a range onto the queue
  template<typename Iter>
  void push(Iter b, Iter e);

  /**
   * Pushes initial range onto the queue. Called with the same b and e on each
   * thread
   */
  template<typename RangeTy>
  void push_initial(const RangeTy&);

  //! Pops a value from the queue.
  Galois::optional<value_type> pop();

  /**
   * (optional) Returns true if the worklist is empty. Called infrequently
   * by scheduler after pop has failed. Good way to split retrieving work
   * into pop (fast path) and empty (slow path).
   */
  bool empty();
};

} // end namespace anonymous
} // end namespace WorkList
} // end namespace Galois

#endif
