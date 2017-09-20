/** Worklists -*- C++ -*-
 * @file
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
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#ifndef GALOIS_WORKLIST_WORKLIST_H
#define GALOIS_WORKLIST_WORKLIST_H

#include "galois/optional.h"

#include "AltChunked.h"
#include "BulkSynchronous.h"
#include "Chunked.h"
#include "Simple.h"
#include "LocalQueue.h"
#include "Obim.h"
#include "OrderedList.h"
#include "OwnerComputes.h"
#include "StableIterator.h"

namespace galois {
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
 * galois::for_each(begin, end, fn, galois::wl<galois::worklists::dChunkedFIFO<32>>());
 * \endcode
 */
namespace worklists {
namespace { // don't pollute the symbol table with the example

// Worklists may not be copied.
// All classes (should) conform to:
template<typename T>
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
  template<typename _T>
  using retype = AbstractWorkList<_T>;

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
  galois::optional<value_type> pop();

  /**
   * (optional) Returns true if the worklist is empty. Called infrequently
   * by scheduler after pop has failed. Good way to split retrieving work
   * into pop (fast path) and empty (slow path).
   */
  bool empty();
};

} // end namespace anonymous
} // end namespace worklists
} // end namespace galois

#endif
