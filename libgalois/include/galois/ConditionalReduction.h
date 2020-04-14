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

/**
 * @file ConditionalReduction.h
 *
 * Contains conditional accumulator wrapper (conditionally reduce to provided
 * accumulator)
 */

#ifndef __COND_REDUCTION__
#define __COND_REDUCTION__

#include "galois/config.h"
#include "galois/Reduction.h"

/**
 * A conditional accumulator.
 *
 * @tparam Accumulator accumulator to conditionally reduce
 * @tparam Active determines whether or not reduction will actually do anything
 *
 * @todo add the rest of the GSimpleReducible functions?
 */
template <typename Accumulator, bool Active>
class ConditionalAccumulator {
  //! accumulator type
  typename std::conditional<Active, Accumulator, char>::type accumulator;

public:
  //! type of the value being accumulated
  using T = typename Accumulator::AccumType;

  /**
   * Returns true if the conditionally accumulator is active
   */
  bool isActive() { return Active; }

  /**
   * Reset the accumulator.
   */
  template <bool A = Active, typename std::enable_if<A>::type* = nullptr>
  void reset() {
    accumulator.reset();
  }

  //! no-op
  template <bool A = Active, typename std::enable_if<!A>::type* = nullptr>
  void reset() {
    // no-op
  }

  //! Update the accumulator using the accumulator's update function
  template <bool A = Active, typename std::enable_if<A>::type* = nullptr>
  void update(T newValue) {
    accumulator.update(newValue);
  }

  //! no-op
  template <bool A = Active, typename std::enable_if<!A>::type* = nullptr>
  void update(T newValue) {
    // no-op
  }

  //! Reduce the accumulator and return the reduced value
  template <bool A = Active, typename std::enable_if<A>::type* = nullptr>
  T reduce() {
    return accumulator.reduce();
  }

  //! no-op
  //! @todo choose value that works better regardless of T
  template <bool A = Active, typename std::enable_if<!A>::type* = nullptr>
  T reduce() {
    return 0;
  }
};
#endif
