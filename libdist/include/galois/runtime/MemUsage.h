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

/*
 */

/**
 * @file MemUsage.h
 *
 * Contains MemUsageTracker, a class that tracks memory usage throughout
 * runtime of a program of send/receive buffers.
 */

#pragma once
#include <atomic>
namespace galois {
namespace runtime {

/**
 * Class that tracks memory usage (mainly of send and receive buffers).
 */
class MemUsageTracker {
  std::atomic<int64_t>
      currentMemUsage; //!< mem usage of send and receive buffers
  int64_t maxMemUsage; //!< max mem usage of send and receive buffers

public:
  //! Default constructor initializes everything to 0.
  MemUsageTracker() : currentMemUsage(0), maxMemUsage(0) {}

  /**
   * Increment memory usage.
   *
   * @param size amount to increment mem usage by
   */
  inline void incrementMemUsage(uint64_t size) {
    currentMemUsage += size;
    if (currentMemUsage > maxMemUsage)
      maxMemUsage = currentMemUsage;
  }

  /**
   * Decrement memory usage.
   *
   * @param size amount to decrement mem usage by
   */
  inline void decrementMemUsage(uint64_t size) { currentMemUsage -= size; }

  /**
   * Reset mem usage and max mem usage to 0.
   */
  inline void resetMemUsage() {
    currentMemUsage = 0;
    maxMemUsage     = 0;
  }

  /**
   * Get max mem usage.
   *
   * @returns maximum memory usage tracked so far
   */
  inline int64_t getMaxMemUsage() const { return maxMemUsage; }
};

} // namespace runtime
} // namespace galois
