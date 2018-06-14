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
  std::atomic<int64_t> currentMemUsage; //!< mem usage of send and receive buffers
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
  inline void decrementMemUsage(uint64_t size) {
    currentMemUsage -= size;
  }

  /**
   * Reset mem usage and max mem usage to 0.
   */
  inline void resetMemUsage() {
    currentMemUsage = 0;
    maxMemUsage = 0;
  }

  /**
   * Get max mem usage.
   *
   * @returns maximum memory usage tracked so far
   */
  inline int64_t getMaxMemUsage() const {
    return maxMemUsage;
  }
};

} // namespace runtime
} // namespace galois
