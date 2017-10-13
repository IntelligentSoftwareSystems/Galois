#pragma once
#include <atomic>
namespace galois {
namespace runtime {
class MemUsageTracker {
  std::atomic<int64_t> currentMemUsage; // of send and receive buffers
  int64_t maxMemUsage; // of send and receive buffers

public:
  MemUsageTracker() : currentMemUsage(0), maxMemUsage(0) {}
  inline void incrementMemUsage(uint64_t size) {
    currentMemUsage += size;
    if (currentMemUsage > maxMemUsage)
      maxMemUsage = currentMemUsage;
  }
  inline void decrementMemUsage(uint64_t size) {
    currentMemUsage -= size;
  }
  inline void resetMemUsage() {
    currentMemUsage = 0;
    maxMemUsage = 0;
  }
  inline int64_t getMaxMemUsage() const {
    return maxMemUsage;
  }
};
} //namespace runtime
} //namespace galois
