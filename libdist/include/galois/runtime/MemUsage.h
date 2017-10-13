#pragma once
namespace galois {
namespace runtime {
class MemUsageTracker {
  uint64_t currentMemUsage; // of send and receive buffers
  uint64_t maxMemUsage; // of send and receive buffers

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
  inline uint64_t getMaxMemUsage() const {
    return maxMemUsage;
  }
};
} //namespace runtime
} //namespace galois
