#ifndef APPS_PAGERANK_PAGERANK_H
#define APPS_PAGERANK_PAGERANK_H

#include "llvm/Support/CommandLine.h"

//! d is the damping factor. Alpha is the prob that user will do a random jump,
//! i.e., 1 - d
static const float alpha = 1.0 - 0.85;
static const float alpha2 =
    0.85; // Joyce changed to this which is a usual way to define alpha.

//! maximum relative change until we deem convergence
static const float tolerance = 0.01;
// static const float tolerance = 0.00001; // Joyce

// ICC v13.1 doesn't yet support std::atomic<float> completely, emmulate its
// behavor with std::atomic<int>
struct atomic_float : public std::atomic<int> {
  static_assert(sizeof(int) == sizeof(float),
                "int and float must be the same size");

  float atomicIncrement(float value) {
    while (true) {
      union {
        float as_float;
        int as_int;
      } oldValue = {read()};
      union {
        float as_float;
        int as_int;
      } newValue = {oldValue.as_float + value};
      if (this->compare_exchange_strong(oldValue.as_int, newValue.as_int))
        return newValue.as_float;
    }
  }

  float read() {
    union {
      int as_int;
      float as_float;
    } caster = {this->load(std::memory_order_relaxed)};
    return caster.as_float;
  }

  void write(float v) {
    union {
      float as_float;
      int as_int;
    } caster = {v};
    this->store(caster.as_int, std::memory_order_relaxed);
  }
};

struct PNode {
  float value;
  atomic_float accum;
  PNode() {}

  float getPageRank() { return value; }
};

extern llvm::cl::opt<unsigned int> memoryLimit;
extern llvm::cl::opt<std::string> filename;
extern llvm::cl::opt<unsigned int> maxIterations;

#endif
