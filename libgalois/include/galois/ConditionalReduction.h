#ifndef __COND_REDUCTION__
#define __COND_REDUCTION__

#include "galois/Reduction.h"

/**
 *
 *
 */
template<typename Accumulator, bool Active>
class ConditionalAccumulator {
  typename std::conditional<Active, Accumulator, char>::type accumulator;
 public:
  using T = typename Accumulator::AccumType;

  bool isActive() {
    return Active;
  }

  template<bool A = Active, typename std::enable_if<A>::type* = nullptr>
  void reset() {
    accumulator.reset();
  }

  template<bool A = Active, typename std::enable_if<!A>::type* = nullptr>
  void reset() {
    // no-op
  }

  template<bool A = Active, typename std::enable_if<A>::type* = nullptr>
  void update(T newValue) {
    accumulator.update(newValue);
  }

  template<bool A = Active, typename std::enable_if<!A>::type* = nullptr>
  void update(T newValue) {
    // no-op
  }

  template<bool A = Active, typename std::enable_if<A>::type* = nullptr>
  T reduce() {
    return accumulator.reduce();
  }

  template<bool A = Active, typename std::enable_if<!A>::type* = nullptr>
  T reduce() {
    return 0; // TODO choose value that works better regardless of T
  }

  // TODO add the rest of the GSimpleReducible functions?
};

#endif
