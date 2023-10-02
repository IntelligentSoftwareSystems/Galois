// random number generators for CPU
#pragma once

#include <random>
#include "galois/Galois.h"
#include "deepgalois/GraphTypes.h"

namespace deepgalois {

class PerThreadRNG {
  galois::substrate::PerThreadStorage<std::default_random_engine> engine;
  galois::substrate::PerThreadStorage<std::uniform_real_distribution<float_t>>
      distribution;

public:
  //! init distribution
  PerThreadRNG() : distribution{0.0, 1.0} {};

  //! thread local RNG float from 0 to 1
  float_t get_number() {
    float_t num = (*distribution.getLocal())(*engine.getLocal());
    return num;
  }
};

class random_generator {
public:
  static random_generator& get_instance() {
    static random_generator instance;
    return instance;
  }
  std::mt19937& operator()() { return gen_; }
  void set_seed(unsigned int seed) { gen_.seed(seed); }

private:
  random_generator() : gen_(1) {}
  std::mt19937 gen_;
};

template <typename T>
inline typename std::enable_if<std::is_integral<T>::value, T>::type
uniform_rand(T min, T max) {
  std::uniform_int_distribution<T> dst(min, max);
  return dst(random_generator::get_instance()());
}

template <typename T>
inline typename std::enable_if<std::is_floating_point<T>::value, T>::type
uniform_rand(T min, T max) {
  std::uniform_real_distribution<T> dst(min, max);
  return dst(random_generator::get_instance()());
}
} // namespace deepgalois
