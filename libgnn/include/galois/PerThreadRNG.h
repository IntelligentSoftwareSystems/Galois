#pragma once
#include <random>
#include "galois/substrate/PerThreadStorage.h"
#include "galois/GNNTypes.h"

namespace galois {

//! Per thread RNG object for generating numbers in parallel
class PerThreadRNG {
public:
  //! Default seed 0, default distribution 0 to 1
  PerThreadRNG() : distribution_{0.0, 1.0} {};
  //! User specified range
  PerThreadRNG(float begin, float end) : distribution_{begin, end} {};
  //! Returns a random number between numbers specified during init
  GNNFloat GetRandomNumber() {
    return (*distribution_.getLocal())(*engine_.getLocal());
  }
  //! Return true or false based on some dropout rate
  bool DoBernoulli(float dropout_rate) {
    return (GetRandomNumber() > dropout_rate) ? 1 : 0;
  }

private:
  //! Per thread generator of random
  galois::substrate::PerThreadStorage<std::default_random_engine> engine_;
  //! Per thread distribution of random
  galois::substrate::PerThreadStorage<std::uniform_real_distribution<GNNFloat>>
      distribution_;
};

} // namespace galois
