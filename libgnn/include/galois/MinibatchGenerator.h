#pragma once

#include "galois/GNNTypes.h"

namespace galois {

//! Generates minibatchs given a mask for the class of things to generate
//! the minibatch for
class MinibatchGenerator {
public:
  MinibatchGenerator(const GNNMask& mask_to_minibatch, size_t minibatch_size)
      : mask_to_minibatch_{mask_to_minibatch}, minibatch_size_{minibatch_size} {
  }
  void GetNextMinibatch(std::vector<char>* batch_mask);
  //! True if no more minibatches from this generator
  bool NoMoreMinibatches() {
    return current_position_ == mask_to_minibatch_.size();
  }
  //! Reset the only state (a position bit)
  void ResetMinibatchState() { current_position_ = 0; }

private:
  const GNNMask& mask_to_minibatch_;
  size_t minibatch_size_;
  size_t current_position_{0};
};

} // namespace galois
