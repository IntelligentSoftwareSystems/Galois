#pragma once

#include "galois/GNNTypes.h"
#include "galois/Logging.h"

namespace galois {

//! Generates minibatchs given a mask for the class of things to generate
//! the minibatch for
class MinibatchGenerator {
public:
  MinibatchGenerator(const GNNMask& mask_to_minibatch, size_t minibatch_size,
                     size_t master_bound)
      : mask_to_minibatch_{mask_to_minibatch}, minibatch_size_{minibatch_size},
        current_position_{0}, master_bound_{master_bound} {
    GALOIS_LOG_ASSERT(master_bound_ <= mask_to_minibatch_.size());
  }
  void GetNextMinibatch(std::vector<char>* batch_mask);
  //! True if no more minibatches from this generator
  bool NoMoreMinibatches() { return current_position_ == master_bound_; }
  //! Reset the only state (a position bit)
  void ResetMinibatchState() { current_position_ = 0; }

private:
  const GNNMask& mask_to_minibatch_;
  size_t minibatch_size_;
  size_t current_position_;
  size_t master_bound_;
};

} // namespace galois
