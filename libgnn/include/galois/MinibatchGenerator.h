#pragma once

#include "galois/GNNTypes.h"
#include "galois/Logging.h"
#include <ctime>
#include <random>
#include <algorithm>

namespace galois {

//! Generates minibatchs given a mask for the class of things to generate
//! the minibatch for
class MinibatchGenerator {
public:
  MinibatchGenerator(const GNNMask& mask_to_minibatch, size_t minibatch_size,
                     size_t master_bound)
      : mask_to_minibatch_{mask_to_minibatch}, minibatch_size_{minibatch_size},
        current_position_{0}, master_bound_{master_bound} {
    // set seed based on time then initialize random generate with rand()
    srand(time(NULL));
    rand_generator_ = std::make_unique<std::mt19937>(rand());
    GALOIS_LOG_ASSERT(master_bound_ <= mask_to_minibatch_.size());
  }

  void GetNextMinibatch(std::vector<char>* batch_mask) {
    if (!shuffle_mode_) {
      OriginalGetNextMinibatch(batch_mask);
    } else {
      ShuffleGetNextMinibatch(batch_mask);
    }
  }

  //! True if no more minibatches from this generator
  bool NoMoreMinibatches() {
    if (!shuffle_mode_) {
      return current_position_ == master_bound_;
    } else {
      return current_position_ >= all_indices_.size();
    }
  }

  //! Reset the only state (a position bit)
  void ResetMinibatchState() {
    current_position_ = 0;
    if (shuffle_mode_) {
      std::shuffle(all_indices_.begin(), all_indices_.end(), *rand_generator_);
    }
  }

  void ShuffleMode() {
    if (!shuffle_mode_) {
      shuffle_mode_ = true;
      all_indices_.reserve(master_bound_);
      // setup all set indices for the minibatch
      for (size_t pos = 0; pos < master_bound_; pos++) {
        if (mask_to_minibatch_[pos]) {
          all_indices_.emplace_back(pos);
        }
      }
      // shuffle it
      std::shuffle(all_indices_.begin(), all_indices_.end(), *rand_generator_);
      printf("Number of things in minibatch generator is %lu\n",
             all_indices_.size());
    }
  }

private:
  const GNNMask& mask_to_minibatch_;
  size_t minibatch_size_;
  size_t current_position_;
  size_t master_bound_;
  std::vector<uint32_t> all_indices_;
  bool shuffle_mode_ = false;
  std::unique_ptr<std::mt19937> rand_generator_;

  void OriginalGetNextMinibatch(std::vector<char>* batch_mask);
  void ShuffleGetNextMinibatch(std::vector<char>* batch_mask);
};

} // namespace galois
