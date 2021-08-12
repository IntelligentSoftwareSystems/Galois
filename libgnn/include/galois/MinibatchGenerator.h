#pragma once

#include "galois/GNNTypes.h"
#include "galois/Logging.h"
#include "galois/graphs/DistributedGraph.h"
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
    // srand(1);
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

  // void GetNextMinibatch(std::vector<char>* batch_mask, size_t num_to_get) {
  //  if (!shuffle_mode_) {
  //    // TODO
  //    GALOIS_LOG_FATAL("not yet implemented");
  //  } else {
  //    ShuffleGetNextMinibatch(batch_mask, num_to_get);
  //  }
  //}

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

  //! Original shuffle mode in which every host only considers locally owned
  //! training nodes in the all indices array
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

  //! Distributed shuffle mode: all hosts create array with ALL global training
  //! node IDs and initialize shuffler to same seed. All hosts then advance it
  //! at the same time, resulting in a consistent minibatch across all hosts.
  //! Will *NOT* balance # of training nodes done on a host each minibatch
  //! unlike original shuffle.
  void
  DistributedShuffleMode(const galois::graphs::DistGraph<char, void>& graph,
                         GNNMask& global_training_mask,
                         size_t total_train_nodes) {
    if (!shuffle_mode_) {
      shuffle_mode_ = true;
      all_indices_.reserve(total_train_nodes);
      // setup all set indices for the minibatch
      for (size_t pos = 0; pos < global_training_mask.size(); pos++) {
        if (global_training_mask[pos]) {
          if (graph.isLocal(pos)) {
            all_indices_.emplace_back(graph.getLID(pos));
          } else {
            // size is greater than LID; use this as a "not present"
            all_indices_.emplace_back(graph.size());
          }
        }
      }
      GALOIS_LOG_VASSERT(all_indices_.size() == total_train_nodes,
                         "{} vs right {}", all_indices_.size(),
                         total_train_nodes);

      // shuffle it
      std::shuffle(all_indices_.begin(), all_indices_.end(), *rand_generator_);
      printf("Number of things in minibatch generator is %lu\n",
             all_indices_.size());
    }
  }

  //! Total number of nodes that can be minibatched by this minibatch
  //! generator on this host
  size_t ShuffleMinibatchTotal() {
    if (shuffle_mode_) {
      return all_indices_.size();
    } else {
      return 0;
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

  // Do not use these unless you know what they're doing
  void DistributedShuffleGetNextMinibatch(std::vector<char>* batch_mask);
  void DistributedShuffleGetNextMinibatch(std::vector<char>* batch_mask,
                                          size_t num_to_get);
};

} // namespace galois
