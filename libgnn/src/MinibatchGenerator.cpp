#include "galois/MinibatchGenerator.h"
#include <cassert>

void galois::MinibatchGenerator::OriginalGetNextMinibatch(
    std::vector<char>* batch_mask) {
  assert(current_position_ <= mask_to_minibatch_.size());
  assert(current_position_ <= master_bound_);
  assert(batch_mask->size() == mask_to_minibatch_.size());

  std::fill(batch_mask->begin(), batch_mask->end(), 0);
  if (current_position_ >= master_bound_) {
    return;
  }

  size_t current_count = 0;
  // start from last positiion
  while (current_position_ < master_bound_) {
    if (mask_to_minibatch_[current_position_]) {
      // XXX and a master node; seed nodes only exist locally
      (*batch_mask)[current_position_] = 1;
      current_count++;
    }
    // break when minibatch is large enough
    current_position_++;
    if (current_count == minibatch_size_)
      break;
  }

  // advance current position to next set bit for next call (or to end to detect
  // no more minibatches
  while (!mask_to_minibatch_[current_position_] &&
         (current_position_ < master_bound_)) {
    current_position_++;
  }
}

void galois::MinibatchGenerator::ShuffleGetNextMinibatch(
    std::vector<char>* batch_mask) {
  size_t current_count = 0;
  std::fill(batch_mask->begin(), batch_mask->end(), 0);
  while (current_position_ < all_indices_.size()) {
    (*batch_mask)[all_indices_[current_position_++]] = 1;
    current_count++;
    if (current_count == minibatch_size_)
      break;
  }
}
