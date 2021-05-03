#include "galois/MinibatchGenerator.h"
#include <cassert>

void galois::MinibatchGenerator::GetNextMinibatch(
    std::vector<char>* batch_mask) {
  std::fill(batch_mask->begin(), batch_mask->end(), 0);
  assert(current_position_ <= mask_to_minibatch_.size());
  assert(batch_mask->size() == mask_to_minibatch_.size());
  if (current_position_ >= mask_to_minibatch_.size()) {
    return;
  }

  size_t current_count = 0;
  // start from last positiion
  while (current_position_ < mask_to_minibatch_.size()) {
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
         (current_position_ < mask_to_minibatch_.size())) {
    current_position_++;
  }
}
