#include <algorithm>
#include "galois/DistributedMinibatchTracker.h"

size_t galois::DistributedMinibatchTracker::GetNumberForNextMinibatch() {
  galois::StatTimer timer("DistributedGetNumberForNextMinibatch");
  timer.start();

  // TODO
  for (size_t i = 0; i < total_minibatch_size_; i++) {
    // pick a host, increment
    unsigned chosen_host = int_distribution_(rng_object_);
    assert(chosen_host < num_hosts_);
    sampled_num_on_hosts_[chosen_host]++;
  }
  // sync and post process *the same way on all hosts*
  MPI_Allreduce(MPI_IN_PLACE, static_cast<void*>(sampled_num_on_hosts_.data()),
                num_hosts_, MPI_UINT32_T, MPI_SUM, MPI_COMM_WORLD);

  size_t to_return              = 0;
  uint32_t leftover_to_allocate = 0;

  // TODO parallel?
  for (size_t i = 0; i < num_hosts_; i++) {
    uint32_t proposed_to_sample = sampled_num_on_hosts_[i];
    size_t left_to_sample       = current_num_on_hosts_[i];
    size_t actual_to_sample     = 0;
    if (left_to_sample > 0) {
      actual_to_sample = std::min(proposed_to_sample, current_num_on_hosts_[i]);

      if (actual_to_sample < left_to_sample && leftover_to_allocate) {
        // more left to sample and we have extra; dump more from extra if
        // possible
        uint32_t what_is_left = left_to_sample - actual_to_sample;
        size_t more_to_sample = std::min(what_is_left, leftover_to_allocate);
        leftover_to_allocate -= more_to_sample;
        actual_to_sample += more_to_sample;
        assert(actual_to_sample <= left_to_sample);
      }
    }
    leftover_to_allocate = proposed_to_sample - actual_to_sample;
    current_num_on_hosts_[i] -= actual_to_sample;

    sampled_num_on_hosts_[i] = 0;
    if (my_host_id_ == i) {
      to_return = actual_to_sample;
    }
  }
  timer.stop();

  if (leftover_to_allocate) {
    // if there are leftovers, it means that there is no more work
    // in this system period
    complete_hosts_ = num_hosts_;
  }

  return to_return;
}
