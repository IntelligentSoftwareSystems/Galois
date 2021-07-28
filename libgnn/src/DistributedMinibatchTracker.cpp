#include <algorithm>
#include "galois/DistributedMinibatchTracker.h"

size_t galois::DistributedMinibatchTracker::GetNumberForNextMinibatch() {
  galois::StatTimer timer("DistributedGetNumberForNextMinibatch");
  timer.start();

  uint32_t my_share = int_distribution_(rng_object_);
  if (current_num_on_hosts_[my_host_id_] == 0) {
    my_share = 0;
  }
  sampled_num_on_hosts_[my_host_id_] = my_share;
  // sync and post process *the same way on all hosts*
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_UINT32_T,
                static_cast<void*>(sampled_num_on_hosts_.data()), 1,
                MPI_UINT32_T, MPI_COMM_WORLD);

  for (size_t i = 1; i < sampled_num_on_hosts_.size(); i++) {
    sampled_num_on_hosts_[i] += sampled_num_on_hosts_[i - 1];
  }
  uint32_t share_sum = sampled_num_on_hosts_.back();
  uint32_t num_per_unit =
      std::max((total_minibatch_size_ + share_sum - 1) / share_sum, size_t{1});

  size_t my_value_to_take    = 0;
  size_t extra_to_distribute = 0;
  size_t sanity_sum          = 0;
  for (size_t host = 0; host < num_hosts_; host++) {
    // determine how much to pull from each host based on sampled number
    uint32_t start;
    uint32_t end;
    if (host == 0) {
      start = 0;
      end   = std::min(num_per_unit * sampled_num_on_hosts_[host],
                     (uint32_t)total_minibatch_size_);
    } else if (host == (num_hosts_ - 1)) {
      start = std::min(num_per_unit * sampled_num_on_hosts_[host - 1],
                       (uint32_t)total_minibatch_size_);
      end   = total_minibatch_size_;
    } else {
      start = std::min(num_per_unit * sampled_num_on_hosts_[host - 1],
                       (uint32_t)total_minibatch_size_);
      end   = std::min(num_per_unit * sampled_num_on_hosts_[host],
                     (uint32_t)total_minibatch_size_);
    }

    uint32_t proposed_to_take = end - start;
    sanity_sum += proposed_to_take;

    // is there actually that much? check
    uint32_t actual_to_take =
        std::min(proposed_to_take, current_num_on_hosts_[host]);

    if (actual_to_take < proposed_to_take) {
      extra_to_distribute += proposed_to_take - actual_to_take;
    }
    // update counts, then return
    current_num_on_hosts_[host] -= actual_to_take;
    if (host == my_host_id_) {
      my_value_to_take = actual_to_take;
    }
  }
  GALOIS_LOG_ASSERT(sanity_sum == total_minibatch_size_);

  // redistribute extra to hosts with remaining
  for (size_t host = 0; host < num_hosts_; host++) {
    if (!extra_to_distribute) {
      // leave when there is nothing selse to distribute
      break;
    }

    size_t left_on_host = current_num_on_hosts_[host];
    if (left_on_host) {
      uint32_t to_take = std::min(extra_to_distribute, left_on_host);
      extra_to_distribute -= to_take;
      current_num_on_hosts_[host] -= to_take;
      // update my count as neccessary
      if (my_host_id_ == host) {
        my_value_to_take += to_take;
      }
    }
  }
  timer.stop();

  return my_value_to_take;
}
