#pragma once

#include "galois/graphs/GNNGraph.h"
#include <random>

namespace galois {

//! Tracks how many nodes remain to be chosen from every host's
//! minibatch and also determines how many to pull from this
//! particular host every iteration.
class DistributedMinibatchTracker {
public:
  DistributedMinibatchTracker(size_t my_host_id, size_t num_hosts,
                              size_t my_minibatch_nodes,
                              size_t local_minibatch_size)
      : my_host_id_{my_host_id}, num_hosts_{num_hosts},
        local_minibatch_size_{local_minibatch_size},
        total_minibatch_size_{local_minibatch_size_ * num_hosts_},
        complete_hosts_{0}, rng_object_{(long unsigned)rand() *
                                        (my_host_id_ + 1)},
        int_distribution_{1, 10} {
    max_num_on_hosts_.resize(num_hosts_, 0);
    current_num_on_hosts_.resize(num_hosts_, 0);
    sampled_num_on_hosts_.resize(num_hosts_, 0);
    max_num_on_hosts_[my_host_id_] = my_minibatch_nodes;

    // all reduce so all get the right values
    // TODO technically all reduce would be sending unnecessary 0s
    // but whatever this is relatively small
    MPI_Allreduce(MPI_IN_PLACE, static_cast<void*>(max_num_on_hosts_.data()),
                  num_hosts_, MPI_UINT32_T, MPI_SUM, MPI_COMM_WORLD);
  }

  //! Reset epoch = set all current sampled back to initial state
  void ResetEpoch() {
    galois::do_all(
        galois::iterate(size_t{0}, num_hosts_), [&](size_t host_id_) {
          current_num_on_hosts_[host_id_] = max_num_on_hosts_[host_id_];
        });
    complete_hosts_ = 0;
  }

  size_t GetNumberForNextMinibatch();

  bool OutOfWork() {
    GALOIS_LOG_FATAL("NEED TO IMPLEMENT");
    return complete_hosts_ == num_hosts_;
  }

private:
  size_t my_host_id_;
  size_t num_hosts_;
  size_t local_minibatch_size_;
  size_t total_minibatch_size_;
  unsigned complete_hosts_;

  std::mt19937 rng_object_;
  std::uniform_int_distribution<unsigned> int_distribution_;
  //! Maximum amount of nodes on each host; used to reset state
  std::vector<uint32_t> max_num_on_hosts_;
  //! Current number of nodes left on each host; used to know how
  //! to sample on each host
  std::vector<uint32_t> current_num_on_hosts_;
  //! Vector to be sync'd indicating how many to grab from each
  //! batch
  std::vector<uint32_t> sampled_num_on_hosts_;
};

} // namespace galois
