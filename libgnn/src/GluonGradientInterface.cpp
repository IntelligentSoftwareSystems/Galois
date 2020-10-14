#include "galois/layers/GluonGradientInterface.h"

galois::GluonGradientInterface::GluonGradientInterface(
    std::vector<GNNFloat>& gradients)
    : gradients_(gradients), num_weights_(gradients_.size()) {
  size_t my_host   = galois::runtime::getSystemNetworkInterface().ID;
  size_t num_hosts = galois::runtime::getSystemNetworkInterface().Num;

  // allocate a vector for each host
  mirror_nodes_.resize(num_hosts);

  // loop through distribution of weights to hosts
  for (unsigned h = 0; h < num_hosts; h++) {
    std::pair<size_t, size_t> cur_range =
        galois::block_range((size_t)0, num_weights_, h, num_hosts);

    if (h != my_host) {
      // setup mirrors for the host h which is just the list of IDs
      size_t current_weight   = cur_range.first;
      size_t last_weight      = cur_range.second;
      size_t num_host_weights = last_weight - current_weight;

      // set mirrors for host h
      mirror_nodes_[h].reserve(num_host_weights);
      for (; current_weight < last_weight; current_weight++) {
        mirror_nodes_[h].push_back(current_weight);
      }
    } else {
      // these belong to this host; save, then mirror ranges can be
      // calculated from this
      begin_master_ = cur_range.first;
      end_master_   = cur_range.second;
      num_owned_    = end_master_ - begin_master_;

      // first range is 0 to begin master
      if (begin_master_ > 0) {
        mirror_ranges_.emplace_back(0, begin_master_);
      }

      // second range is endMaster to end
      if (end_master_ < num_weights_) {
        mirror_ranges_.emplace_back(end_master_, num_weights_);
      }
    }
  }

  galois::gInfo("[", my_host, "] Weight gradients: this host owns ",
                begin_master_, " to ", end_master_);
}
