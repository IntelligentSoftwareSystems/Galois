#pragma once

#include "galois/GNNTypes.h"
#include "galois/gstl.h"
#include "galois/runtime/Network.h"

namespace galois {

// TODO figure out which function calls can be removed without causing compiler
// to complain

//! Wraps a matrix and allows it to be synchronized via Gluon as it provides
//! all the functions Gluon needs.
//! Assumes the matrix is initialized the same way across all hosts (if not
//! they'll all see the same values after the first round of sync anyways)
class GluonGradientInterface {
public:
  // typedefs required by GPU end to build; not actually used anywhere in this
  // class (...at the moment)
  // as such, dummy declarations that don't particularly make sense
  // TODO will likely need to revisit once GPU substrate for this needs to be
  // setup
  using GraphNode     = uint32_t;
  using edge_iterator = boost::counting_iterator<uint64_t>;
  using EdgeType      = char;

  //! Save reference to weight gradients.
  //! Then setup mirror metadata for Gluon to use during setup.
  GluonGradientInterface(std::vector<GNNFloat>& gradients);

  //! Size is number of weights since all hosts own everything
  size_t size() const { return num_weights_; }
  //! Global size is number of weights
  size_t globalSize() const { return num_weights_; }
  //! Return the weights owned by this host
  size_t numMasters() const { return num_owned_; }
  //! GID is same as LID since all hosts have all weights
  uint32_t getGID(const uint32_t node_id) const { return node_id; }
  //! LID is same as GID since all hosts have all weights
  uint32_t getLID(const uint32_t node_id) const { return node_id; }
  //! Return weight w
  GNNFloat& getData(uint32_t w) const { return gradients_[w]; }
  //! Return ranges for mirrors (unowned nodes)
  const std::vector<std::pair<uint32_t, uint32_t>>& getMirrorRanges() const {
    return mirror_ranges_;
  }
  //! Return mirror nodes for each host from this host's point of view
  std::vector<std::vector<size_t>>& getMirrorNodes() { return mirror_nodes_; }

  //////////////////////////////////////////////////////////////////////////////

  // for all that follow, no edges in this sync so most of this returns what
  // you expect
  // size_t getNumNodesWithEdges() const { return 0; }
  bool is_vertex_cut() const { return false; }
  unsigned edge_begin(uint32_t) const { return 0; }
  unsigned edge_end(uint32_t) const { return 0; }
  unsigned getEdgeDst(uint32_t) const { return 0; }
  unsigned getEdgeData(uint32_t) const { return 0; }
  void deallocate() const {};

private:
  //! Reference to gradients that can get synchronized
  std::vector<GNNFloat>& gradients_;
  //! number of weight gradients
  size_t num_weights_;
  //! number of single gradients this host is responsible for
  size_t num_owned_;
  //! First weight that's a master
  size_t begin_master_;
  //! Last weight that's a master
  size_t end_master_;
  //! My nodes whose's masters are on other hosts; global ids
  std::vector<std::vector<size_t>> mirror_nodes_;
  //! nodes that are mirrors on this host
  std::vector<std::pair<uint32_t, uint32_t>> mirror_ranges_;
};

} // namespace galois
