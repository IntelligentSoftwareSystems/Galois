#ifndef __GLUON_GRADIENTS__
#define __GLUON_GRADIENTS__

#include "deepgalois/types.h"

/**
 * Wraps the weight gradients and provides an interface for Gluon to
 * synchronize them during distributed execution.
 */
class GluonGradients {
private:
  //! Data type used for gradients
  using GradientType = float_t;
  //! type that's being used by the gradient vector
  using GradientVecType = vec_t;

  GradientVecType& _gradients;
  size_t _numWeights;
  size_t _numOwned;

  //! my nodes whose's masters are on other hosts; global ids
  std::vector<std::vector<size_t>> mirrorNodes;
  // TODO save mirror ranges here as well

public:
  /**
   * Save weight gradients + number of them (i.e. size).
   * Then setup mirror metadata for Gluon to use during setup.
   */
  GluonGradients(GradientVecType& gradients, size_t numWeights)
      : _gradients(gradients), _numWeights(numWeights) {
  }

  //! Size is number of weights
  size_t size() const {
    return _numWeights;
  }

  //! Global size is number of weights
  size_t globalSize() const {
    return _numWeights;
  }

  //! Return the weights owned by this host
  size_t numMasters const {
    return _numOwned;
  }

  //! GID is same as LID since all hosts have all weights
  uint32_t getGID(const uint32_t nodeID) const {
    return nodeID;
  }

  //! LID is same as GID since all hosts have all weights
  uint32_t getLID(const uint32_t nodeID) const {
    return nodeID;
  }

  //! Return local weight w
  GradientType getData(uint32_t w) {
    return _gradients[w];
  }

  std::vector<std::pair<uint32_t, uint32_t>> getMirrorRanges() const {
    // TODO
  }

  //! Return mirror nodes for each host from this host's point of view
  std::vector<std::vector<size_t>>& getMirrorNodes() {
    return mirrorNodes;
  }

  //! clears the vector
  // TODO return to this when we start distributing on GPUs
  void deallocate() {
    _gradients.clear();
  }

  // Essentially no-op functions follow

  //! no nodes with edges
  size_t getNumNodesWithEdges() {
    return 0;
  }

  //! No edges; not a vertex cut
  bool is_vertex_cut() const {
    return false;
  }

  //! no edges, return 0
  unsigned edge_begin(uint32_t dummy) {
    return 0;
  }

  //! no edges, return 0
  unsigned edge_end(uint32_t dummy) {
    return 0;
  }

  //! no edges, return 0
  unsigned getEdgeDst(uint32_t dummy) {
    return 0;
  }

  //! no edges, return 0
  unsigned getEdgeData(uint32_t dummy) {
    return 0;
  }
};

#endif // end header guard
