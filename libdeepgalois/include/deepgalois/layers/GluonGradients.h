#ifndef __GLUON_GRADIENTS__
#define __GLUON_GRADIENTS__

#include "galois/gstl.h"
#include "galois/runtime/Network.h"
#include "deepgalois/types.h"

namespace deepgalois {

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
  //! number of weight gradients
  size_t _numWeights;
  //! number of gradients this host is responsible for
  size_t _numOwned;

  //! My host ID
  unsigned _myHost;
  //! Total num hosts in system
  unsigned _totalHosts;

  //! first node I own
  unsigned _beginMaster;
  //! last node I own (contiguous chunk)
  unsigned _endMaster;

  //! my nodes whose's masters are on other hosts; global ids
  std::vector<std::vector<size_t>> _mirrorNodes;
  //! nodes that are mirrors on this host
  std::vector<std::pair<uint32_t, uint32_t>> _mirrorRanges;
public:
  /**
   * Save weight gradients + number of them (i.e. size).
   * Then setup mirror metadata for Gluon to use during setup.
   */
  GluonGradients(GradientVecType& gradients, size_t numWeights)
      : _gradients(gradients), _numWeights(numWeights) {
    _myHost = galois::runtime::getSystemNetworkInterface().ID;
    _totalHosts = galois::runtime::getSystemNetworkInterface().Num;

    // allocate a vector for each host
    _mirrorNodes.resize(_totalHosts);

    // loop through distribution of weights to hosts
    for (unsigned h = 0; h < _totalHosts; h++) {
      std::pair<size_t, size_t> curRange =
        galois::block_range((size_t)0, _numWeights, h, _totalHosts);

      if (h != _myHost) {
        // setup mirrors for the host h which is just the list of IDs
        size_t curW = curRange.first;
        size_t lastW = curRange.second;
        size_t numW = lastW - curW;

        // set mirrors for host h
        _mirrorNodes[h].reserve(numW);
        for (; curW < lastW; curW++) {
          _mirrorNodes[h].push_back(curW);
        }
      } else {
        // these belong to this host; save, then mirror ranges can be
        // calculated from this
        _beginMaster = curRange.first;
        _endMaster = curRange.second;
        _numOwned = _endMaster - _beginMaster;

        // first range is 0 to begin master
        if (_beginMaster > 0) {
          galois::gInfo("[", _myHost, "] Mirror range ", 0, " to ",
                        _beginMaster);
          _mirrorRanges.emplace_back(0, _beginMaster);
        }

        // second range is endMaster to end
        if (_endMaster < _numWeights) {
          galois::gInfo("[", _myHost, "] Mirror range ", _endMaster, " to ",
                        _numWeights);
          _mirrorRanges.emplace_back(_endMaster, _numWeights);
        }
      }
    }

    galois::gInfo("[", _myHost, "] This host owns ", _beginMaster, " to ",
                  _endMaster);
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
  size_t numMasters() const {
    return _numOwned;
  }

  //! Return host ID
  unsigned myHostID() const {
    return _myHost;
  }

  //! Return num hosts in the system
  unsigned numHosts() const {
    return _totalHosts;
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
  GradientType getData(uint32_t w) const {
    return _gradients[w];
  }

  //! Return ranges for mirrors (unowned nodes)
  const std::vector<std::pair<uint32_t, uint32_t>>& getMirrorRanges() const {
    return _mirrorRanges;
  }

  //! Return mirror nodes for each host from this host's point of view
  std::vector<std::vector<size_t>>& getMirrorNodes() {
    return _mirrorNodes;
  }

  //! clears the vector
  // TODO return to this when we start distributing on GPUs; wrapper
  // end probably shouldn't be managing this MAYBE
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

}

#endif // end header guard
