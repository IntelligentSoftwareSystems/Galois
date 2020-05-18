#ifndef _GALOIS_DIST_GENERICPARTS_H
#define _GALOIS_DIST_GENERICPARTS_H

#include "DistributedGraph.h"
#include "BasePolicies.h"
#include <utility>
#include <cmath>
#include <limits>

class NoCommunication : public galois::graphs::ReadMasterAssignment {
public:
  NoCommunication(uint32_t, uint32_t numHosts, uint64_t, uint64_t)
      : galois::graphs::ReadMasterAssignment(0, numHosts, 0, 0) {}

  uint32_t getEdgeOwner(uint32_t src, uint32_t, uint64_t) const {
    return retrieveMaster(src);
  }

  bool noCommunication() { return true; }
  bool isVertexCut() const { return false; }
  void serializePartition(boost::archive::binary_oarchive&) {}
  void deserializePartition(boost::archive::binary_iarchive&) {}
  std::pair<unsigned, unsigned> cartesianGrid() {
    return std::make_pair(0u, 0u);
  }
};

/**
 */
class MiningPolicyNaive : public galois::graphs::ReadMasterAssignment {
public:
  MiningPolicyNaive(uint32_t, uint32_t numHosts, uint64_t, uint64_t,
                    std::vector<uint64_t>&)
      : galois::graphs::ReadMasterAssignment(0, numHosts, 0, 0) {}

  static bool needNodeDegrees() { return false; }

  bool keepEdge(uint32_t src, uint32_t dst) const { return src < dst; }
};

class MiningPolicyDegrees : public galois::graphs::ReadMasterAssignment {
  std::vector<uint64_t>& ndegrees;

public:
  MiningPolicyDegrees(uint32_t, uint32_t numHosts, uint64_t, uint64_t,
                      std::vector<uint64_t>& _ndeg)
      : galois::graphs::ReadMasterAssignment(0, numHosts, 0, 0),
        ndegrees(_ndeg) {}

  static bool needNodeDegrees() { return true; }

  bool keepEdge(uint32_t src, uint32_t dst) const {
    uint64_t sourceDegree = ndegrees[src];
    uint64_t destDegree   = ndegrees[dst];
    if ((destDegree > sourceDegree) ||
        ((destDegree == sourceDegree) && (src < dst))) {
      return true;
    } else {
      return false;
    }
  }
};

////////////////////////////////////////////////////////////////////////////////

class GenericCVC : public galois::graphs::ReadMasterAssignment {
  unsigned numRowHosts;
  unsigned numColumnHosts;
  unsigned _h_offset;

  void factorizeHosts() {
    numColumnHosts = sqrt(_numHosts);

    while ((_numHosts % numColumnHosts) != 0)
      numColumnHosts--;

    numRowHosts = _numHosts / numColumnHosts;
    assert(numRowHosts >= numColumnHosts);

    // if (moreColumnHosts) {
    //  std::swap(numRowHosts, numColumnHosts);
    //}

    if (_hostID == 0) {
      galois::gPrint("Cartesian grid: ", numRowHosts, " x ", numColumnHosts,
                     "\n");
    }
  }

  //! Returns the grid row ID of this host
  unsigned gridRowID() const { return (_hostID / numColumnHosts); }
  //! Returns the grid row ID of the specified host
  unsigned gridRowID(unsigned id) const { return (id / numColumnHosts); }
  //! Returns the grid column ID of this host
  unsigned gridColumnID() const { return (_hostID % numColumnHosts); }
  //! Returns the grid column ID of the specified host
  unsigned gridColumnID(unsigned id) const { return (id % numColumnHosts); }

  //! Find the column of a particular node
  unsigned getColumnOfNode(uint64_t gid) const {
    return gridColumnID(retrieveMaster(gid));
  }

public:
  GenericCVC(uint32_t hostID, uint32_t numHosts, uint64_t numNodes,
             uint64_t numEdges)
      : galois::graphs::ReadMasterAssignment(hostID, numHosts, numNodes,
                                             numEdges) {
    factorizeHosts();
    _h_offset = gridRowID() * numColumnHosts;
  }

  uint32_t getEdgeOwner(uint32_t, uint32_t dst, uint64_t) const {
    int i = getColumnOfNode(dst);
    return _h_offset + i;
  }

  bool noCommunication() { return false; }
  bool isVertexCut() const {
    if ((numRowHosts == 1) || (numColumnHosts == 1))
      return false;
    return true;
  }
  void serializePartition(boost::archive::binary_oarchive& ar) {
    ar << numRowHosts;
    ar << numColumnHosts;
  }
  void deserializePartition(boost::archive::binary_iarchive& ar) {
    ar >> numRowHosts;
    ar >> numColumnHosts;
  }

  std::pair<unsigned, unsigned> cartesianGrid() {
    return std::make_pair(numRowHosts, numColumnHosts);
  }
};

////////////////////////////////////////////////////////////////////////////////

// same as above, except columns are flipped (changes behavior of vertex cut
// call as well)
class GenericCVCColumnFlip : public galois::graphs::ReadMasterAssignment {
  unsigned numRowHosts;
  unsigned numColumnHosts;
  unsigned _h_offset;

  void factorizeHosts() {
    numColumnHosts = sqrt(_numHosts);

    while ((_numHosts % numColumnHosts) != 0)
      numColumnHosts--;

    numRowHosts = _numHosts / numColumnHosts;
    assert(numRowHosts >= numColumnHosts);

    // column flip
    std::swap(numRowHosts, numColumnHosts);

    if (_hostID == 0) {
      galois::gPrint("Cartesian grid: ", numRowHosts, " x ", numColumnHosts,
                     "\n");
    }
  }

  //! Returns the grid row ID of this host
  unsigned gridRowID() const { return (_hostID / numColumnHosts); }
  //! Returns the grid row ID of the specified host
  unsigned gridRowID(unsigned id) const { return (id / numColumnHosts); }
  //! Returns the grid column ID of this host
  unsigned gridColumnID() const { return (_hostID % numColumnHosts); }
  //! Returns the grid column ID of the specified host
  unsigned gridColumnID(unsigned id) const { return (id % numColumnHosts); }

  //! Find the column of a particular node
  unsigned getColumnOfNode(uint64_t gid) const {
    return gridColumnID(retrieveMaster(gid));
  }

public:
  GenericCVCColumnFlip(uint32_t hostID, uint32_t numHosts, uint64_t numNodes,
                       uint64_t numEdges)
      : galois::graphs::ReadMasterAssignment(hostID, numHosts, numNodes,
                                             numEdges) {
    factorizeHosts();
    _h_offset = gridRowID() * numColumnHosts;
  }

  uint32_t getEdgeOwner(uint32_t, uint32_t dst, uint64_t) const {
    int i = getColumnOfNode(dst);
    return _h_offset + i;
  }

  bool noCommunication() { return false; }
  bool isVertexCut() const {
    if ((numRowHosts == 1) && (numColumnHosts == 1))
      return false;
    return true;
  }

  void serializePartition(boost::archive::binary_oarchive& ar) {
    ar << numRowHosts;
    ar << numColumnHosts;
  }

  void deserializePartition(boost::archive::binary_iarchive& ar) {
    ar >> numRowHosts;
    ar >> numColumnHosts;
  }

  std::pair<unsigned, unsigned> cartesianGrid() {
    return std::make_pair(numRowHosts, numColumnHosts);
  }
};
////////////////////////////////////////////////////////////////////////////////
class GenericHVC : public galois::graphs::ReadMasterAssignment {
  uint32_t _vCutThreshold;

public:
  GenericHVC(uint32_t hostID, uint32_t numHosts, uint64_t numNodes,
             uint64_t numEdges)
      : galois::graphs::ReadMasterAssignment(hostID, numHosts, numNodes,
                                             numEdges) {
    _vCutThreshold = 1000; // can be changed, but default seems to be 1000
  }

  uint32_t getEdgeOwner(uint32_t src, uint32_t dst, uint64_t numEdges) const {
    if (numEdges > _vCutThreshold) {
      return retrieveMaster(dst);
    } else {
      return retrieveMaster(src);
    }
  }

  bool noCommunication() { return false; }
  // TODO I should be able to make this runtime detectable
  bool isVertexCut() const { return true; }
  void serializePartition(boost::archive::binary_oarchive&) {}
  void deserializePartition(boost::archive::binary_iarchive&) {}
  std::pair<unsigned, unsigned> cartesianGrid() {
    return std::make_pair(0u, 0u);
  }
};

////////////////////////////////////////////////////////////////////////////////

class GingerP : public galois::graphs::CustomMasterAssignment {
  // used in hybrid cut
  uint32_t _vCutThreshold;
  // ginger scoring constants
  double _gamma;
  double _alpha;
  // ginger node/edge ratio
  double _neRatio;

  /**
   * Returns Ginger's composite balance parameter for a given host
   */
  double getCompositeBalanceParam(
      unsigned host, const std::vector<uint64_t>& nodeLoads,
      const std::vector<galois::CopyableAtomic<uint64_t>>& nodeAccum,
      const std::vector<uint64_t>& edgeLoads,
      const std::vector<galois::CopyableAtomic<uint64_t>>& edgeAccum) {
    // get node/edge loads
    uint64_t hostNodeLoad = nodeLoads[host] + nodeAccum[host].load();
    uint64_t hostEdgeLoad = edgeLoads[host] + edgeAccum[host].load();

    return (hostNodeLoad + (_neRatio * hostEdgeLoad)) / 2;
  }

  /**
   * Use FENNEL balance equation to get a score value for partition
   * scoring
   */
  double getFennelBalanceScore(double param) {
    return _alpha * _gamma * pow(param, _gamma - 1);
  }

public:
  GingerP(uint32_t hostID, uint32_t numHosts, uint64_t numNodes,
          uint64_t numEdges)
      : galois::graphs::CustomMasterAssignment(hostID, numHosts, numNodes,
                                               numEdges) {
    _vCutThreshold = 1000;
    _gamma         = 1.5;
    _alpha   = numEdges * pow(numHosts, _gamma - 1.0) / pow(numNodes, _gamma);
    _neRatio = (double)numNodes / (double)numEdges;
  }

  template <typename EdgeTy>
  uint32_t getMaster(uint32_t src,
                     galois::graphs::BufferedGraph<EdgeTy>& bufGraph,
                     const std::vector<uint32_t>& localNodeToMaster,
                     std::unordered_map<uint64_t, uint32_t>& gid2offsets,
                     const std::vector<uint64_t>& nodeLoads,
                     std::vector<galois::CopyableAtomic<uint64_t>>& nodeAccum,
                     const std::vector<uint64_t>& edgeLoads,
                     std::vector<galois::CopyableAtomic<uint64_t>>& edgeAccum) {
    auto ii = bufGraph.edgeBegin(src);
    auto ee = bufGraph.edgeEnd(src);
    // number of edges
    uint64_t ne = std::distance(ii, ee);

    // high in-degree nodes masters stay the same
    if (ne > _vCutThreshold) {
      return _hostID;
    } else {
      // low in degree masters move based on augmented FENNEL scoring metric
      // initialize array to hold scores
      galois::PODResizeableArray<double> scores;
      scores.resize(_numHosts);
      for (unsigned i = 0; i < _numHosts; i++) {
        scores[i] = 0.0;
      }

      for (; ii < ee; ++ii) {
        uint64_t dst         = bufGraph.edgeDestination(*ii);
        size_t offsetIntoMap = (unsigned)-1;

        auto it = gid2offsets.find(dst);
        if (it != gid2offsets.end()) {
          offsetIntoMap = it->second;
        } else {
          // determine offset
          offsetIntoMap = dst - bufGraph.getNodeOffset();
        }

        assert(offsetIntoMap != (unsigned)-1);
        assert(offsetIntoMap >= 0);
        assert(offsetIntoMap < localNodeToMaster.size());

        unsigned currentAssignment = localNodeToMaster[offsetIntoMap];

        if (currentAssignment != (unsigned)-1) {
          scores[currentAssignment] += 1.0;
        } else {
          galois::gDebug("[", _hostID, "] ", dst, " unassigned");
        }
      }

      // subtraction of the composite balance term
      for (unsigned i = 0; i < _numHosts; i++) {
        scores[i] -= getFennelBalanceScore(getCompositeBalanceParam(
            i, nodeLoads, nodeAccum, edgeLoads, edgeAccum));
      }

      unsigned bestHost = -1;
      double bestScore  = std::numeric_limits<double>::lowest();
      // find max score
      for (unsigned i = 0; i < _numHosts; i++) {
        if (scores[i] >= bestScore) {
          // galois::gDebug("best score ", bestScore, " beaten by ", scores[i]);
          bestScore = scores[i];
          bestHost  = i;
        }
      }

      galois::gDebug("[", _hostID, "] ", src, " assigned to ", bestHost,
                     " with num edge ", ne);

      // update metadata; TODO make this a nicer interface
      galois::atomicAdd(nodeAccum[bestHost], (uint64_t)1);
      galois::atomicAdd(edgeAccum[bestHost], ne);

      return bestHost;
    }
  }

  uint32_t getEdgeOwner(uint32_t src, uint32_t dst, uint64_t numEdges) const {
    // if high indegree, then move to source (which is dst), else stay on
    // dst (which is src)
    // note "dst" here is actually the source on the actual graph
    // since we're reading transpose
    if (numEdges > _vCutThreshold) {
      return retrieveMaster(dst);
    } else {
      return retrieveMaster(src);
    }
  }

  bool noCommunication() { return false; }
  // TODO I should be able to make this runtime detectable
  bool isVertexCut() const { return true; }
  void serializePartition(boost::archive::binary_oarchive&) {}
  void deserializePartition(boost::archive::binary_iarchive&) {}
  std::pair<unsigned, unsigned> cartesianGrid() {
    return std::make_pair(0u, 0u);
  }
};

class FennelP : public galois::graphs::CustomMasterAssignment {
  // used in hybrid cut
  uint32_t _vCutThreshold;
  // ginger scoring constants
  double _gamma;
  double _alpha;
  // ginger node/edge ratio
  double _neRatio;

  /**
   * Returns Ginger's composite balance parameter for a given host
   */
  double getCompositeBalanceParam(
      unsigned host, const std::vector<uint64_t>& nodeLoads,
      const std::vector<galois::CopyableAtomic<uint64_t>>& nodeAccum,
      const std::vector<uint64_t>& edgeLoads,
      const std::vector<galois::CopyableAtomic<uint64_t>>& edgeAccum) {
    // get node/edge loads
    uint64_t hostNodeLoad = nodeLoads[host] + nodeAccum[host].load();
    uint64_t hostEdgeLoad = edgeLoads[host] + edgeAccum[host].load();

    return (hostNodeLoad + (_neRatio * hostEdgeLoad)) / 2;
  }

  /**
   * Use FENNEL balance equation to get a score value for partition
   * scoring
   */
  double getFennelBalanceScore(double param) {
    return _alpha * _gamma * pow(param, _gamma - 1);
  }

public:
  FennelP(uint32_t hostID, uint32_t numHosts, uint64_t numNodes,
          uint64_t numEdges)
      : galois::graphs::CustomMasterAssignment(hostID, numHosts, numNodes,
                                               numEdges) {
    _vCutThreshold = 1000;
    _gamma         = 1.5;
    _alpha   = numEdges * pow(numHosts, _gamma - 1.0) / pow(numNodes, _gamma);
    _neRatio = (double)numNodes / (double)numEdges;
  }

  template <typename EdgeTy>
  uint32_t getMaster(uint32_t src,
                     galois::graphs::BufferedGraph<EdgeTy>& bufGraph,
                     const std::vector<uint32_t>& localNodeToMaster,
                     std::unordered_map<uint64_t, uint32_t>& gid2offsets,
                     const std::vector<uint64_t>& nodeLoads,
                     std::vector<galois::CopyableAtomic<uint64_t>>& nodeAccum,
                     const std::vector<uint64_t>& edgeLoads,
                     std::vector<galois::CopyableAtomic<uint64_t>>& edgeAccum) {
    auto ii = bufGraph.edgeBegin(src);
    auto ee = bufGraph.edgeEnd(src);
    // number of edges
    uint64_t ne = std::distance(ii, ee);

    // high degree nodes masters stay the same
    if (ne > _vCutThreshold) {
      return _hostID;
    } else {
      // low degree masters move based on augmented FENNEL scoring metric
      // initialize array to hold scores
      galois::PODResizeableArray<double> scores;
      scores.resize(_numHosts);
      for (unsigned i = 0; i < _numHosts; i++) {
        scores[i] = 0.0;
      }

      for (; ii < ee; ++ii) {
        uint64_t dst         = bufGraph.edgeDestination(*ii);
        size_t offsetIntoMap = (unsigned)-1;

        auto it = gid2offsets.find(dst);
        if (it != gid2offsets.end()) {
          offsetIntoMap = it->second;
        } else {
          // determine offset
          offsetIntoMap = dst - bufGraph.getNodeOffset();
        }

        assert(offsetIntoMap != (unsigned)-1);
        assert(offsetIntoMap >= 0);
        assert(offsetIntoMap < localNodeToMaster.size());

        unsigned currentAssignment = localNodeToMaster[offsetIntoMap];

        if (currentAssignment != (unsigned)-1) {
          scores[currentAssignment] += 1.0;
        } else {
          galois::gDebug("[", _hostID, "] ", dst, " unassigned");
        }
      }

      // subtraction of the composite balance term
      for (unsigned i = 0; i < _numHosts; i++) {
        scores[i] -= getFennelBalanceScore(getCompositeBalanceParam(
            i, nodeLoads, nodeAccum, edgeLoads, edgeAccum));
      }

      unsigned bestHost = -1;
      double bestScore  = std::numeric_limits<double>::lowest();
      // find max score
      for (unsigned i = 0; i < _numHosts; i++) {
        if (scores[i] >= bestScore) {
          // galois::gDebug("best score ", bestScore, " beaten by ", scores[i]);
          bestScore = scores[i];
          bestHost  = i;
        }
      }

      galois::gDebug("[", _hostID, "] ", src, " assigned to ", bestHost,
                     " with num edge ", ne);

      // update metadata; TODO make this a nicer interface
      galois::atomicAdd(nodeAccum[bestHost], (uint64_t)1);
      galois::atomicAdd(edgeAccum[bestHost], ne);

      return bestHost;
    }
  }

  // Fennel is an edge cut: all edges on source
  uint32_t getEdgeOwner(uint32_t src, uint32_t, uint64_t) const {
    return retrieveMaster(src);
  }

  bool noCommunication() { return false; }
  // TODO I should be able to make this runtime detectable
  bool isVertexCut() const { return false; }
  void serializePartition(boost::archive::binary_oarchive&) {}
  void deserializePartition(boost::archive::binary_iarchive&) {}
  std::pair<unsigned, unsigned> cartesianGrid() {
    return std::make_pair(0u, 0u);
  }
};

class SugarP : public galois::graphs::CustomMasterAssignment {
  // used in hybrid cut
  uint32_t _vCutThreshold;
  // ginger scoring constants
  double _gamma;
  double _alpha;
  // ginger node/edge ratio
  double _neRatio;

  unsigned numRowHosts;
  unsigned numColumnHosts;

  void factorizeHosts() {
    numColumnHosts = sqrt(_numHosts);

    while ((_numHosts % numColumnHosts) != 0)
      numColumnHosts--;

    numRowHosts = _numHosts / numColumnHosts;
    assert(numRowHosts >= numColumnHosts);

    if (_hostID == 0) {
      galois::gPrint("Cartesian grid: ", numRowHosts, " x ", numColumnHosts,
                     "\n");
    }
  }

  //! Returns the grid row ID of this host
  unsigned gridRowID() const { return (_hostID / numColumnHosts); }
  //! Returns the grid row ID of the specified host
  unsigned gridRowID(unsigned id) const { return (id / numColumnHosts); }
  //! Returns the grid column ID of this host
  unsigned gridColumnID() const { return (_hostID % numColumnHosts); }
  //! Returns the grid column ID of the specified host
  unsigned gridColumnID(unsigned id) const { return (id % numColumnHosts); }

  //! Find the row of a particular node
  unsigned getRowOfNode(uint64_t gid) const {
    return gridRowID(retrieveMaster(gid));
  }

  //! Find the column of a particular node
  unsigned getColumnOfNode(uint64_t gid) const {
    return gridColumnID(retrieveMaster(gid));
  }

  /**
   * Returns Ginger's composite balance parameter for a given host
   */
  double getCompositeBalanceParam(
      unsigned host, const std::vector<uint64_t>& nodeLoads,
      const std::vector<galois::CopyableAtomic<uint64_t>>& nodeAccum,
      const std::vector<uint64_t>& edgeLoads,
      const std::vector<galois::CopyableAtomic<uint64_t>>& edgeAccum) {
    // get node/edge loads
    uint64_t hostNodeLoad = nodeLoads[host] + nodeAccum[host].load();
    uint64_t hostEdgeLoad = edgeLoads[host] + edgeAccum[host].load();

    return (hostNodeLoad + (_neRatio * hostEdgeLoad)) / 2;
  }

  /**
   * Use FENNEL balance equation to get a score value for partition
   * scoring
   */
  double getFennelBalanceScore(double param) {
    return _alpha * _gamma * pow(param, _gamma - 1);
  }

public:
  SugarP(uint32_t hostID, uint32_t numHosts, uint64_t numNodes,
         uint64_t numEdges)
      : galois::graphs::CustomMasterAssignment(hostID, numHosts, numNodes,
                                               numEdges) {
    _vCutThreshold = 1000;
    _gamma         = 1.5;
    _alpha   = numEdges * pow(numHosts, _gamma - 1.0) / pow(numNodes, _gamma);
    _neRatio = (double)numNodes / (double)numEdges;
    // CVC things
    factorizeHosts();
  }

  template <typename EdgeTy>
  uint32_t getMaster(uint32_t src,
                     galois::graphs::BufferedGraph<EdgeTy>& bufGraph,
                     const std::vector<uint32_t>& localNodeToMaster,
                     std::unordered_map<uint64_t, uint32_t>& gid2offsets,
                     const std::vector<uint64_t>& nodeLoads,
                     std::vector<galois::CopyableAtomic<uint64_t>>& nodeAccum,
                     const std::vector<uint64_t>& edgeLoads,
                     std::vector<galois::CopyableAtomic<uint64_t>>& edgeAccum) {
    auto ii = bufGraph.edgeBegin(src);
    auto ee = bufGraph.edgeEnd(src);
    // number of edges
    uint64_t ne = std::distance(ii, ee);

    // high degree nodes masters stay the same
    if (ne > _vCutThreshold) {
      return _hostID;
    } else {
      // low degree masters move based on augmented FENNEL scoring metric
      // initialize array to hold scores
      galois::PODResizeableArray<double> scores;
      scores.resize(_numHosts);
      for (unsigned i = 0; i < _numHosts; i++) {
        scores[i] = 0.0;
      }

      for (; ii < ee; ++ii) {
        uint64_t dst         = bufGraph.edgeDestination(*ii);
        size_t offsetIntoMap = (unsigned)-1;

        auto it = gid2offsets.find(dst);
        if (it != gid2offsets.end()) {
          offsetIntoMap = it->second;
        } else {
          // determine offset
          offsetIntoMap = dst - bufGraph.getNodeOffset();
        }

        assert(offsetIntoMap != (unsigned)-1);
        assert(offsetIntoMap >= 0);
        assert(offsetIntoMap < localNodeToMaster.size());

        unsigned currentAssignment = localNodeToMaster[offsetIntoMap];

        if (currentAssignment != (unsigned)-1) {
          scores[currentAssignment] += 1.0;
        } else {
          // galois::gDebug("[", _hostID, "] ", dst, " unassigned");
        }
      }

      // subtraction of the composite balance term
      for (unsigned i = 0; i < _numHosts; i++) {
        scores[i] -= getFennelBalanceScore(getCompositeBalanceParam(
            i, nodeLoads, nodeAccum, edgeLoads, edgeAccum));
      }

      unsigned bestHost = -1;
      double bestScore  = std::numeric_limits<double>::lowest();
      // find max score
      for (unsigned i = 0; i < _numHosts; i++) {
        if (scores[i] >= bestScore) {
          // galois::gDebug("best score ", bestScore, " beaten by ", scores[i]);
          bestScore = scores[i];
          bestHost  = i;
        }
      }

      galois::gDebug("[", _hostID, "] ", src, " assigned to ", bestHost,
                     " with num edge ", ne);

      // update metadata; TODO make this a nicer interface
      galois::atomicAdd(nodeAccum[bestHost], (uint64_t)1);
      galois::atomicAdd(edgeAccum[bestHost], ne);

      return bestHost;
    }
  }

  /**
   * return owner of edge using cartesian edge owner determination
   */
  uint32_t getEdgeOwner(uint32_t src, uint32_t dst, uint64_t) const {
    unsigned blockedRowOffset   = getRowOfNode(src) * numColumnHosts;
    unsigned cyclicColumnOffset = getColumnOfNode(dst);
    return blockedRowOffset + cyclicColumnOffset;
  }

  bool noCommunication() { return false; }
  bool isVertexCut() const {
    if ((numRowHosts == 1) || (numColumnHosts == 1))
      return false;
    return true;
  }

  void serializePartition(boost::archive::binary_oarchive& ar) {
    ar << numRowHosts;
    ar << numColumnHosts;
  }

  void deserializePartition(boost::archive::binary_iarchive& ar) {
    ar >> numRowHosts;
    ar >> numColumnHosts;
  }

  std::pair<unsigned, unsigned> cartesianGrid() {
    return std::make_pair(numRowHosts, numColumnHosts);
  }
};

class SugarColumnFlipP : public galois::graphs::CustomMasterAssignment {
  // used in hybrid cut
  uint32_t _vCutThreshold;
  // ginger scoring constants
  double _gamma;
  double _alpha;
  // ginger node/edge ratio
  double _neRatio;

  unsigned numRowHosts;
  unsigned numColumnHosts;

  void factorizeHosts() {
    numColumnHosts = sqrt(_numHosts);

    while ((_numHosts % numColumnHosts) != 0)
      numColumnHosts--;

    numRowHosts = _numHosts / numColumnHosts;
    assert(numRowHosts >= numColumnHosts);

    // column flip
    std::swap(numRowHosts, numColumnHosts);

    if (_hostID == 0) {
      galois::gPrint("Cartesian grid: ", numRowHosts, " x ", numColumnHosts,
                     "\n");
    }
  }

  //! Returns the grid row ID of this host
  unsigned gridRowID() const { return (_hostID / numColumnHosts); }
  //! Returns the grid row ID of the specified host
  unsigned gridRowID(unsigned id) const { return (id / numColumnHosts); }
  //! Returns the grid column ID of this host
  unsigned gridColumnID() const { return (_hostID % numColumnHosts); }
  //! Returns the grid column ID of the specified host
  unsigned gridColumnID(unsigned id) const { return (id % numColumnHosts); }

  //! Find the row of a particular node
  unsigned getRowOfNode(uint64_t gid) const {
    return gridRowID(retrieveMaster(gid));
  }

  //! Find the column of a particular node
  unsigned getColumnOfNode(uint64_t gid) const {
    return gridColumnID(retrieveMaster(gid));
  }

  /**
   * Returns Ginger's composite balance parameter for a given host
   */
  double getCompositeBalanceParam(
      unsigned host, const std::vector<uint64_t>& nodeLoads,
      const std::vector<galois::CopyableAtomic<uint64_t>>& nodeAccum,
      const std::vector<uint64_t>& edgeLoads,
      const std::vector<galois::CopyableAtomic<uint64_t>>& edgeAccum) {
    // get node/edge loads
    uint64_t hostNodeLoad = nodeLoads[host] + nodeAccum[host].load();
    uint64_t hostEdgeLoad = edgeLoads[host] + edgeAccum[host].load();

    return (hostNodeLoad + (_neRatio * hostEdgeLoad)) / 2;
  }

  /**
   * Use FENNEL balance equation to get a score value for partition
   * scoring
   */
  double getFennelBalanceScore(double param) {
    return _alpha * _gamma * pow(param, _gamma - 1);
  }

public:
  SugarColumnFlipP(uint32_t hostID, uint32_t numHosts, uint64_t numNodes,
                   uint64_t numEdges)
      : galois::graphs::CustomMasterAssignment(hostID, numHosts, numNodes,
                                               numEdges) {
    _vCutThreshold = 1000;
    _gamma         = 1.5;
    _alpha   = numEdges * pow(numHosts, _gamma - 1.0) / pow(numNodes, _gamma);
    _neRatio = (double)numNodes / (double)numEdges;
    // CVC things
    factorizeHosts();
  }

  template <typename EdgeTy>
  uint32_t getMaster(uint32_t src,
                     galois::graphs::BufferedGraph<EdgeTy>& bufGraph,
                     const std::vector<uint32_t>& localNodeToMaster,
                     std::unordered_map<uint64_t, uint32_t>& gid2offsets,
                     const std::vector<uint64_t>& nodeLoads,
                     std::vector<galois::CopyableAtomic<uint64_t>>& nodeAccum,
                     const std::vector<uint64_t>& edgeLoads,
                     std::vector<galois::CopyableAtomic<uint64_t>>& edgeAccum) {
    auto ii = bufGraph.edgeBegin(src);
    auto ee = bufGraph.edgeEnd(src);
    // number of edges
    uint64_t ne = std::distance(ii, ee);

    // high degree nodes masters stay the same
    if (ne > _vCutThreshold) {
      return _hostID;
    } else {
      // low degree masters move based on augmented FENNEL scoring metric
      // initialize array to hold scores
      galois::PODResizeableArray<double> scores;
      scores.resize(_numHosts);
      for (unsigned i = 0; i < _numHosts; i++) {
        scores[i] = 0.0;
      }

      for (; ii < ee; ++ii) {
        uint64_t dst         = bufGraph.edgeDestination(*ii);
        size_t offsetIntoMap = (unsigned)-1;

        auto it = gid2offsets.find(dst);
        if (it != gid2offsets.end()) {
          offsetIntoMap = it->second;
        } else {
          // determine offset
          offsetIntoMap = dst - bufGraph.getNodeOffset();
        }

        assert(offsetIntoMap != (unsigned)-1);
        assert(offsetIntoMap >= 0);
        assert(offsetIntoMap < localNodeToMaster.size());

        unsigned currentAssignment = localNodeToMaster[offsetIntoMap];

        if (currentAssignment != (unsigned)-1) {
          scores[currentAssignment] += 1.0;
        } else {
          galois::gDebug("[", _hostID, "] ", dst, " unassigned");
        }
      }

      // subtraction of the composite balance term
      for (unsigned i = 0; i < _numHosts; i++) {
        scores[i] -= getFennelBalanceScore(getCompositeBalanceParam(
            i, nodeLoads, nodeAccum, edgeLoads, edgeAccum));
      }

      unsigned bestHost = -1;
      double bestScore  = std::numeric_limits<double>::lowest();
      // find max score
      for (unsigned i = 0; i < _numHosts; i++) {
        if (scores[i] >= bestScore) {
          // galois::gDebug("best score ", bestScore, " beaten by ", scores[i]);
          bestScore = scores[i];
          bestHost  = i;
        }
      }

      galois::gDebug("[", _hostID, "] ", src, " assigned to ", bestHost,
                     " with num edge ", ne);

      // update metadata; TODO make this a nicer interface
      galois::atomicAdd(nodeAccum[bestHost], (uint64_t)1);
      galois::atomicAdd(edgeAccum[bestHost], ne);

      return bestHost;
    }
  }

  /**
   * return owner of edge using cartesian edge owner determination
   */
  uint32_t getEdgeOwner(uint32_t src, uint32_t dst, uint64_t) const {
    unsigned blockedRowOffset   = getRowOfNode(src) * numColumnHosts;
    unsigned cyclicColumnOffset = getColumnOfNode(dst);
    return blockedRowOffset + cyclicColumnOffset;
  }

  bool noCommunication() { return false; }
  bool isVertexCut() const {
    if ((numRowHosts == 1) && (numColumnHosts == 1))
      return false;
    return true;
  }
  void serializePartition(boost::archive::binary_oarchive& ar) {
    ar << numRowHosts;
    ar << numColumnHosts;
  }
  void deserializePartition(boost::archive::binary_iarchive& ar) {
    ar >> numRowHosts;
    ar >> numColumnHosts;
  }

  std::pair<unsigned, unsigned> cartesianGrid() {
    return std::make_pair(numRowHosts, numColumnHosts);
  }
};

#endif
