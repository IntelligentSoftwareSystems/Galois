#ifndef _GALOIS_DIST_GENERICPARTS_H
#define _GALOIS_DIST_GENERICPARTS_H

#include "DistributedGraph.h"
#include <utility>

class GenericCVC {
  std::vector<std::pair<uint64_t, uint64_t>> _gid2host;
  uint32_t _hostID;
  uint32_t _numHosts;
  unsigned numRowHosts;
  unsigned numColumnHosts;
  unsigned _h_offset;

  void factorizeHosts() {
    numColumnHosts  = sqrt(_numHosts);

    while ((_numHosts % numColumnHosts) != 0)
      numColumnHosts--;

    numRowHosts = _numHosts / numColumnHosts;
    assert(numRowHosts >= numColumnHosts);

    //if (moreColumnHosts) {
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
  unsigned gridColumnID() const {
    return (_hostID % numColumnHosts);
  }
  //! Returns the grid column ID of the specified host
  unsigned gridColumnID(unsigned id) const {
    return (id % numColumnHosts);
  }

  //! Find the column of a particular node
  unsigned getColumnOfNode(uint64_t gid) const {
    return gridColumnID(getMaster(gid));
  }

 public:
  GenericCVC(uint32_t hostID, uint32_t numHosts) {
    _hostID = hostID;
    _numHosts = numHosts;
    factorizeHosts();
    _h_offset = gridRowID() * numColumnHosts;
  }

  void saveGIDToHost(std::vector<std::pair<uint64_t, uint64_t>>& gid2host) {
    _gid2host = gid2host;
  }

  uint32_t getMaster(uint32_t gid) const {
    for (auto h = 0U; h < _numHosts; ++h) {
      uint64_t start, end;
      std::tie(start, end) = _gid2host[h];
      if (gid >= start && gid < end) {
        return h;
      }
    }
    assert(false);
    return _numHosts;
  }

  uint32_t getEdgeOwner(uint32_t src, uint32_t dst, uint64_t numEdges) const {
    int i         = getColumnOfNode(dst);
    return _h_offset + i;
  }

  bool isVertexCut() const {
    if ((numRowHosts == 1) || (numColumnHosts == 1)) return false;
    return true;
  }

  constexpr static bool isCartCut() {
    return true;
  }

  bool isNotCommunicationPartner(unsigned host, unsigned syncType,
                                 WriteLocation writeLocation,
                                 ReadLocation readLocation,
                                 bool transposed) {
    if (transposed) {
      if (syncType == 0) {
        switch (writeLocation) {
        case writeSource:
          return (gridColumnID() != gridColumnID(host));
        case writeDestination:
          return (gridRowID() != gridRowID(host));
        case writeAny:
          assert((gridRowID() == gridRowID(host)) ||
                 (gridColumnID() == gridColumnID(host)));
          return ((gridRowID() != gridRowID(host)) &&
                  (gridColumnID() != gridColumnID(host))); // false
        default:
          assert(false);
        }
      } else { // syncBroadcast
        switch (readLocation) {
        case readSource:
          return (gridColumnID() != gridColumnID(host));
        case readDestination:
          return (gridRowID() != gridRowID(host));
        case readAny:
          assert((gridRowID() == gridRowID(host)) ||
                 (gridColumnID() == gridColumnID(host)));
          return ((gridRowID() != gridRowID(host)) &&
                  (gridColumnID() != gridColumnID(host))); // false
        default:
          assert(false);
        }
      }
    } else {
      if (syncType == 0) {
        switch (writeLocation) {
        case writeSource:
          return (gridRowID() != gridRowID(host));
        case writeDestination:
          return (gridColumnID() != gridColumnID(host));
        case writeAny:
          assert((gridRowID() == gridRowID(host)) ||
                 (gridColumnID() == gridColumnID(host)));
          return ((gridRowID() != gridRowID(host)) &&
                  (gridColumnID() != gridColumnID(host))); // false
        default:
          assert(false);
        }
      } else { // syncBroadcast, 1
        switch (readLocation) {
        case readSource:
          return (gridRowID() != gridRowID(host));
        case readDestination:
          return (gridColumnID() != gridColumnID(host));
        case readAny:
          assert((gridRowID() == gridRowID(host)) ||
                 (gridColumnID() == gridColumnID(host)));
          return ((gridRowID() != gridRowID(host)) &&
                  (gridColumnID() != gridColumnID(host))); // false
        default:
          assert(false);
        }
      }
      return false;
    }
    return false;
  }

  void serializePartition(boost::archive::binary_oarchive& ar) {
    ar << numRowHosts;
    ar << numColumnHosts;
  }

  void deserializePartition(boost::archive::binary_iarchive& ar) {
    ar >> numRowHosts;
    ar >> numColumnHosts;
  }

};

// same as above, except columns are flipped (changes behavior of vertex cut
// call as well)
class GenericCVCColumnFlip {
  std::vector<std::pair<uint64_t, uint64_t>> _gid2host;
  uint32_t _hostID;
  uint32_t _numHosts;
  unsigned numRowHosts;
  unsigned numColumnHosts;
  unsigned _h_offset;

  void factorizeHosts() {
    numColumnHosts  = sqrt(_numHosts);

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
  unsigned gridColumnID() const {
    return (_hostID % numColumnHosts);
  }
  //! Returns the grid column ID of the specified host
  unsigned gridColumnID(unsigned id) const {
    return (id % numColumnHosts);
  }

  //! Find the column of a particular node
  unsigned getColumnOfNode(uint64_t gid) const {
    return gridColumnID(getMaster(gid));
  }

 public:
  GenericCVCColumnFlip(uint32_t hostID, uint32_t numHosts) {
    _hostID = hostID;
    _numHosts = numHosts;
    factorizeHosts();
    _h_offset = gridRowID() * numColumnHosts;
  }

  void saveGIDToHost(std::vector<std::pair<uint64_t, uint64_t>>& gid2host) {
    _gid2host = gid2host;
  }

  uint32_t getMaster(uint32_t gid) const {
    for (auto h = 0U; h < _numHosts; ++h) {
      uint64_t start, end;
      std::tie(start, end) = _gid2host[h];
      if (gid >= start && gid < end) {
        return h;
      }
    }
    assert(false);
    return _numHosts;
  }

  uint32_t getEdgeOwner(uint32_t src, uint32_t dst, uint64_t numEdges) const {
    int i         = getColumnOfNode(dst);
    return _h_offset + i;
  }

  bool isVertexCut() const {
    if ((numRowHosts == 1) && (numColumnHosts == 1)) return false;
    return true;
  }

  constexpr static bool isCartCut() {
    return true;
  }

  bool isNotCommunicationPartner(unsigned host, unsigned syncType,
                                 WriteLocation writeLocation,
                                 ReadLocation readLocation,
                                 bool transposed) {
    if (transposed) {
      if (syncType == 0) {
        switch (writeLocation) {
        case writeSource:
          return (gridColumnID() != gridColumnID(host));
        case writeDestination:
          return (gridRowID() != gridRowID(host));
        case writeAny:
          assert((gridRowID() == gridRowID(host)) ||
                 (gridColumnID() == gridColumnID(host)));
          return ((gridRowID() != gridRowID(host)) &&
                  (gridColumnID() != gridColumnID(host))); // false
        default:
          assert(false);
        }
      } else { // syncBroadcast
        switch (readLocation) {
        case readSource:
          return (gridColumnID() != gridColumnID(host));
        case readDestination:
          return (gridRowID() != gridRowID(host));
        case readAny:
          assert((gridRowID() == gridRowID(host)) ||
                 (gridColumnID() == gridColumnID(host)));
          return ((gridRowID() != gridRowID(host)) &&
                  (gridColumnID() != gridColumnID(host))); // false
        default:
          assert(false);
        }
      }
    } else {
      if (syncType == 0) {
        switch (writeLocation) {
        case writeSource:
          return (gridRowID() != gridRowID(host));
        case writeDestination:
          return (gridColumnID() != gridColumnID(host));
        case writeAny:
          assert((gridRowID() == gridRowID(host)) ||
                 (gridColumnID() == gridColumnID(host)));
          return ((gridRowID() != gridRowID(host)) &&
                  (gridColumnID() != gridColumnID(host))); // false
        default:
          assert(false);
        }
      } else { // syncBroadcast, 1
        switch (readLocation) {
        case readSource:
          return (gridRowID() != gridRowID(host));
        case readDestination:
          return (gridColumnID() != gridColumnID(host));
        case readAny:
          assert((gridRowID() == gridRowID(host)) ||
                 (gridColumnID() == gridColumnID(host)));
          return ((gridRowID() != gridRowID(host)) &&
                  (gridColumnID() != gridColumnID(host))); // false
        default:
          assert(false);
        }
      }
      return false;
    }
    return false;
  }

  void serializePartition(boost::archive::binary_oarchive& ar) {
    ar << numRowHosts;
    ar << numColumnHosts;
  }

  void deserializePartition(boost::archive::binary_iarchive& ar) {
    ar >> numRowHosts;
    ar >> numColumnHosts;
  }
};

class GenericHVC {
  std::vector<std::pair<uint64_t, uint64_t>> _gid2host;
  uint32_t _hostID;
  uint32_t _numHosts;
  uint32_t _vCutThreshold;
 public:
  GenericHVC(uint32_t hostID, uint32_t numHosts) {
    _hostID = hostID;
    _numHosts = numHosts;
    _vCutThreshold = 1000; // can be changed, but default seems to be 1000
  }

  void saveGIDToHost(std::vector<std::pair<uint64_t, uint64_t>>& gid2host) {
    _gid2host = gid2host;
  }

  uint32_t getMaster(uint32_t gid) const {
    for (auto h = 0U; h < _numHosts; ++h) {
      uint64_t start, end;
      std::tie(start, end) = _gid2host[h];
      if (gid >= start && gid < end) {
        return h;
      }
    }
    assert(false);
    return _numHosts;
  }

  uint32_t getEdgeOwner(uint32_t src, uint32_t dst, uint64_t numEdges) const {
    if (numEdges > _vCutThreshold) {
      return getMaster(dst);
    } else {
      return getMaster(src);
    }
  }

  // TODO I should be able to make this runtime detectable
  bool isVertexCut() const {
    return true;
  }

  constexpr static bool isCartCut() {
    return false;
  }

  // not used by this
  bool isNotCommunicationPartner(unsigned host, unsigned syncType,
                                 WriteLocation writeLocation,
                                 ReadLocation readLocation,
                                 bool transposed) {
    return false;
  }

  void serializePartition(boost::archive::binary_oarchive& ar) {
    return;
  }

  void deserializePartition(boost::archive::binary_iarchive& ar) {
    return;
  }

};
#endif
