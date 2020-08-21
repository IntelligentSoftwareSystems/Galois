/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#ifndef _GALOIS_DIST_OFFLINE_GRAPH_
#define _GALOIS_DIST_OFFLINE_GRAPH_

#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>

#include <fcntl.h>
#include <sys/mman.h>

#include <boost/iterator/counting_iterator.hpp>

#include "galois/config.h"
#include "galois/graphs/Details.h"
#include "galois/graphs/GraphHelpers.h"
#include "galois/substrate/SimpleLock.h"

namespace galois {
namespace graphs {

// File format V1:
// version (1) {uint64_t LE}
// EdgeType size {uint64_t LE}
// numNodes {uint64_t LE}
// numEdges {uint64_t LE}
// outindexs[numNodes] {uint64_t LE} (outindex[nodeid] is index of first edge
// for nodeid + 1 (end interator.  node 0 has an implicit start iterator of 0.
// outedges[numEdges] {uint32_t LE}
// potential padding (32bit max) to Re-Align to 64bits
// EdgeType[numEdges] {EdgeType size}

// File format V2:
// version (2) {uint64_t LE}
// EdgeType size {uint64_t LE}
// numNodes {uint64_t LE}
// numEdges {uint64_t LE}
// outindexs[numNodes] {uint64_t LE} (outindex[nodeid] is index of first edge
// for nodeid + 1 (end interator.  node 0 has an implicit start iterator of 0.
// outedges[numEdges] {uint64_t LE}
// EdgeType[numEdges] {EdgeType size}

class OfflineGraph {
  std::ifstream fileEdgeDst, fileIndex, fileEdgeData;
  std::streamoff locEdgeDst, locIndex, locEdgeData;

  uint64_t numNodes;
  uint64_t numEdges;
  uint64_t sizeEdgeData;
  size_t length;
  bool v2;
  uint64_t numSeeksEdgeDst, numSeeksIndex, numSeeksEdgeData;
  uint64_t numBytesReadEdgeDst, numBytesReadIndex, numBytesReadEdgeData;

  galois::substrate::SimpleLock lock;

  uint64_t outIndexs(uint64_t node) {
    std::lock_guard<decltype(lock)> lg(lock);
    std::streamoff pos = (4 + node) * sizeof(uint64_t);

    // move to correct position in file
    if (locEdgeDst != pos) {
      numSeeksEdgeDst++;
      fileEdgeDst.seekg(pos, fileEdgeDst.beg);
      locEdgeDst = pos;
    }

    // read the value
    uint64_t retval;
    try {
      fileEdgeDst.read(reinterpret_cast<char*>(&retval), sizeof(uint64_t));
    } catch (const std::ifstream::failure& e) {
      std::cerr << "Exception while reading edge destinations:" << e.what()
                << "\n";
      std::cerr << "IO error flags: EOF " << fileEdgeDst.eof() << " FAIL "
                << fileEdgeDst.fail() << " BAD " << fileEdgeDst.bad() << "\n";
    }

    // metadata update
    auto numBytesRead = fileEdgeDst.gcount();
    assert(numBytesRead == sizeof(uint64_t));
    locEdgeDst += numBytesRead;
    numBytesReadEdgeDst += numBytesRead;

    return retval;
  }

  uint64_t outEdges(uint64_t edge) {
    std::lock_guard<decltype(lock)> lg(lock);
    std::streamoff pos = (4 + numNodes) * sizeof(uint64_t) +
                         edge * (v2 ? sizeof(uint64_t) : sizeof(uint32_t));

    // move to correct position
    if (locIndex != pos) {
      numSeeksIndex++;
      fileIndex.seekg(pos, fileEdgeDst.beg);
      locIndex = pos;
    }

    // v2 reads 64 bits, v1 reads 32 bits
    if (v2) {
      uint64_t retval;
      try {
        fileIndex.read(reinterpret_cast<char*>(&retval), sizeof(uint64_t));
      } catch (const std::ifstream::failure& e) {
        std::cerr << "Exception while reading index:" << e.what() << "\n";
        std::cerr << "IO error flags: EOF " << fileIndex.eof() << " FAIL "
                  << fileIndex.fail() << " BAD " << fileIndex.bad() << "\n";
      }

      auto numBytesRead = fileIndex.gcount();
      assert(numBytesRead == sizeof(uint64_t));
      locIndex += numBytesRead;
      numBytesReadIndex += numBytesRead;
      return retval;
    } else {
      uint32_t retval;
      try {
        fileIndex.read(reinterpret_cast<char*>(&retval), sizeof(uint32_t));
      } catch (const std::ifstream::failure& e) {
        std::cerr << "Exception while reading index:" << e.what() << "\n";
        std::cerr << "IO error flags: EOF " << fileIndex.eof() << " FAIL "
                  << fileIndex.fail() << " BAD " << fileIndex.bad() << "\n";
      }

      auto numBytesRead = fileIndex.gcount();
      assert(numBytesRead == sizeof(uint32_t));
      locIndex += numBytesRead;
      numBytesReadIndex += numBytesRead;
      return retval;
    }
  }

  template <typename T>
  T edgeData(uint64_t edge) {
    assert(sizeof(T) <= sizeEdgeData);
    std::lock_guard<decltype(lock)> lg(lock);
    std::streamoff pos = (4 + numNodes) * sizeof(uint64_t) +
                         numEdges * (v2 ? sizeof(uint64_t) : sizeof(uint32_t));

    // align + move to correct position
    pos = (pos + 7) & ~7;
    pos += edge * sizeEdgeData;

    if (locEdgeData != pos) {
      numSeeksEdgeData++;
      fileEdgeData.seekg(pos, fileEdgeDst.beg);
      locEdgeData = pos;
    }

    T retval;
    try {
      fileEdgeData.read(reinterpret_cast<char*>(&retval), sizeof(T));
    } catch (const std::ifstream::failure& e) {
      std::cerr << "Exception while reading edge data:" << e.what() << "\n";
      std::cerr << "IO error flags: EOF " << fileEdgeData.eof() << " FAIL "
                << fileEdgeData.fail() << " BAD " << fileEdgeData.bad() << "\n";
    }

    auto numBytesRead = fileEdgeData.gcount();
    assert(numBytesRead == sizeof(T));
    locEdgeData += numBytesRead;
    numBytesReadEdgeData += numBytesRead;
    /*fprintf(stderr, "READ:: %ld[", edge);
    for(int i=0; i<sizeof(T); ++i){
       fprintf(stderr, "%c", reinterpret_cast<char*>(&retval)[i]);
    }
    fprintf(stderr, "]");*/
    return retval;
  }

protected:
  void setSize(size_t val) { numNodes = val; }
  void setSizeEdges(size_t val) { numEdges = val; }

public:
  typedef boost::counting_iterator<uint64_t> iterator;
  typedef boost::counting_iterator<uint64_t> edge_iterator;
  typedef uint64_t GraphNode;

  OfflineGraph() {}

  OfflineGraph(const std::string& name)
      : fileEdgeDst(name, std::ios_base::binary),
        fileIndex(name, std::ios_base::binary),
        fileEdgeData(name, std::ios_base::binary), locEdgeDst(0), locIndex(0),
        locEdgeData(0), numSeeksEdgeDst(0), numSeeksIndex(0),
        numSeeksEdgeData(0), numBytesReadEdgeDst(0), numBytesReadIndex(0),
        numBytesReadEdgeData(0) {
    if (!fileEdgeDst.is_open() || !fileEdgeDst.good())
      throw "Bad filename";
    if (!fileIndex.is_open() || !fileIndex.good())
      throw "Bad filename";
    if (!fileEdgeData.is_open() || !fileEdgeData.good())
      throw "Bad filename";

    fileEdgeDst.exceptions(std::ifstream::eofbit | std::ifstream::failbit |
                           std::ifstream::badbit);
    fileIndex.exceptions(std::ifstream::eofbit | std::ifstream::failbit |
                         std::ifstream::badbit);
    fileEdgeData.exceptions(std::ifstream::eofbit | std::ifstream::failbit |
                            std::ifstream::badbit);

    uint64_t ver = 0;

    try {
      fileEdgeDst.read(reinterpret_cast<char*>(&ver), sizeof(uint64_t));
      fileEdgeDst.read(reinterpret_cast<char*>(&sizeEdgeData),
                       sizeof(uint64_t));
      fileEdgeDst.read(reinterpret_cast<char*>(&numNodes), sizeof(uint64_t));
      fileEdgeDst.read(reinterpret_cast<char*>(&numEdges), sizeof(uint64_t));
    } catch (const std::ifstream::failure& e) {
      std::cerr << "Exception while reading graph header:" << e.what() << "\n";
      std::cerr << "IO error flags: EOF " << fileEdgeDst.eof() << " FAIL "
                << fileEdgeDst.fail() << " BAD " << fileEdgeDst.bad() << "\n";
    }

    if (ver == 0 || ver > 2)
      throw "Bad Version";

    v2 = ver == 2;

    if (!fileEdgeDst)
      throw "Out of data";

    // File length
    fileEdgeDst.seekg(0, fileEdgeDst.end);
    length = fileEdgeDst.tellg();
    if (length < sizeof(uint64_t) * (4 + numNodes) +
                     (v2 ? sizeof(uint64_t) : sizeof(uint32_t)) * numEdges)
      throw "File too small";

    fileEdgeDst.seekg(0, std::ios_base::beg);
    fileEdgeData.seekg(0, std::ios_base::beg);
    fileIndex.seekg(0, std::ios_base::beg);
  }

  uint64_t num_seeks() {
    // std::cout << "Seeks :: " << numSeeksEdgeDst << " , " << numSeeksEdgeData
    //          << " , " << numSeeksIndex << " \n";
    return numSeeksEdgeDst + numSeeksEdgeData + numSeeksIndex;
  }

  uint64_t num_bytes_read() {
    // std::cout << "Bytes read :: " << numBytesReadEdgeDst << " , " <<
    // numBytesReadEdgeData << " , " << numBytesReadIndex << " \n";
    return numBytesReadEdgeDst + numBytesReadEdgeData + numBytesReadIndex;
  }

  void reset_seek_counters() {
    numSeeksEdgeDst = numSeeksEdgeData = numSeeksIndex = 0;
    numBytesReadEdgeDst = numBytesReadEdgeData = numBytesReadIndex = 0;
  }

  OfflineGraph(OfflineGraph&&) = default;

  size_t size() const { return numNodes; }
  size_t sizeEdges() const { return numEdges; }
  size_t edgeSize() const { return sizeEdgeData; }

  iterator begin() { return iterator(0); }
  iterator end() { return iterator(numNodes); }

  edge_iterator edge_begin(GraphNode N) {
    if (N == 0)
      return edge_iterator(0);
    else
      return edge_iterator(outIndexs(N - 1));
  }

  edge_iterator edge_end(GraphNode N) { return edge_iterator(outIndexs(N)); }

  GraphNode getEdgeDst(edge_iterator ni) { return outEdges(*ni); }

  runtime::iterable<NoDerefIterator<edge_iterator>> edges(GraphNode N) {
    return internal::make_no_deref_range(edge_begin(N), edge_end(N));
  }

  template <typename T>
  T getEdgeData(edge_iterator ni) {
    return edgeData<T>(*ni);
  }

  /**
   * Accesses the prefix sum on disk.
   *
   * @param n Index into edge prefix sum
   * @returns The value located at index n in the edge prefix sum array
   */
  uint64_t operator[](uint64_t n) { return outIndexs(n); }

  // typedefs used by divide by node below
  typedef std::pair<iterator, iterator> NodeRange;
  typedef std::pair<edge_iterator, edge_iterator> EdgeRange;
  typedef std::pair<NodeRange, EdgeRange> GraphRange;

  /**
   * Returns 2 ranges (one for nodes, one for edges) for a particular division.
   * The ranges specify the nodes/edges that a division is responsible for. The
   * function attempts to split them evenly among threads given some kind of
   * weighting
   *
   * @param nodeWeight weight to give to a node in division
   * @param edgeWeight weight to give to an edge in division
   * @param id Division number you want the ranges for
   * @param total Total number of divisions
   * @param scaleFactor Vector specifying if certain divisions should get more
   * than other divisions
   */
  virtual auto
  divideByNode(size_t nodeWeight, size_t edgeWeight, size_t id, size_t total,
               std::vector<unsigned> scaleFactor = std::vector<unsigned>())
      -> GraphRange {
    return galois::graphs::divideNodesBinarySearch<OfflineGraph>(
        numNodes, numEdges, nodeWeight, edgeWeight, id, total, *this,
        scaleFactor);
  }
};

class OfflineGraphWriter {
  std::fstream file;
  uint64_t numNodes, numEdges;
  bool smallData;
  uint64_t ver;
  std::vector<uint64_t> bufferDst;

  std::deque<uint64_t> edgeOffsets;

  std::streamoff offsetOfDst(uint64_t edge) {
    return sizeof(uint64_t) * (4 + numNodes + edge);
  }
  std::streamoff offsetOfData(uint64_t edge) {
    return sizeof(uint64_t) * (4 + numNodes + numEdges) +
           (smallData ? sizeof(float) : sizeof(double)) * edge;
  }

  void setEdge32(uint64_t src, uint64_t offset, uint64_t dst, uint32_t val) {
    if (src)
      offset += edgeOffsets[src - 1];
    file.seekg(offsetOfDst(offset), std::ios_base::beg);
    file.write(reinterpret_cast<char*>(&dst), sizeof(uint64_t));
    file.seekg(offsetOfData(offset), std::ios_base::beg);
    file.write(reinterpret_cast<char*>(&val), sizeof(uint32_t));
  }

  void setEdge64(uint64_t src, uint64_t offset, uint64_t dst, uint64_t val) {
    if (src)
      offset += edgeOffsets[src - 1];
    file.seekg(offsetOfDst(offset), std::ios_base::beg);
    file.write(reinterpret_cast<char*>(&dst), sizeof(uint64_t));
    file.seekg(offsetOfData(offset), std::ios_base::beg);
    file.write(reinterpret_cast<char*>(&val), sizeof(uint64_t));
  }

  void setEdge_sorted(uint64_t dst) {
    if (ver == 1) {
      uint32_t dst32 = dst;
      file.write(reinterpret_cast<char*>(&dst32), sizeof(uint32_t));
    } else {
      file.write(reinterpret_cast<char*>(&dst), sizeof(uint64_t));
    }
  }

  void setEdge_sortedBuffer() {
    if (ver == 1) {
      std::vector<uint32_t> tmp(bufferDst.begin(), bufferDst.end());
      file.write(reinterpret_cast<char*>(&tmp[0]),
                 (sizeof(uint32_t) * tmp.size()));
    }
    file.write(reinterpret_cast<char*>(&bufferDst[0]),
               (sizeof(uint64_t) * bufferDst.size()));
  }

  // void setEdge64_sorted(uint64_t dst) {
  // file.write(reinterpret_cast<char*>(&dst), sizeof(uint32_t));
  //}

public:
  OfflineGraphWriter(const std::string& name, bool use32 = false,
                     uint64_t _numNodes = 0, uint64_t _numEdges = 0)
      : file(name, std::ios_base::in | std::ios_base::out |
                       std::ios_base::binary | std::ios_base::trunc),
        numNodes(_numNodes), numEdges(_numEdges), smallData(use32), ver(1) {
    if (!file.is_open() || !file.good())
      throw "Bad filename";
    uint64_t etSize = smallData ? sizeof(float) : sizeof(double);
    file.write(reinterpret_cast<char*>(&ver), sizeof(uint64_t));
    file.write(reinterpret_cast<char*>(&etSize), sizeof(uint64_t));
    file.write(reinterpret_cast<char*>(&numNodes), sizeof(uint64_t));
    file.write(reinterpret_cast<char*>(&numEdges), sizeof(uint64_t));
    file.seekg(0, std::ios_base::beg);
  }

  ~OfflineGraphWriter() {}

  // sets the number of nodes and edges.  points to an container of edge counts
  void setCounts(std::deque<uint64_t> edgeCounts) {
    edgeOffsets = std::move(edgeCounts);
    numNodes    = edgeOffsets.size();
    numEdges    = std::accumulate(edgeOffsets.begin(), edgeOffsets.end(), 0);
    std::cout << " NUM EDGES  : " << numEdges << "\n";
    std::partial_sum(edgeOffsets.begin(), edgeOffsets.end(),
                     edgeOffsets.begin());
    // Nodes are greater than 2^32 so need ver = 2.
    if (numNodes >= 4294967296) {
      ver = 2;
    } else {
      ver = 1;
    }
    std::cout << " USING VERSION : " << ver << "\n";
    uint64_t etSize = 0; // smallData ? sizeof(float) : sizeof(double);
    file.seekg(0, std::ios_base::beg);
    file.write(reinterpret_cast<char*>(&ver), sizeof(uint64_t));
    file.write(reinterpret_cast<char*>(&etSize), sizeof(uint64_t));
    // file.seekg(sizeof(uint64_t)*2, std::ios_base::beg);
    file.write(reinterpret_cast<char*>(&numNodes), sizeof(uint64_t));
    file.write(reinterpret_cast<char*>(&numEdges), sizeof(uint64_t));
    for (auto i : edgeOffsets)
      file.write(reinterpret_cast<char*>(&i), sizeof(uint64_t));
    file.seekg(0, std::ios_base::beg);
  }

  void setEdge(uint64_t src, uint64_t offset, uint64_t dst, uint64_t val) {
    if (smallData)
      setEdge32(src, offset, dst, val);
    else
      setEdge64(src, offset, dst, val);
  }

  void setEdgeSorted(uint64_t dst) { setEdge_sorted(dst); }

  void seekEdgesDstStart() { file.seekg(offsetOfDst(0), std::ios_base::beg); }
};

} // namespace graphs
} // namespace galois

#endif //_GALOIS_DIST_OFFLINE_GRAPH_
