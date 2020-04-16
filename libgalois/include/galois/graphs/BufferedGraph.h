/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

/**
 * @file BufferedGraph.h
 *
 * Contains the implementation of BufferedGraph
 */

#ifndef GALOIS_GRAPH_BUFGRAPH_H
#define GALOIS_GRAPH_BUFGRAPH_H

#include <fstream>

#include <boost/iterator/counting_iterator.hpp>

#include "galois/config.h"
#include "galois/gIO.h"
#include "galois/Reduction.h"

namespace galois {
namespace graphs {

/**
 * Class that loads a portion of a Galois graph from disk directly into
 * memory buffers for access.
 *
 * @tparam EdgeDataType type of the edge data
 * @todo version 2 Galois binary graph support; currently only suppports
 * version 1
 */
template <typename EdgeDataType>
class BufferedGraph {
private:
  // buffers that you load data into
  //! buffer that tells you where a particular node's edges begin
  uint64_t* outIndexBuffer = nullptr;
  //! buffer that tells the destination of edges
  uint32_t* edgeDestBuffer = nullptr;
  //! buffer that stores edge data
  EdgeDataType* edgeDataBuffer = nullptr;

  //! size of the entire graph (not just locallly loaded portion)
  uint32_t globalSize = 0;
  //! number of edges in the entire graph (not just locallly loaded portion)
  uint64_t globalEdgeSize = 0;

  //! number of nodes loaded into this graph
  uint32_t numLocalNodes = 0;
  //! number of edges loaded into this graph
  uint64_t numLocalEdges = 0;

  //! specifies how many nodes are skipped from the beginning of the graph
  //! in this loaded portion of it
  uint64_t nodeOffset = 0;
  //! specifies how many edges are skipped from the beginning of the graph
  //! in this loaded portion of it
  uint64_t edgeOffset = 0;
  //! specifies whether or not the graph is loaded
  bool graphLoaded = false;

  // accumulators for tracking bytes read
  //! number of bytes read related to the out index buffer
  galois::GAccumulator<uint64_t> numBytesReadOutIndex;
  //! number of bytes read related to the edge dest buffer
  galois::GAccumulator<uint64_t> numBytesReadEdgeDest;
  //! number of bytes read related to the edge data buffer
  galois::GAccumulator<uint64_t> numBytesReadEdgeData;

  /**
   * Load the out indices (i.e. where a particular node's edges begin in the
   * array of edges) from the file.
   *
   * @param graphFile loaded file for the graph
   * @param nodeStart the first node to load
   * @param numNodesToLoad number of nodes to load
   */
  void loadOutIndex(std::ifstream& graphFile, uint64_t nodeStart,
                    uint64_t numNodesToLoad) {
    if (numNodesToLoad == 0) {
      return;
    }
    assert(outIndexBuffer == nullptr);
    outIndexBuffer = (uint64_t*)malloc(sizeof(uint64_t) * numNodesToLoad);

    if (outIndexBuffer == nullptr) {
      GALOIS_DIE("Failed to allocate memory for out index buffer.");
    }

    // position to start of contiguous chunk of nodes to read
    uint64_t readPosition = (4 + nodeStart) * sizeof(uint64_t);
    graphFile.seekg(readPosition);

    uint64_t numBytesToLoad = numNodesToLoad * sizeof(uint64_t);
    uint64_t bytesRead      = 0;

    while (numBytesToLoad > 0) {
      graphFile.read(((char*)this->outIndexBuffer) + bytesRead, numBytesToLoad);
      size_t numRead = graphFile.gcount();
      numBytesToLoad -= numRead;
      bytesRead += numRead;
    }

    assert(numBytesToLoad == 0);

    nodeOffset = nodeStart;
  }

  /**
   * Load the edge destination information from the file.
   *
   * @param graphFile loaded file for the graph
   * @param edgeStart the first edge to load
   * @param numEdgesToLoad number of edges to load
   * @param numGlobalNodes total number of nodes in the graph file; needed
   * to determine offset into the file
   */
  void loadEdgeDest(std::ifstream& graphFile, uint64_t edgeStart,
                    uint64_t numEdgesToLoad, uint64_t numGlobalNodes) {
    if (numEdgesToLoad == 0) {
      return;
    }

    assert(edgeDestBuffer == nullptr);
    edgeDestBuffer = (uint32_t*)malloc(sizeof(uint32_t) * numEdgesToLoad);

    if (edgeDestBuffer == nullptr) {
      GALOIS_DIE("Failed to allocate memory for edge dest buffer.");
    }

    // position to start of contiguous chunk of edges to read
    uint64_t readPosition = (4 + numGlobalNodes) * sizeof(uint64_t) +
                            (sizeof(uint32_t) * edgeStart);
    graphFile.seekg(readPosition);

    uint64_t numBytesToLoad = numEdgesToLoad * sizeof(uint32_t);
    uint64_t bytesRead      = 0;
    while (numBytesToLoad > 0) {
      graphFile.read(((char*)this->edgeDestBuffer) + bytesRead, numBytesToLoad);
      size_t numRead = graphFile.gcount();
      numBytesToLoad -= numRead;
      bytesRead += numRead;
    }

    assert(numBytesToLoad == 0);
    // save edge offset of this graph for later use
    edgeOffset = edgeStart;
  }

  /**
   * Load the edge data information from the file.
   *
   * @tparam EdgeType must be non-void in order to call this function
   *
   * @param edgeStart the first edge to load
   * @param numEdgesToLoad number of edges to load
   * @param numGlobalNodes total number of nodes in the graph file; needed
   * to determine offset into the file
   * @param numGlobalEdges total number of edges in the graph file; needed
   * to determine offset into the file
   */
  template <
      typename EdgeType,
      typename std::enable_if<!std::is_void<EdgeType>::value>::type* = nullptr>
  void loadEdgeData(std::ifstream& graphFile, uint64_t edgeStart,
                    uint64_t numEdgesToLoad, uint64_t numGlobalNodes,
                    uint64_t numGlobalEdges) {
    galois::gDebug("Loading edge data");

    if (numEdgesToLoad == 0) {
      return;
    }

    assert(edgeDataBuffer == nullptr);
    edgeDataBuffer =
        (EdgeDataType*)malloc(sizeof(EdgeDataType) * numEdgesToLoad);

    if (edgeDataBuffer == nullptr) {
      GALOIS_DIE("Failed to allocate memory for edge data buffer.");
    }

    // position after nodes + edges
    uint64_t baseReadPosition = (4 + numGlobalNodes) * sizeof(uint64_t) +
                                (sizeof(uint32_t) * numGlobalEdges);

    // version 1 padding TODO make version agnostic
    if (numGlobalEdges % 2) {
      baseReadPosition += sizeof(uint32_t);
    }

    // jump to first byte of edge data
    uint64_t readPosition =
        baseReadPosition + (sizeof(EdgeDataType) * edgeStart);
    graphFile.seekg(readPosition);
    uint64_t numBytesToLoad = numEdgesToLoad * sizeof(EdgeDataType);
    uint64_t bytesRead      = 0;

    while (numBytesToLoad > 0) {
      graphFile.read(((char*)this->edgeDataBuffer) + bytesRead, numBytesToLoad);
      size_t numRead = graphFile.gcount();
      numBytesToLoad -= numRead;
      bytesRead += numRead;
    }

    assert(numBytesToLoad == 0);
  }

  /**
   * Load edge data function for when the edge data type is void, i.e.
   * no edge data to load.
   *
   * Does nothing of importance.
   *
   * @tparam EdgeType if EdgeType is void, this function will be used
   */
  template <
      typename EdgeType,
      typename std::enable_if<std::is_void<EdgeType>::value>::type* = nullptr>
  void loadEdgeData(std::ifstream&, uint64_t, uint64_t, uint64_t, uint64_t) {
    galois::gDebug("Not loading edge data");
    // do nothing (edge data is void, i.e. no edge data)
  }

  /**
   * Resets graph metadata to default values. Does NOT touch the buffers.
   */
  void resetGraphStatus() {
    graphLoaded    = false;
    globalSize     = 0;
    globalEdgeSize = 0;
    nodeOffset     = 0;
    edgeOffset     = 0;
    numLocalNodes  = 0;
    numLocalEdges  = 0;
    resetReadCounters();
  }

  /**
   * Free all of the buffers in memory.
   */
  void freeMemory() {
    free(outIndexBuffer);
    outIndexBuffer = nullptr;
    free(edgeDestBuffer);
    edgeDestBuffer = nullptr;
    free(edgeDataBuffer);
    edgeDataBuffer = nullptr;
  }

public:
  /**
   * Class vars should be initialized by in-class initialization; all
   * left is to reset read counters.
   */
  BufferedGraph() { resetReadCounters(); }

  /**
   * On destruction, free allocated buffers (if necessary).
   */
  ~BufferedGraph() noexcept { freeMemory(); }

  // copy not allowed
  //! disabled copy constructor
  BufferedGraph(const BufferedGraph&) = delete;
  //! disabled copy constructor operator
  BufferedGraph& operator=(const BufferedGraph&) = delete;
  // move not allowed
  //! disabled move operator
  BufferedGraph(BufferedGraph&&) = delete;
  //! disabled move constructor operator
  BufferedGraph& operator=(BufferedGraph&&) = delete;

  /**
   * Gets the number of global nodes in the graph
   * @returns the total number of nodes in the graph (not just local loaded
   * nodes)
   */
  uint32_t size() const { return globalSize; }

  /**
   * Gets the number of global edges in the graph
   * @returns the total number of edges in the graph (not just local loaded
   * edges)
   */
  uint32_t sizeEdges() const { return globalEdgeSize; }

  //! @returns node offset of this buffered graph
  uint64_t getNodeOffset() const { return nodeOffset; }

  /**
   * Loads given Galois CSR graph into memory.
   *
   * @param filename name of graph to load; should be in Galois binary graph
   * format
   */
  void loadGraph(const std::string& filename) {
    if (graphLoaded) {
      GALOIS_DIE("Cannot load an buffered graph more than once.");
    }

    std::ifstream graphFile(filename.c_str());
    uint64_t header[4];
    graphFile.read(((char*)header), sizeof(uint64_t) * 4);

    numLocalNodes = globalSize = header[2];
    numLocalEdges = globalEdgeSize = header[3];

    loadOutIndex(graphFile, 0, globalSize);
    loadEdgeDest(graphFile, 0, globalEdgeSize, globalSize);
    // may or may not do something depending on EdgeDataType
    loadEdgeData<EdgeDataType>(graphFile, 0, globalEdgeSize, globalSize,
                               globalEdgeSize);
    graphLoaded = true;

    graphFile.close();
  }

  /**
   * Given a node/edge range to load, loads the specified portion of the graph
   * into memory buffers using read.
   *
   * @param filename name of graph to load; should be in Galois binary graph
   * format
   * @param nodeStart First node to load
   * @param nodeEnd Last node to load, non-inclusive
   * @param edgeStart First edge to load; should correspond to first edge of
   * first node
   * @param edgeEnd Last edge to load, non-inclusive
   * @param numGlobalNodes Total number of nodes in the graph
   * @param numGlobalEdges Total number of edges in the graph
   */
  void loadPartialGraph(const std::string& filename, uint64_t nodeStart,
                        uint64_t nodeEnd, uint64_t edgeStart, uint64_t edgeEnd,
                        uint64_t numGlobalNodes, uint64_t numGlobalEdges) {
    if (graphLoaded) {
      GALOIS_DIE("Cannot load an buffered graph more than once.");
    }

    std::ifstream graphFile(filename.c_str());

    globalSize     = numGlobalNodes;
    globalEdgeSize = numGlobalEdges;

    assert(nodeEnd >= nodeStart);
    numLocalNodes = nodeEnd - nodeStart;
    loadOutIndex(graphFile, nodeStart, numLocalNodes);

    assert(edgeEnd >= edgeStart);
    numLocalEdges = edgeEnd - edgeStart;
    loadEdgeDest(graphFile, edgeStart, numLocalEdges, numGlobalNodes);

    // may or may not do something depending on EdgeDataType
    loadEdgeData<EdgeDataType>(graphFile, edgeStart, numLocalEdges,
                               numGlobalNodes, numGlobalEdges);
    graphLoaded = true;

    graphFile.close();
  }

  //! Edge iterator typedef
  using EdgeIterator = boost::counting_iterator<uint64_t>;
  /**
   * Get the index to the first edge of the provided node THAT THIS GRAPH
   * HAS LOADED (not necessary the first edge of it globally).
   *
   * @param globalNodeID the global node id of the node to get the edge
   * for
   * @returns a GLOBAL edge id iterator
   */
  EdgeIterator edgeBegin(uint64_t globalNodeID) {
    if (!graphLoaded) {
      GALOIS_DIE("Graph hasn't been loaded yet.");
    }

    if (numLocalNodes == 0) {
      return EdgeIterator(0);
    }
    assert(nodeOffset <= globalNodeID);
    assert(globalNodeID < (nodeOffset + numLocalNodes));

    uint64_t localNodeID = globalNodeID - nodeOffset;

    if (localNodeID != 0) {
      numBytesReadOutIndex += sizeof(uint64_t);
      return EdgeIterator(outIndexBuffer[localNodeID - 1]);
    } else {
      return EdgeIterator(edgeOffset);
    }
  }

  /**
   * Get the index to the first edge of the node after the provided node.
   *
   * @param globalNodeID the global node id of the node to get the edge
   * for
   * @returns a GLOBAL edge id iterator
   */
  EdgeIterator edgeEnd(uint64_t globalNodeID) {
    if (!graphLoaded) {
      GALOIS_DIE("Graph hasn't been loaded yet.");
    }

    if (numLocalNodes == 0) {
      return EdgeIterator(0);
    }
    assert(nodeOffset <= globalNodeID);
    assert(globalNodeID < (nodeOffset + numLocalNodes));

    numBytesReadOutIndex += sizeof(uint64_t);

    uint64_t localNodeID = globalNodeID - nodeOffset;
    return EdgeIterator(outIndexBuffer[localNodeID]);
  }

  /**
   * Get the global node id of the destination of the provided edge.
   *
   * @param globalEdgeID the global edge id of the edge to get the destination
   * for (should obtain from edgeBegin/End)
   */
  uint64_t edgeDestination(uint64_t globalEdgeID) {
    if (!graphLoaded) {
      GALOIS_DIE("Graph hasn't been loaded yet.");
    }

    if (numLocalEdges == 0) {
      return 0;
    }
    assert(edgeOffset <= globalEdgeID);
    assert(globalEdgeID < (edgeOffset + numLocalEdges));

    numBytesReadEdgeDest += sizeof(uint32_t);

    uint64_t localEdgeID = globalEdgeID - edgeOffset;
    return edgeDestBuffer[localEdgeID];
  }

  /**
   * Get the edge data of some edge.
   *
   * @param globalEdgeID the global edge id of the edge to get the data of
   * @returns the edge data of the requested edge id
   */
  template <typename K = EdgeDataType,
            typename std::enable_if<!std::is_void<K>::value>::type* = nullptr>
  EdgeDataType edgeData(uint64_t globalEdgeID) {
    if (!graphLoaded) {
      GALOIS_DIE("Graph hasn't been loaded yet.");
    }
    if (edgeDataBuffer == nullptr) {
      GALOIS_DIE("Trying to get edge data when graph has no edge data.");
    }

    if (numLocalEdges == 0) {
      return 0;
    }

    assert(edgeOffset <= globalEdgeID);
    assert(globalEdgeID < (edgeOffset + numLocalEdges));

    numBytesReadEdgeData += sizeof(EdgeDataType);

    uint64_t localEdgeID = globalEdgeID - edgeOffset;
    return edgeDataBuffer[localEdgeID];
  }

  /**
   * Version of above function when edge data type is void.
   */
  template <typename K = EdgeDataType,
            typename std::enable_if<std::is_void<K>::value>::type* = nullptr>
  unsigned edgeData(uint64_t) {
    galois::gWarn("Getting edge data on graph when it doesn't exist\n");
    return 0;
  }

  /**
   * Reset reading counters.
   */
  void resetReadCounters() {
    numBytesReadOutIndex.reset();
    numBytesReadEdgeDest.reset();
    numBytesReadEdgeData.reset();
  }

  /**
   * Returns the total number of bytes read from this graph so far.
   *
   * @returns Total number of bytes read using the "get" functions on
   * out indices, edge destinations, and edge data.
   */
  uint64_t getBytesRead() {
    return numBytesReadOutIndex.reduce() + numBytesReadEdgeDest.reduce() +
           numBytesReadEdgeData.reduce();
  }

  /**
   * Free all of the in memory buffers in this object and reset graph status.
   */
  void resetAndFree() {
    freeMemory();
    resetGraphStatus();
  }
};
} // namespace graphs
} // namespace galois
#endif
