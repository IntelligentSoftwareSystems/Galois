/** MPI Graph -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2017, The University of Texas at Austin. All rights reserved.
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
 *
 * @section Description
 *
 * Graph that uses MPI read to load relavent portions of a graph into memory
 * for later reading.
 *
 * @author Loc Hoang <l_hoang@utexas.edu>
 */

#ifndef GALOIS_GRAPH_MPIGRAPH_H
#define GALOIS_GRAPH_MPIGRAPH_H

#include <galois/gIO.h>
#include <galois/Reduction.h>
#include <mpi.h>

#include <boost/iterator/counting_iterator.hpp>
namespace galois {
namespace graphs {

template <typename EdgeDataType>
class MPIGraph {
private:
  // buffers that you load data into
  uint64_t* outIndexBuffer;
  uint32_t* edgeDestBuffer;
  EdgeDataType* edgeDataBuffer;

  uint64_t numLocalNodes;
  uint64_t numLocalEdges;

  uint64_t nodeOffset;
  uint64_t edgeOffset;
  bool graphLoaded;

  typedef boost::counting_iterator<uint64_t> EdgeIterator;

  // accumulators for tracking bytes read
  galois::GAccumulator<uint64_t> numBytesReadOutIndex;
  galois::GAccumulator<uint64_t> numBytesReadEdgeDest;
  galois::GAccumulator<uint64_t> numBytesReadEdgeData;

  /**
   * Initialize the MPI system if it hasn't been initialized.
   */
  void initializeMPI() {
    int mpiInitialized = 0;

    MPI_Initialized(&mpiInitialized);

    if (!mpiInitialized) {
      char*** dummyBuffer = nullptr;
      MPI_Init(0, dummyBuffer);
    }
  }

  /**
   * Load the out indices (i.e. where a particular node's edges begin in the
   * array of edges) from the file.
   *
   * @param graphFile loaded MPI file for the graph
   * @param nodeStart the first node to load
   * @param numNodesToLoad number of nodes to load
   */
  void loadOutIndex(MPI_File& graphFile, uint64_t nodeStart, 
                    uint64_t numNodesToLoad) {
    if (numNodesToLoad == 0) {
      return;
    }
    assert(outIndexBuffer == nullptr);
    outIndexBuffer = (uint64_t*)malloc(sizeof(uint64_t) * numNodesToLoad);

    // position to start of contiguous chunk of nodes to read
    uint64_t readPosition = (4 + nodeStart) * sizeof(uint64_t);

    MPI_File_read_at(graphFile, readPosition, (char*)outIndexBuffer, 
                     numNodesToLoad, MPI_UINT64_T, MPI_STATUS_IGNORE); 

    nodeOffset = nodeStart;
  }

  /**
   * Load the edge destination information from the file.
   *
   * @param graphFile loaded MPI file for the graph
   * @param edgeStart the first node to load
   * @param numEdgesToLoad number of edges to load
   * @param numGlobalNodes total number of nodes in the graph file; needed
   * to determine offset into the file
   */
  void loadEdgeDest(MPI_File& graphFile, uint64_t edgeStart, 
                    uint64_t numEdgesToLoad, uint64_t numGlobalNodes) {
    if (numEdgesToLoad == 0) {
      return;
    }

    assert(edgeDestBuffer == nullptr);
    edgeDestBuffer = (uint32_t*)malloc(sizeof(uint32_t) * numEdgesToLoad);

    // position to start of contiguous chunk of edges to read
    uint64_t readPosition = (4 + numGlobalNodes) * sizeof(uint64_t) +
                            (sizeof(uint32_t) * edgeStart);

    MPI_File_read_at(graphFile, readPosition, (char*)edgeDestBuffer, 
                     numEdgesToLoad, MPI_UINT32_T, MPI_STATUS_IGNORE); 


    edgeOffset = edgeStart;
  }

  //template<typename std::enable_if<!std::is_void(EdgeDataType)>::type* = nullptr>
  //void loadOutIndex() {

  //}

  /**
   * TODO
   */
  template<typename std::enable_if<std::is_void<EdgeDataType>::value>::type* = 
                                   nullptr>
  void loadEdgeData(MPI_File& graphFile, uint64_t edgeStart, 
                    uint64_t numEdgesToLoad, uint64_t numGlobalNodes) {
    // do nothing (edge data is void, i.e. no edge data)
  }

public:
  /**
   * Initialize class variables and MPI if necessary.
   */
  MPIGraph() {
    outIndexBuffer = nullptr;
    edgeDestBuffer = nullptr;
    edgeDataBuffer = nullptr;
    graphLoaded = false;
    nodeOffset = 0;
    edgeOffset = 0;

    resetReadCounters();
    initializeMPI();
  }

  /**
   * On destruction, free allocated buffers.
   */
  ~MPIGraph() {
    if (outIndexBuffer != nullptr) {
      free(outIndexBuffer);
    }
    if (edgeDestBuffer != nullptr) {
      free(edgeDestBuffer);
    }
    if (edgeDataBuffer != nullptr) {
      free(edgeDataBuffer);
    }
  }

  /**
   * Load the entire graph specified by the filename.
   */
  // UNIMPLEMENTED
  //void loadFullGraph() {
  //}

  /**
   * Given a node/edge range to load, loads the specified portion of the graph 
   * into memory buffers using MPI read.
   */
  void loadPartialGraph(const std::string& filename, uint64_t nodeStart,
                        uint64_t nodeEnd, uint64_t edgeStart, 
                        uint64_t edgeEnd, uint64_t numGlobalNodes) {
    if (graphLoaded) {
      GALOIS_DIE("Cannot load an MPI graph more than once.");
    }
    printf("edge start %lu edge end %lu\n", edgeStart, edgeEnd);

    MPI_File graphFile;
    // TODO can give striping info to file open?
    MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_RDONLY, 
                  MPI_INFO_NULL, &graphFile);

    assert(nodeEnd >= nodeStart);
    numLocalNodes = nodeEnd - nodeStart;
    loadOutIndex(graphFile, nodeStart, numLocalNodes);

    assert(edgeEnd >= edgeStart);
    numLocalEdges = edgeEnd - edgeStart;
    loadEdgeDest(graphFile, edgeStart, numLocalEdges, numGlobalNodes);

    // TODO edge data stuff

    graphLoaded = true;

    int d = MPI_File_close(&graphFile);
  }

  /**
   * Get the index to the first edge of the provided node.
   *
   * @param globalNodeID the global node id of the node to get the edge
   * for
   * @returns a GLOBAL edge id
   */
  EdgeIterator edgeBegin(uint64_t globalNodeID) {
    assert(graphLoaded);
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
   */
  EdgeIterator edgeEnd(uint64_t globalNodeID) {
    assert(graphLoaded);
    assert(nodeOffset <= globalNodeID);
    assert(globalNodeID < (nodeOffset + numLocalNodes));

    numBytesReadOutIndex += sizeof(uint64_t);

    uint64_t localNodeID = globalNodeID - nodeOffset;
    return EdgeIterator(outIndexBuffer[localNodeID]);
  }

  /**
   * Get the global node id of the destination of the provided edge.
   *
   * @param localEdgeID the LOCAL edge id of the edge to get the destination
   * for (should obtain from edgeBegin/End)
   */
  uint64_t edgeDestination(uint64_t globalEdgeID) {
    assert(graphLoaded);
    if (edgeOffset > globalEdgeID) {
      printf("edge offset is %lu, id %lu\n", edgeOffset, globalEdgeID);
    }
    assert(edgeOffset <= globalEdgeID); 
    assert(globalEdgeID < (edgeOffset + numLocalEdges));

    numBytesReadEdgeDest += sizeof(uint32_t);

    uint64_t localEdgeID = globalEdgeID - edgeOffset;
    return edgeDestBuffer[localEdgeID];
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
    return numBytesReadOutIndex.reduce() +
           numBytesReadEdgeDest.reduce() +
           numBytesReadEdgeData.reduce();
  }
};


} // end graph namespace
} // end galois namespace
#endif
