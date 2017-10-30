/** Distributed graph converter helpers (header) -*- C++ -*-
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
 * Distributed graph converter helper signatures/definitions.
 *
 * @author Loc Hoang <l_hoang@utexas.edu>
 */
#ifndef _GALOIS_DIST_CONVERT_HELP_
#define _GALOIS_DIST_CONVERT_HELP_

#include <fstream>
#include <mpi.h>

#include "galois/Galois.h"
#include "galois/gstl.h"
#include "galois/runtime/Network.h"
#include "galois/DistAccumulator.h"

/**
 * Wrapper for MPI calls that return an error code. Make sure it is success
 * else die.
 *
 * @param errcode error code returned by an mpi call
 */
void MPICheck(int errcode);

/**
 * "Free" memory used by a vector by swapping it out with an empty one.
 *
 * @tparam VectorTy type of vector 
 * @param toFree vector to free memory of
 */
template <typename VectorTy>
void freeVector(VectorTy& toFree) {
  VectorTy dummyVector;
  toFree.swap(dummyVector);
}

/**
 * TODO documentation
 */
std::vector<std::pair<uint64_t, uint64_t>> 
  getHostToNodeMapping(uint64_t numHosts, uint64_t totalNumNodes);

/**
 * Get the assigned host of some node given its global id.
 *
 * @param gID global ID of a node
 * @param hostToNodes Vector containing information about which host has which
 * nodes
 * @returns Host that requested node resides on or -1 if it couldn't be found
 */
uint32_t findHostID(const uint64_t gID, 
                 const std::vector<std::pair<uint64_t, uint64_t>>& hostToNodes);

/**
 * Returns the file size of an ifstream.
 *
 * @param openFile an open ifstream
 * @returns file size in bytes of the ifstream
 */
uint64_t getFileSize(std::ifstream& openFile);

/**
 * TODO
 */
std::pair<uint64_t, uint64_t> determineByteRange(std::ifstream& edgeListFile,
                                                 uint64_t fileSize);

/**
 * Accumulates some value from all hosts + return it.
 *
 * @param value value to accumulate across hosts
 * @return Accumulated value (add all values from all hosts up)
 */
uint64_t accumulateValue(uint64_t value);

/**
 * Find an index into the provided prefix sum that gets the desired "weight"
 * (weight comes from the units of the prefix sum).
 * 
 * TODO params
 */
uint64_t findIndexPrefixSum(uint64_t targetWeight, uint64_t lb, uint64_t ub,
                            const std::vector<uint64_t>& prefixSum);

/**
 * Given a prefix sum, a partition ID, and the total number of partitions, 
 * find a good contiguous division using the prefix sum such that 
 * partitions get roughly an even amount of units (based on prefix sum).
 *
 * TODO params
 */
std::pair<uint64_t, uint64_t> binSearchDivision(uint64_t id, uint64_t totalID, 
                                  const std::vector<uint64_t>& prefixSum);


/**
 * Finds the unique source nodes of a set of edges in memory. Assumes
 * edges are laid out in (src, dest) order in the vector.
 *
 * @param localEdges vector of edges to find unique sources of: needs to have
 * (src, dest) layout
 * @returns Set of global IDs of unique sources found in the provided edge
 * vector
 */
std::set<uint64_t>
findUniqueSourceNodes(const std::vector<uint32_t>& localEdges);

/**
 * Given a chunk to node mapping and a set of unique nodes, find the unique 
 * chunks corresponding to the unique nodes provided.
 *
 * @param uniqueNodes set of unique nodes
 * @param chunkToNode a mapping of a chunk to the range of nodes that the chunk
 * has
 * @returns a set of chunk ids corresponding to the nodes passed in (i.e. chunks
 * those nodes are included in)
 */
std::set<uint64_t> 
findUniqueChunks(const std::set<uint64_t>& uniqueNodes,
                 const std::vector<std::pair<uint64_t, uint64_t>>& chunkToNode);

/**
 * Attempts to evenly assign nodes to hosts such that each host roughly gets
 * an even number of edges. 
 *
 * TODO params
 */
std::vector<std::pair<uint64_t, uint64_t>> getEvenNodeToHostMapping(
    const std::vector<uint32_t>& localEdges, uint64_t totalNodeCount, 
    uint64_t totalEdgeCount
);

/**
 * Determine/send to each host how many edges they should expect to receive
 * from the caller (i.e. this host).
 *
 * TODO params
 */
void sendEdgeCounts(
  const std::vector<std::pair<uint64_t, uint64_t>>& hostToNodes,
  uint64_t localNumEdges, const std::vector<uint32_t>& localEdges
); 

/**
 * Receive the messages from other hosts that tell this host how many edges
 * it should expect to receive. Should be called after sendEdgesCounts.
 *
 * TODO params
 */
uint64_t receiveEdgeCounts();

/**
 * TODO
 */
void sendAssignedEdges( 
  const std::vector<std::pair<uint64_t, uint64_t>>& hostToNodes,
  uint64_t localNumEdges, const std::vector<uint32_t>& localEdges,
  std::vector<std::vector<uint32_t>>& localSrcToDest,
  std::vector<std::mutex>& nodeLocks
);

/**
 * TODO
 */
void receiveAssignedEdges(std::atomic<uint64_t>& edgesToReceive,
    const std::vector<std::pair<uint64_t, uint64_t>>& hostToNodes,
    std::vector<std::vector<uint32_t>>& localSrcToDest,
    std::vector<std::mutex>& nodeLocks);

/**
 * Send/receive other hosts number of assigned edges.
 *
 * TODO
 */
std::vector<uint64_t> getEdgesPerHost(uint64_t localAssignedEdges);

/**
 * TODO
 */
void writeGrHeader(MPI_File& gr, uint64_t version, uint64_t sizeOfEdge,
                   uint64_t totalNumNodes, uint64_t totalEdgeCount);

/**
 * TODO
 */
void writeNodeIndexData(MPI_File& gr, uint64_t nodesToWrite, 
                        uint64_t nodeIndexOffset, 
                        std::vector<uint64_t>& edgePrefixSum);

/**
 * TODO
 */
void writeEdgeDestData(MPI_File& gr, uint64_t localNumNodes, 
                       uint64_t edgeDestOffset,
                       std::vector<std::vector<uint32_t>>& localSrcToDest);
#endif
