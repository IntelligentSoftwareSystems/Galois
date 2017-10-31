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

#include <mutex>
#include <fstream>
#include <random>
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
 * Gets a mapping of host to nodes of all hosts in the system. Divides
 * nodes evenly among hosts.
 *
 * @param numHosts total number of hosts
 * @param totalNumNodes total number of nodes
 * @returns A vector of pairs representing node -> host assignments. Evenly 
 * distributed nodes to hosts.
 */
std::vector<std::pair<uint64_t, uint64_t>> 
  getHostToNodeMapping(uint64_t numHosts, uint64_t totalNumNodes);

/**
 * Get the assigned owner of some ID given a mapping from ID to owner.
 *
 * @param gID ID to find owner of
 * @param ownerMapping Vector containing information about which host has which
 * nodes
 * @returns Owner of requested ID on or -1 if it couldn't be found
 */
uint32_t findOwner(const uint64_t gID, 
                 const std::vector<std::pair<uint64_t, uint64_t>>& ownerMapping);

/**
 * Returns the file size of an ifstream.
 *
 * @param openFile an open ifstream
 * @returns file size in bytes of the ifstream
 */
uint64_t getFileSize(std::ifstream& openFile);

/**
 * Determine the byte range that a host should read from a file.
 *
 * @param edgeListFile edge list file to read
 * @param fileSize total size of the file
 * @returns pair that represents the begin/end of this host's byte range to read
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
 * @param targetWeight desired weight that you want the index returned to have
 * @param lb Lower bound of search
 * @param ub Upper bound of search
 * @param prefixSum Prefix sum where the weights/returned index are derived
 * from
 */
uint64_t findIndexPrefixSum(uint64_t targetWeight, uint64_t lb, uint64_t ub,
                            const std::vector<uint64_t>& prefixSum);

/**
 * Given a prefix sum, a partition ID, and the total number of partitions, 
 * find a good contiguous division using the prefix sum such that 
 * partitions get roughly an even amount of units (based on prefix sum).
 *
 * @param id partition ID
 * @param totalID total number of partitions
 * @param prefixSum prefix sum of things that you want to divide among
 * partitions
 * @returns Pair representing the begin/end of the elements that partition
 * "id" is assigned based on the prefix sum
 */
std::pair<uint64_t, uint64_t> binSearchDivision(uint64_t id, uint64_t totalID, 
                                  const std::vector<uint64_t>& prefixSum);


/**
 * Finds the unique source nodes of a set of edges in memory. Assumes
 * edges are laid out in (src, dest) order in the vector.
 *
 * @param localEdges vector of edges to find unique sources of: needs to have
 * (src, dest) layout (i.e. i even: vector[i] is source, i+1 is dest
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
 * Get the edge counts for chunks of edges that we have locally.
 *
 * @param uniqueChunks unique chunks that this host has in its loaded edge list
 * @param localEdges loaded edge list laid out in src, dest, src, dest, etc.
 * @param chunkToNode specifies which chunks have which nodes
 * @param chunkCounts (input/output) a 0-initialized vector that will be 
 * edited to have our local chunk edge counts
 */
void accumulateLocalEdgesToChunks(const std::set<uint64_t>& uniqueChunks,
             const std::vector<uint32_t>& localEdges,
             const std::vector<std::pair<uint64_t, uint64_t>>& chunkToNode,
             std::vector<uint64_t>& chunkCounts);

/**
 * Synchronize chunk edge counts across all hosts, i.e. send and receive
 * local chunk counts and update them.
 *
 * @param chunkCounts local edge chunk counts to be updated to a global chunk
 * edge count across all hosts
 */
void sendAndReceiveEdgeChunkCounts(std::vector<uint64_t>& chunkCounts);

/**
 * Get the number of edges that each node chunk has.
 *
 * @param numNodeChunks total number of chunks
 * @param uniqueChunks unique chunks that this host has in its loaded edge list
 * @param localEdges loaded edge list laid out in src, dest, src, dest, etc.
 * @param chunkToNode specifies which chunks have which nodes
 * @returns A vector specifying the number of edges each chunk has
 */
std::vector<uint64_t> getChunkEdgeCounts(uint64_t numNodeChunks,
                const std::set<uint64_t>& uniqueChunks,
                const std::vector<uint32_t>& localEdges,
                const std::vector<std::pair<uint64_t, uint64_t>>& chunkToNode);


/**
 * Given a chunk edge count prefix sum and the chunk to node mapping, assign
 * chunks (i.e. nodes) to hosts in an attempt to keep hosts with an about even 
 * number of edges and return the node mapping.
 *
 * @param chunkCountsPrefixSum prefix sum of edges in chunks
 * @param chunkToNode mapping of chunk to nodes the chunk has
 * @returns a host to node mapping where each host very roughly has a balanced
 * number of edges
 */
std::vector<std::pair<uint64_t, uint64_t>> getChunkToHostMapping(
      const std::vector<uint64_t>& chunkCountsPrefixSum,
      const std::vector<std::pair<uint64_t, uint64_t>>& chunkToNode);

/**
 * Attempts to evenly assign nodes to hosts such that each host roughly gets
 * an even number of edges. 
 *
 * @param localEdges in-memory buffer of edges this host has loaded
 * @param totalNodeCount total number of nodes in the entire graph
 * @param totalEdgeCount total number of edges in the entire graph
 * @returns a mapping of host to nodes where each host gets an attempted
 * roughly even amount of edges
 */
std::vector<std::pair<uint64_t, uint64_t>> getEvenNodeToHostMapping(
    const std::vector<uint32_t>& localEdges, uint64_t totalNodeCount, 
    uint64_t totalEdgeCount
);

/**
 * Determine/send to each host how many edges they should expect to receive
 * from the caller (i.e. this host).
 *
 * @param hostToNodes mapping of a host to the nodes it is assigned
 * @param localEdges in-memory buffer of edges this host has loaded
 */
void sendEdgeCounts(
  const std::vector<std::pair<uint64_t, uint64_t>>& hostToNodes,
  const std::vector<uint32_t>& localEdges
); 

/**
 * Receive the messages from other hosts that tell this host how many edges
 * it should expect to receive. Should be called after sendEdgesCounts.
 *
 * @returns the number of edges that the caller host should expect to receive
 * in total from all other hosts
 */
uint64_t receiveEdgeCounts();

/**
 * Loop through all local edges and send them to the host they are assigned to.
 *
 * @param hostToNodes mapping of a host to the nodes it is assigned
 * @param localEdges in-memory buffer of edges this host has loaded
 * @param localSrcToDest local mapping of LOCAL sources to destinations (we
 * may have some edges that do not need sending; they are saved here)
 * @param nodeLocks Vector of mutexes (one for each local node) that are used
 * when writing to the local mapping of sources to destinations since vectors
 * are not thread safe
 */
void sendAssignedEdges( 
  const std::vector<std::pair<uint64_t, uint64_t>>& hostToNodes,
  const std::vector<uint32_t>& localEdges,
  std::vector<std::vector<uint32_t>>& localSrcToDest,
  std::vector<std::mutex>& nodeLocks
);

/**
 * Receive this host's assigned edges: should be called after sendAssignedEdges.
 *
 * @param edgesToReceive the number of edges we expect to receive; the function
 * will not exit until all expected edges are received
 * @param hostToNodes mapping of a host to the nodes it is assigned
 * @param localSrcToDest local mapping of LOCAL sources to destinations (we
 * may have some edges that do not need sending; they are saved here)
 * @param nodeLocks Vector of mutexes (one for each local node) that are used
 * when writing to the local mapping of sources to destinations since vectors
 * are not thread safe
 */
void receiveAssignedEdges(std::atomic<uint64_t>& edgesToReceive,
    const std::vector<std::pair<uint64_t, uint64_t>>& hostToNodes,
    std::vector<std::vector<uint32_t>>& localSrcToDest,
    std::vector<std::mutex>& nodeLocks);

/**
 * Send/receive other hosts number of assigned edges.
 *
 * @param localAssignedEdges number of edges assigned to this host
 * @returns a vector that has every hosts number of locally assigned edges
 */
std::vector<uint64_t> getEdgesPerHost(uint64_t localAssignedEdges);

/**
 * Writes a binary galois graph's header information.
 *
 * @param gr File to write to
 * @param version Version of the galois binary graph file
 * @param sizeOfEdge Size of edge data (0 if there is no edge data)
 * @param totalNumNodes total number of nodes in the graph
 * @param totalEdgeConnt total number of edges in the graph
 */
void writeGrHeader(MPI_File& gr, uint64_t version, uint64_t sizeOfEdge,
                   uint64_t totalNumNodes, uint64_t totalEdgeCount);

/**
 * Writes the node index data of a galois binary graph.
 *
 * @param gr File to write to
 * @param nodesToWrite number of nodes to write
 * @param nodeIndexOffset offset into file specifying where to start writing
 * @param edgePrefixSum the node index data to write into the file (index data
 * in graph tells you where to start looking for edges of some node, i.e.
 * it's a prefix sum)
 */
void writeNodeIndexData(MPI_File& gr, uint64_t nodesToWrite, 
                        uint64_t nodeIndexOffset, 
                        std::vector<uint64_t>& edgePrefixSum);

/**
 * Writes the edge destination data of a galois binary graph.
 *
 * @param gr File to write to
 * @param localNumNodes number of source nodes that this host was 
 * assigned to write
 * @param edgeDestOffset offset into file specifying where to start writing
 * @param localSrcToDest Vector of vectors: the vector at index i specifies
 * the destinations for local src node i
 */
void writeEdgeDestData(MPI_File& gr, uint64_t localNumNodes, 
                       uint64_t edgeDestOffset,
                       std::vector<std::vector<uint32_t>>& localSrcToDest);

/**
 * Writes the edge data data of a galois binary graph.
 *
 * @param gr File to write to
 * @param localNumEdges number of edges to write edge data for
 * @param edgeDataOffset offset into file specifying where to start writing
 * @param edgeDataToWrite vector of localNumEdges elements corresponding to
 * edge data that needs to be written
 */
void writeEdgeDataData(MPI_File& gr, uint64_t localNumEdges,
                       uint64_t edgeDataOffset,
                       const std::vector<uint32_t>& edgeDataToWrite);

/**
 * Generates a vector of random uint32_ts.
 *
 * @param count number of numbers to generate
 * @param seed seed to start generating with
 * @param lower lower bound of numbers to generate, inclusive
 * @param upper upper bound of number to generate, inclusive
 * @returns Vector of random uint32_t numbers
 */
std::vector<uint32_t> generateRandomNumbers(uint64_t count, uint64_t seed, 
                                            uint64_t lower, uint64_t upper);

/**
 * Gets the offset into the location of the edge data of some edge in a galois
 * binary graph file.
 *
 * @param totalNumNodes total number of nodes in graph
 * @param totalNumEdges total number of edges in graph
 * @param localEdgeBegin the edge to get the offset to
 * @returns offset into location of edge data of localEdgeBegin
 */
uint64_t getOffsetToLocalEdgeData(uint64_t totalNumNodes, 
                                  uint64_t totalNumEdges, 
                                  uint64_t localEdgeBegin);

/**
 * Given some number, get the chunk of that number that this host is responsible
 * for.
 *
 * @param numToSplit the number to chunk among hosts
 * @returns pair specifying the range that this host is responsible for
 */
std::pair<uint64_t, uint64_t> getLocalAssignment(uint64_t numToSplit);
#endif
