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

#ifndef _GALOIS_DIST_CONVERT_HELP_
#define _GALOIS_DIST_CONVERT_HELP_

#include <iostream>
#include <mutex>
#include <random>

#include <mpi.h>

#include "galois/Galois.h"
#include "galois/DynamicBitset.h"
#include "galois/gstl.h"
#include "galois/runtime/Network.h"
#include "galois/DReducible.h"
#include "galois/graphs/OfflineGraph.h"
#include "galois/graphs/BufferedGraph.h"

// useful typedefs that shorten long declarations
using Uint64Pair       = std::pair<uint64_t, uint64_t>;
using DoubleUint64Pair = std::pair<Uint64Pair, Uint64Pair>;
using VoVUint32        = std::vector<std::vector<uint32_t>>;
using PairVoVUint32    = std::pair<VoVUint32, VoVUint32>;

/**
 * Given a binary node mapping, read a specified region into memory.
 *
 * @param nodeMapBinary name of the binary file with node mappings
 * @param nodeOffset node offset to begin reading at
 * @param numToRead the number of node mappings (i.e. nodes) to read
 * @returns Vector with the read in node mappings
 */
std::vector<uint32_t> readRandomNodeMapping(const std::string& nodeMapBinary,
                                            uint64_t nodeOffset,
                                            uint64_t numToRead);

/**
 * Wrapper for MPI calls that return an error code. Make sure it is success
 * else die.
 *
 * @param errcode error code returned by an mpi call
 */
void MPICheck(int errcode);

/**
 * Reads a gr header from a v1 gr binary file and return number of
 * nodes and edges.
 *
 * @param grFile file name of gr
 * @param isVoid true if edge data shouldn't exist in the graph,
 * false otherwise; used for a safety check
 * @returns a pair with the number of nodes and number of edges in the gr file
 * in that order
 */
Uint64Pair readV1GrHeader(const std::string& grFile, bool isVoid);

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
 * Given a vector representing edges, get the number of edges the vector
 * represents.
 *
 * @tparam EdgeDataTy type of edge data to read
 * @param edgeVector vector with edges laid out in src, dest, and optionally
 * data order (i.e. 3 elements)
 * @returns the number of edges represented by the vector
 */
template <typename EdgeDataTy>
size_t getNumEdges(const std::vector<uint32_t>& edgeVector) {
  size_t numEdges;
  if (std::is_void<EdgeDataTy>::value) {
    numEdges = edgeVector.size() / 2;
  } else {
    numEdges = edgeVector.size() / 3;
  }
  return numEdges;
}

/**
 * Given an open ifstream of an edgelist and a range to read,
 * read the edges into memory.
 *
 * @tparam EdgeDataTy type of edge data to read
 * @param edgeListFile open ifstream of an edge list
 * @param localStartByte First byte to read
 * @param localEndByte Last byte to read (non-inclusive)
 * @param totalNumNodes Total number of nodes in the graph: used for correctness
 * checking of src/dest ids
 * @param startAtOne true if the edge list node ids start at 1
 * @returns Vector representing the read in edges: every 2-3 elements represents
 * src, dest, and edge data (if the latter exists)
 */
template <typename EdgeDataTy>
std::vector<uint32_t>
loadEdgesFromEdgeList(std::ifstream& edgeListFile, uint64_t localStartByte,
                      uint64_t localEndByte, uint64_t totalNumNodes,
                      bool startAtOne = false, bool ignoreWeights = false) {
  // load edges into a vector
  uint64_t localNumEdges = 0;
  std::vector<uint32_t> localEdges; // v1 support only + only uint32_t data

  // read lines until last byte
  edgeListFile.seekg(localStartByte);
  while ((uint64_t(edgeListFile.tellg()) + 1ul) != localEndByte) {
    uint64_t src;
    uint64_t dst;
    edgeListFile >> src >> dst;
    if (startAtOne) {
      src--;
      dst--;
    }
    GALOIS_ASSERT(src < totalNumNodes, "src ", src, " and ", totalNumNodes);
    GALOIS_ASSERT(dst < totalNumNodes, "dst ", dst, " and ", totalNumNodes);
    localEdges.emplace_back(src);
    localEdges.emplace_back(dst);

    // get (or ignore)edge data: IT ONLY SUPPORTS uint32_t AT THE MOMENT
    // TODO function template specializations necessary to read other graph
    // data types
    if (ignoreWeights) {
      // skip edgeweights
      edgeListFile >> src;
    } else if (!std::is_void<EdgeDataTy>::value) {
      uint32_t edgeData;
      edgeListFile >> edgeData;
      localEdges.emplace_back(edgeData);
    }

    localNumEdges++;
  }

  if (std::is_void<EdgeDataTy>::value) {
    GALOIS_ASSERT(localNumEdges == (localEdges.size() / 2));
  } else {
    GALOIS_ASSERT(localNumEdges == (localEdges.size() / 3));
  }

  std::cout << "[" << galois::runtime::getSystemNetworkInterface().ID << "] "
            << "Local num edges from file is " << localNumEdges << "\n";

  return localEdges;
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
std::vector<Uint64Pair> getHostToNodeMapping(uint64_t numHosts,
                                             uint64_t totalNumNodes);

/**
 * Get the assigned owner of some ID given a mapping from ID to owner.
 *
 * @param gID ID to find owner of
 * @param ownerMapping Vector containing information about which host has which
 * nodes
 * @returns Owner of requested ID on or -1 if it couldn't be found
 */
uint32_t findOwner(const uint64_t gID,
                   const std::vector<Uint64Pair>& ownerMapping);

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
Uint64Pair determineByteRange(std::ifstream& edgeListFile, uint64_t fileSize);

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
Uint64Pair binSearchDivision(uint64_t id, uint64_t totalID,
                             const std::vector<uint64_t>& prefixSum);

/**
 * Finds the unique source nodes of a set of edges in memory. Assumes
 * edges are laid out in (src, dest) order in the vector.
 *
 * @tparam EdgeDataTy type of edge data to read
 * @param localEdges vector of edges to find unique sources of: needs to have
 * (src, dest) layout (i.e. i even: vector[i] is source, i+1 is dest
 * @param uniqueNodeBitset bitset marking which unique nodes are present
 * on this host; should be pre-initialized before being passed into this
 * function
 */
template <typename EdgeDataTy>
void findUniqueSourceNodes(const std::vector<uint32_t>& localEdges,
                           galois::DynamicBitSet& uniqueNodeBitset) {
  uint64_t hostID = galois::runtime::getSystemNetworkInterface().ID;
  std::cout << "[" << hostID << "] Finding unique nodes\n";
  uniqueNodeBitset.reset();

  uint64_t localNumEdges = getNumEdges<EdgeDataTy>(localEdges);
  galois::do_all(
      galois::iterate((uint64_t)0, localNumEdges),
      [&](uint64_t edgeIndex) {
        // src node
        if (std::is_void<EdgeDataTy>::value) {
          uniqueNodeBitset.set(localEdges[edgeIndex * 2]);
        } else {
          uniqueNodeBitset.set(localEdges[edgeIndex * 3]);
        }
      },
      galois::loopname("FindUniqueNodes"));

  std::cout << "[" << hostID << "] Unique nodes found\n";
}

/**
 * Given a chunk to node mapping and a set of unique nodes, find the unique
 * chunks corresponding to the unique nodes provided.
 *
 * @param uniqueNodeBitset Bitset specifying which source nodes exist on the
 * edges this host has read
 * @param chunkToNode a mapping of a chunk to the range of nodes that the chunk
 * has
 * @returns a set of chunk ids corresponding to the nodes passed in (i.e. chunks
 * those nodes are included in)
 */
void findUniqueChunks(galois::DynamicBitSet& uniqueNodeBitset,
                      const std::vector<Uint64Pair>& chunkToNode,
                      galois::DynamicBitSet& uniqueChunkBitset);

/**
 * Get the edge counts for chunks of edges that we have locally.
 *
 * @tparam EdgeDataTy type of edge data to read
 * @param uniqueChunkBitset Bitset specifying which chunks are present on
 * this host; will be unusable at the end of the function (memory free)
 * @param localEdges loaded edge list laid out in src, dest, src, dest, etc.
 * @param chunkToNode specifies which chunks have which nodes
 * @param chunkCounts (input/output) a 0-initialized vector that will be
 * edited to have our local chunk edge counts
 */
template <typename EdgeDataTy>
void accumulateLocalEdgesToChunks(galois::DynamicBitSet& uniqueChunkBitset,
                                  const std::vector<uint32_t>& localEdges,
                                  const std::vector<Uint64Pair>& chunkToNode,
                                  std::vector<uint64_t>& chunkCounts) {
  std::map<uint64_t, std::atomic<uint64_t>> chunkToAccumulator;

  // default-initialize necessary chunk atomics
  for (size_t i = 0; i < uniqueChunkBitset.size(); i++) {
    if (uniqueChunkBitset.test(i)) {
      chunkToAccumulator[i];
    }
  }

  freeVector(uniqueChunkBitset.get_vec());

  uint64_t hostID = galois::runtime::getSystemNetworkInterface().ID;
  std::cout << "[" << hostID
            << "] Chunk accumulators created: " << chunkToAccumulator.size()
            << " of them\n";

  uint64_t localNumEdges = getNumEdges<EdgeDataTy>(localEdges);
  // determine which chunk edges go to
  galois::do_all(
      galois::iterate((uint64_t)0, localNumEdges),
      [&](uint64_t edgeIndex) {
        uint32_t src;
        if (std::is_void<EdgeDataTy>::value) {
          src = localEdges[edgeIndex * 2];
        } else {
          src = localEdges[edgeIndex * 3];
        }
        uint32_t chunkNum = findOwner(src, chunkToNode);
        GALOIS_ASSERT(chunkNum != (uint32_t)-1);
        chunkToAccumulator[chunkNum] += 1;
      },
      galois::loopname("ChunkInspection"));

  std::cout << "[" << hostID << "] Chunk accumulators done accumulating\n";

  // update chunk count
  galois::do_all(
      galois::iterate(chunkToAccumulator.cbegin(), chunkToAccumulator.cend()),
      [&](auto& chunkAndCount) {
        chunkCounts[chunkAndCount.first] += chunkAndCount.second.load();
      },
      galois::loopname("ChunkCountUpdate"));
}

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
 * @tparam EdgeDataTy type of edge data to read
 * @param uniqueChunkBitset bitset specifying which chunks are present on this
 * host
 * @param localEdges loaded edge list laid out in src, dest, src, dest, etc.
 * @param chunkToNode specifies which chunks have which nodes
 * @returns A vector specifying the number of edges each chunk has
 */
template <typename EdgeDataTy>
std::vector<uint64_t>
getChunkEdgeCounts(galois::DynamicBitSet& uniqueChunkBitset,
                   const std::vector<uint32_t>& localEdges,
                   const std::vector<Uint64Pair>& chunkToNode) {
  std::vector<uint64_t> chunkCounts;
  chunkCounts.assign(uniqueChunkBitset.size(), 0);
  accumulateLocalEdgesToChunks<EdgeDataTy>(uniqueChunkBitset, localEdges,
                                           chunkToNode, chunkCounts);
  sendAndReceiveEdgeChunkCounts(chunkCounts);

  return chunkCounts;
}

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
std::vector<Uint64Pair>
getChunkToHostMapping(const std::vector<uint64_t>& chunkCountsPrefixSum,
                      const std::vector<Uint64Pair>& chunkToNode);

/**
 * Given a chunk edge count prefix sum and the chunk to node mapping, assign
 * chunks (i.e. nodes) to hosts in an attempt to keep hosts with an about even
 * number of edges and return the node mapping. LINEAR SEARCH VERSION.
 *
 * @param chunkCountsPrefixSum prefix sum of edges in chunks
 * @param chunkToNode mapping of chunk to nodes the chunk has
 * @returns a host to node mapping where each host very roughly has a balanced
 * number of edges
 */
// TODO this crashes for some reason
std::vector<Uint64Pair>
getChunkToHostMappingLinear(const std::vector<uint64_t>& chunkCountsPrefixSum,
                            const std::vector<Uint64Pair>& chunkToNode);

/**
 * Attempts to evenly assign nodes to hosts such that each host roughly gets
 * an even number of edges.
 *
 * @tparam EdgeDataTy type of edge data to read
 * @param localEdges in-memory buffer of edges this host has loaded
 * @param totalNodeCount total number of nodes in the entire graph
 * @param totalEdgeCount total number of edges in the entire graph
 * @returns a mapping of host to nodes where each host gets an attempted
 * roughly even amount of edges
 */
template <typename EdgeDataTy>
std::vector<Uint64Pair>
getEvenNodeToHostMapping(const std::vector<uint32_t>& localEdges,
                         uint64_t totalNodeCount, uint64_t totalEdgeCount) {
  auto& net              = galois::runtime::getSystemNetworkInterface();
  uint64_t hostID        = net.ID;
  uint64_t totalNumHosts = net.Num;

  uint64_t numNodeChunks = totalEdgeCount / totalNumHosts;
  // TODO better heuristics: basically we don't want to run out of memory,
  // so keep number of chunks from growing too large
  while (numNodeChunks > 10000000) {
    numNodeChunks /= 2;
  }

  std::vector<Uint64Pair> chunkToNode;

  if (hostID == 0) {
    std::cout << "Num chunks is " << numNodeChunks << "\n";
  }

  for (unsigned i = 0; i < numNodeChunks; i++) {
    chunkToNode.emplace_back(galois::block_range(
        (uint64_t)0, (uint64_t)totalNodeCount, i, numNodeChunks));
  }

  std::cout << "[" << hostID << "] Determining edge to chunk counts\n";

  galois::DynamicBitSet uniqueNodeBitset;
  uniqueNodeBitset.resize(totalNodeCount);
  findUniqueSourceNodes<EdgeDataTy>(localEdges, uniqueNodeBitset);

  galois::DynamicBitSet uniqueChunkBitset;
  uniqueChunkBitset.resize(numNodeChunks);
  findUniqueChunks(uniqueNodeBitset, chunkToNode, uniqueChunkBitset);

  std::vector<uint64_t> chunkCounts = getChunkEdgeCounts<EdgeDataTy>(
      uniqueChunkBitset, localEdges, chunkToNode);
  std::cout << "[" << hostID << "] Edge to chunk counts determined\n";

  // prefix sum on the chunks (reuse array to save memory)
  for (unsigned i = 1; i < numNodeChunks; i++) {
    chunkCounts[i] += chunkCounts[i - 1];
  }

  // to make access to chunkToNode's last element correct with regard to later
  // access (without this access to chunkToNode[chunkSize] is out of bounds)
  chunkToNode.emplace_back(Uint64Pair(totalNodeCount, totalNodeCount));

  std::vector<Uint64Pair> finalMapping =
      getChunkToHostMapping(chunkCounts, chunkToNode);

  return finalMapping;
}

/**
 * Using OfflineGraph to read the binary gr, divide nodes among hosts such
 * that each hosts gets roughly an even amount of edges to read.
 *
 * @param inputGr file name of the input Galois binary graph
 * @returns 2 pairs: 1 pair specifies what nodes this host is responsible
 * for reading, the other pair specifies what edges this host is responsible
 * for reading
 */
DoubleUint64Pair getNodesToReadFromGr(const std::string& inputGr);

/**
 * Load a Galois binary graph into an BufferedGraph and load assigned
 * nodes/edges into memory.
 *
 * @tparam EdgeDataTy type of edge data to read
 * @param inputFile path to input Galois binary graph
 * @param nodesToRead a pair that has the range of nodes that should be read
 * @param edgesToRead a pair that has the range of edges that should be read
 * @param totalNumNodes Total number of nodes in the graph
 * @param totalNumEdges Total number of edges in the graph
 * @returns a vector with edges corresponding to the nodes/edges pass into the
 * function
 */
template <typename EdgeDataTy>
std::vector<uint32_t>
loadEdgesFromBufferedGraph(const std::string& inputFile, Uint64Pair nodesToRead,
                           Uint64Pair edgesToRead, uint64_t totalNumNodes,
                           uint64_t totalNumEdges) {
  galois::graphs::BufferedGraph<EdgeDataTy> bufGraph;
  bufGraph.loadPartialGraph(inputFile, nodesToRead.first, nodesToRead.second,
                            edgesToRead.first, edgesToRead.second,
                            totalNumNodes, totalNumEdges);

  std::vector<uint32_t> edgeData;

  // void = 2 elements per edge; non-void = 3 elements per edge
  if (std::is_void<EdgeDataTy>::value) {
    edgeData.resize((edgesToRead.second - edgesToRead.first) * 2);
  } else {
    edgeData.resize((edgesToRead.second - edgesToRead.first) * 3);
  }

  if (edgeData.size() > 0) {
    galois::do_all(
        galois::iterate(nodesToRead.first, nodesToRead.second),
        [&](uint32_t gID) {
          uint64_t edgeBegin = *bufGraph.edgeBegin(gID);
          uint64_t edgeEnd   = *bufGraph.edgeEnd(gID);

          // offset into which we should start writing data in
          // edgeData
          uint64_t edgeDataOffset;
          if (std::is_void<EdgeDataTy>::value) {
            edgeDataOffset = (edgeBegin - edgesToRead.first) * 2;
          } else {
            edgeDataOffset = (edgeBegin - edgesToRead.first) * 3;
          }

          // loop through all edges
          for (uint64_t i = edgeBegin; i < edgeEnd; i++) {
            uint32_t edgeDest            = bufGraph.edgeDestination(i);
            edgeData[edgeDataOffset]     = gID;
            edgeData[edgeDataOffset + 1] = edgeDest;

            if (std::is_void<EdgeDataTy>::value) {
              edgeDataOffset += 2;
            } else {
              edgeData[edgeDataOffset + 2] = bufGraph.edgeData(i);
              edgeDataOffset += 3;
            }
          }
        },
        galois::steal(), galois::loopname("LoadEdgesBufferedGraph"));
  }

  return edgeData;
}

/**
 * Load a Galois binary graph into an BufferedGraph and load assigned
 * nodes/edges into memory such that srcs become dests and dests become srcs
 * (i.e. transpose graph).
 *
 * @tparam EdgeDataTy type of edge data to read
 * @param inputFile path to input Galois binary graph
 * @param nodesToRead a pair that has the range of nodes that should be read
 * @param edgesToRead a pair that has the range of edges that should be read
 * @param totalNumNodes Total number of nodes in the graph
 * @param totalNumEdges Total number of edges in the graph
 * @returns a vector with transposed edges corresponding to the nodes/edges
 * pass into the function
 */
template <typename EdgeDataTy>
std::vector<uint32_t> loadTransposedEdgesFromBufferedGraph(
    const std::string& inputFile, Uint64Pair nodesToRead,
    Uint64Pair edgesToRead, uint64_t totalNumNodes, uint64_t totalNumEdges) {
  galois::graphs::BufferedGraph<EdgeDataTy> bufGraph;
  bufGraph.loadPartialGraph(inputFile, nodesToRead.first, nodesToRead.second,
                            edgesToRead.first, edgesToRead.second,
                            totalNumNodes, totalNumEdges);

  std::vector<uint32_t> edgeData;

  // void = 2 elements per edge; non-void = 3 elements per edge
  if (std::is_void<EdgeDataTy>::value) {
    edgeData.resize((edgesToRead.second - edgesToRead.first) * 2);
  } else {
    edgeData.resize((edgesToRead.second - edgesToRead.first) * 3);
  }

  if (edgeData.size() > 0) {
    galois::do_all(
        galois::iterate(nodesToRead.first, nodesToRead.second),
        [&](uint32_t gID) {
          uint64_t edgeBegin = *bufGraph.edgeBegin(gID);
          uint64_t edgeEnd   = *bufGraph.edgeEnd(gID);

          // offset into which we should start writing data in
          // edgeData
          uint64_t edgeDataOffset;
          if (std::is_void<EdgeDataTy>::value) {
            edgeDataOffset = (edgeBegin - edgesToRead.first) * 2;
          } else {
            edgeDataOffset = (edgeBegin - edgesToRead.first) * 3;
          }

          // loop through all edges
          for (uint64_t i = edgeBegin; i < edgeEnd; i++) {
            uint32_t edgeSource = bufGraph.edgeDestination(i);
            // src is saved as dest and dest is saved as source
            // (transpose)
            edgeData[edgeDataOffset]     = edgeSource;
            edgeData[edgeDataOffset + 1] = gID;

            if (std::is_void<EdgeDataTy>::value) {
              edgeDataOffset += 2;
            } else {
              edgeData[edgeDataOffset + 2] = bufGraph.edgeData(i);
              edgeDataOffset += 3;
            }
          }
        },
        galois::steal(), galois::loopname("LoadTransposeEdgesBufferedGraph"));
  }

  return edgeData;
}

/**
 * Load a Galois binary graph into an BufferedGraph and load assigned
 * nodes/edges into memory such that each edge is loaded twice (extra in reverse
 * direction).
 *
 * @tparam EdgeDataTy type of edge data to read
 * @param inputFile path to input Galois binary graph
 * @param nodesToRead a pair that has the range of nodes that should be read
 * @param edgesToRead a pair that has the range of edges that should be read
 * @param totalNumNodes Total number of nodes in the graph
 * @param totalNumEdges Total number of edges in the graph
 * @returns a vector with edges corresponding to the nodes/edges
 * passed into the function; 1 edge in original becomes 2
 */
template <typename EdgeDataTy>
std::vector<uint32_t> loadSymmetricEdgesFromBufferedGraph(
    const std::string& inputFile, Uint64Pair nodesToRead,
    Uint64Pair edgesToRead, uint64_t totalNumNodes, uint64_t totalNumEdges) {
  // TODO change this
  galois::graphs::BufferedGraph<EdgeDataTy> bufGraph;
  bufGraph.loadPartialGraph(inputFile, nodesToRead.first, nodesToRead.second,
                            edgesToRead.first, edgesToRead.second,
                            totalNumNodes, totalNumEdges);

  std::vector<uint32_t> edgeData;

  // void = 2 elements per edge; non-void = 3 elements per edge
  if (std::is_void<EdgeDataTy>::value) {
    edgeData.resize(((edgesToRead.second - edgesToRead.first) * 2) * 2);
  } else {
    edgeData.resize(((edgesToRead.second - edgesToRead.first) * 3) * 2);
  }

  if (edgeData.size() > 0) {
    galois::do_all(
        galois::iterate(nodesToRead.first, nodesToRead.second),
        [&](uint32_t gID) {
          uint64_t edgeBegin = *bufGraph.edgeBegin(gID);
          uint64_t edgeEnd   = *bufGraph.edgeEnd(gID);

          // offset into which we should start writing data in
          // edgeData
          uint64_t edgeDataOffset;
          if (std::is_void<EdgeDataTy>::value) {
            edgeDataOffset = (edgeBegin - edgesToRead.first) * 4;
          } else {
            edgeDataOffset = (edgeBegin - edgesToRead.first) * 6;
          }

          // loop through all edges, create 2 edges for every edge
          for (uint64_t i = edgeBegin; i < edgeEnd; i++) {
            uint32_t edgeDest            = bufGraph.edgeDestination(i);
            edgeData[edgeDataOffset]     = gID;
            edgeData[edgeDataOffset + 1] = edgeDest;

            if (std::is_void<EdgeDataTy>::value) {
              edgeData[edgeDataOffset + 2] = edgeDest;
              edgeData[edgeDataOffset + 3] = gID;
              edgeDataOffset += 4;
            } else {
              uint32_t edgeWeight = bufGraph.edgeData(i);

              edgeData[edgeDataOffset + 2] = edgeWeight;

              edgeData[edgeDataOffset + 3] = edgeDest;
              edgeData[edgeDataOffset + 4] = gID;
              edgeData[edgeDataOffset + 5] = edgeWeight;

              edgeDataOffset += 6;
            }
          }
        },
        galois::steal(), galois::loopname("LoadSymmetricEdgesBufferedGraph"));
  }

  return edgeData;
}

/**
 * Load a Galois binary graph such that multiedges and self loops are ignored.
 * Weights are completely ignored regardless if they exist or not.
 *
 * @param inputFile path to input Galois binary graph
 * @param nodesToRead a pair that has the range of nodes that should be read
 * @param edgesToRead a pair that has the range of edges that should be read
 * @param totalNumNodes Total number of nodes in the graph
 * @param totalNumEdges Total number of edges in the graph
 * @returns a vector with edges corresponding to the nodes/edges
 * passed into the function; multi edges and self loops removed
 */
std::vector<uint32_t>
loadCleanEdgesFromBufferedGraph(const std::string& inputFile,
                                Uint64Pair nodesToRead, Uint64Pair edgesToRead,
                                uint64_t totalNumNodes, uint64_t totalNumEdges,
                                bool keepSelfLoops);

/**
 * Loads the node to new node mapping, then reads the edges that this host
 * has been assigned into a buffer. The catch is that it reads them
 * in a TRANSPOSED manner, and it remaps the original source node to
 * its new node id.
 *
 * i.e. source nodes are remapped to new id, but destination nodes aren't
 * Edges are returned in a destination, source (and optionally edge data)
 * order
 *
 * @tparam EdgeDataTy type of edge data to read
 *
 * @param inputFile path to input Galois binary graph
 * @param nodesToRead a pair that has the range of nodes that should be read
 * @param edgesToRead a pair that has the range of edges that should be read
 * @param totalNumNodes Total number of nodes in the graph
 * @param totalNumEdges Total number of edges in the graph
 * @param mappedBinary binary file with info that maps a node to its new node
 *
 * @returns A vector of transposed edges (with or without edge data
 * depending on edge data type)
 */
template <typename EdgeDataTy>
std::vector<uint32_t> loadMappedSourceEdgesFromBufferedGraph(
    const std::string& inputFile, Uint64Pair nodesToRead,
    Uint64Pair edgesToRead, uint64_t totalNumNodes, uint64_t totalNumEdges,
    const std::string& mappedBinary) {
  galois::graphs::BufferedGraph<EdgeDataTy> bufGraph;
  bufGraph.loadPartialGraph(inputFile, nodesToRead.first, nodesToRead.second,
                            edgesToRead.first, edgesToRead.second,
                            totalNumNodes, totalNumEdges);
  std::vector<uint32_t> edgeData;
  // void = 2 elements per edge; non-void = 3 elements per edge
  if (std::is_void<EdgeDataTy>::value) {
    edgeData.resize((edgesToRead.second - edgesToRead.first) * 2);
  } else {
    edgeData.resize((edgesToRead.second - edgesToRead.first) * 3);
  }

  std::vector<uint32_t> node2NewNode = readRandomNodeMapping(
      mappedBinary, nodesToRead.first, nodesToRead.second - nodesToRead.first);

  if (edgeData.size() > 0) {
    galois::do_all(
        galois::iterate(nodesToRead.first, nodesToRead.second),
        [&](uint32_t gID) {
          uint64_t edgeBegin = *bufGraph.edgeBegin(gID);
          uint64_t edgeEnd   = *bufGraph.edgeEnd(gID);

          // offset into which we should start writing data in
          // edgeData
          uint64_t edgeDataOffset;
          if (std::is_void<EdgeDataTy>::value) {
            edgeDataOffset = (edgeBegin - edgesToRead.first) * 2;
          } else {
            edgeDataOffset = (edgeBegin - edgesToRead.first) * 3;
          }

          uint32_t lID          = gID - nodesToRead.first;
          uint32_t mappedSource = node2NewNode[lID];

          // loop through all edges
          for (uint64_t i = edgeBegin; i < edgeEnd; i++) {
            uint32_t edgeSource = bufGraph.edgeDestination(i);
            // src is saved as dest and dest is saved as source
            // (transpose)
            edgeData[edgeDataOffset]     = edgeSource;
            edgeData[edgeDataOffset + 1] = mappedSource;

            if (std::is_void<EdgeDataTy>::value) {
              edgeDataOffset += 2;
            } else {
              edgeData[edgeDataOffset + 2] = bufGraph.edgeData(i);
              edgeDataOffset += 3;
            }
          }
        },
        galois::steal(), galois::loopname("RemapDestinations"));
  }

  return edgeData;
}

/**
 * Determine/send to each host how many edges they should expect to receive
 * from the caller (i.e. this host).
 *
 * @tparam EdgeDataTy type of edge data to read
 * @param hostToNodes mapping of a host to the nodes it is assigned
 * @param localEdges in-memory buffer of edges this host has loaded
 */
template <typename EdgeDataTy>
void sendEdgeCounts(const std::vector<Uint64Pair>& hostToNodes,
                    const std::vector<uint32_t>& localEdges) {
  auto& net              = galois::runtime::getSystemNetworkInterface();
  uint64_t hostID        = net.ID;
  uint64_t totalNumHosts = net.Num;

  std::cout << "[" << hostID << "] Determinining edge counts\n";

  std::vector<galois::GAccumulator<uint64_t>> numEdgesPerHost(totalNumHosts);

  uint64_t localNumEdges = getNumEdges<EdgeDataTy>(localEdges);
  // determine to which host each edge will go
  galois::do_all(
      galois::iterate((uint64_t)0, localNumEdges),
      [&](uint64_t edgeIndex) {
        uint32_t src;
        if (std::is_void<EdgeDataTy>::value) {
          src = localEdges[edgeIndex * 2];
        } else {
          src = localEdges[edgeIndex * 3];
        }

        uint32_t edgeOwner = findOwner(src, hostToNodes);
        numEdgesPerHost[edgeOwner] += 1;
      },
      galois::loopname("EdgeInspection"));

  std::cout << "[" << hostID << "] Sending edge counts\n";

  for (unsigned h = 0; h < totalNumHosts; h++) {
    if (h == hostID)
      continue;
    galois::runtime::SendBuffer b;
    galois::runtime::gSerialize(b, numEdgesPerHost[h].reduce());
    net.sendTagged(h, galois::runtime::evilPhase, b);
  }
};

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
 * @param localSrcToData Vector of vectors: the vector at index i specifies
 * the data of edges owned by local node i
 * @param nodeLocks Vector of mutexes (one for each local node) that are used
 * when writing to the local mapping of sources to destinations since vectors
 * are not thread safe
 */
// TODO make implementation smaller/cleaner i.e. refactor
// TODO merge with the non void version below because the code duplication
// here is ugly and messy
template <
    typename EdgeDataTy,
    typename std::enable_if<std::is_void<EdgeDataTy>::value>::type* = nullptr>
void sendAssignedEdges(const std::vector<Uint64Pair>& hostToNodes,
                       const std::vector<uint32_t>& localEdges,
                       std::vector<std::vector<uint32_t>>& localSrcToDest,
                       std::vector<std::vector<uint32_t>>&,
                       std::vector<std::mutex>& nodeLocks) {
  auto& net              = galois::runtime::getSystemNetworkInterface();
  uint64_t hostID        = net.ID;
  uint64_t totalNumHosts = net.Num;

  std::cout << "[" << hostID << "] Going to send assigned edges\n";

  using EdgeVectorTy = std::vector<std::vector<uint32_t>>;
  galois::substrate::PerThreadStorage<EdgeVectorTy> dstVectors(totalNumHosts);

  using SendBufferVectorTy = std::vector<galois::runtime::SendBuffer>;
  galois::substrate::PerThreadStorage<SendBufferVectorTy> sendBuffers(
      totalNumHosts);
  galois::substrate::PerThreadStorage<std::vector<uint64_t>>
      lastSourceSentStorage(totalNumHosts);

  // initialize last source sent
  galois::on_each([&](unsigned, unsigned) {
    for (unsigned h = 0; h < totalNumHosts; h++) {
      (*(lastSourceSentStorage.getLocal()))[h] = 0;
    }
  });

  std::cout << "[" << hostID << "] Passing through edges and assigning\n";

  uint64_t localNumEdges = getNumEdges<EdgeDataTy>(localEdges);
  // determine to which host each edge will go
  galois::do_all(
      galois::iterate((uint64_t)0, localNumEdges),
      [&](uint64_t edgeIndex) {
        uint32_t src       = localEdges[edgeIndex * 2];
        uint32_t edgeOwner = findOwner(src, hostToNodes);
        uint32_t dst       = localEdges[(edgeIndex * 2) + 1];
        uint32_t localID   = src - hostToNodes[edgeOwner].first;

        if (edgeOwner != hostID) {
          // send off to correct host
          auto& hostSendBuffer = (*(sendBuffers.getLocal()))[edgeOwner];
          auto& dstVector      = (*(dstVectors.getLocal()))[edgeOwner];
          auto& lastSourceSent =
              (*(lastSourceSentStorage.getLocal()))[edgeOwner];

          if (lastSourceSent == localID) {
            dstVector.emplace_back(dst);
          } else {
            // serialize vector if anything exists in it + send
            // buffer if reached some limit
            if (dstVector.size() > 0) {
              uint64_t globalSourceID =
                  lastSourceSent + hostToNodes[edgeOwner].first;
              galois::runtime::gSerialize(hostSendBuffer, globalSourceID,
                                          dstVector);
              dstVector.clear();
              if (hostSendBuffer.size() > 1400) {
                net.sendTagged(edgeOwner, galois::runtime::evilPhase,
                               hostSendBuffer);
                hostSendBuffer.getVec().clear();
              }
            }

            dstVector.emplace_back(dst);
            lastSourceSent = localID;
          }
        } else {
          // save to edge dest array
          nodeLocks[localID].lock();
          localSrcToDest[localID].emplace_back(dst);
          nodeLocks[localID].unlock();
        }
      },
      galois::loopname("Pass2"));

  std::cout << "[" << hostID << "] Buffer cleanup\n";

  // cleanup: each thread serialize + send out remaining stuff
  galois::on_each(
      [&](unsigned, unsigned) {
        for (unsigned h = 0; h < totalNumHosts; h++) {
          if (h == hostID)
            continue;
          auto& hostSendBuffer    = (*(sendBuffers.getLocal()))[h];
          auto& dstVector         = (*(dstVectors.getLocal()))[h];
          uint64_t lastSourceSent = (*(lastSourceSentStorage.getLocal()))[h];

          if (dstVector.size() > 0) {
            uint64_t globalSourceID = lastSourceSent + hostToNodes[h].first;
            galois::runtime::gSerialize(hostSendBuffer, globalSourceID,
                                        dstVector);
            dstVector.clear();
          }

          if (hostSendBuffer.size() > 0) {
            net.sendTagged(h, galois::runtime::evilPhase, hostSendBuffer);
            hostSendBuffer.getVec().clear();
          }
        }
      },
      galois::loopname("Pass2Cleanup"));
}

// Non-void variant of the above; uint32_t only
template <
    typename EdgeDataTy,
    typename std::enable_if<!std::is_void<EdgeDataTy>::value>::type* = nullptr>
void sendAssignedEdges(const std::vector<Uint64Pair>& hostToNodes,
                       const std::vector<uint32_t>& localEdges,
                       std::vector<std::vector<uint32_t>>& localSrcToDest,
                       std::vector<std::vector<uint32_t>>& localSrcToData,
                       std::vector<std::mutex>& nodeLocks) {
  auto& net              = galois::runtime::getSystemNetworkInterface();
  uint64_t hostID        = net.ID;
  uint64_t totalNumHosts = net.Num;

  std::cout << "[" << hostID << "] Going to send assigned edges\n";

  // initialize localsrctodata
  GALOIS_ASSERT(localSrcToData.empty());
  using EdgeVectorTy = std::vector<std::vector<uint32_t>>;
  EdgeVectorTy tmp   = EdgeVectorTy(localSrcToDest.size());
  localSrcToData.swap(tmp);
  GALOIS_ASSERT(localSrcToData.size() == localSrcToDest.size());

  galois::substrate::PerThreadStorage<EdgeVectorTy> dstVectors(totalNumHosts);
  // currently only uint32_t support for edge data
  galois::substrate::PerThreadStorage<EdgeVectorTy> dataVectors(totalNumHosts);

  using SendBufferVectorTy = std::vector<galois::runtime::SendBuffer>;
  galois::substrate::PerThreadStorage<SendBufferVectorTy> sendBuffers(
      totalNumHosts);
  galois::substrate::PerThreadStorage<std::vector<uint64_t>>
      lastSourceSentStorage(totalNumHosts);

  // initialize last source sent
  galois::on_each([&](unsigned, unsigned) {
    for (unsigned h = 0; h < totalNumHosts; h++) {
      (*(lastSourceSentStorage.getLocal()))[h] = 0;
    }
  });

  std::cout << "[" << hostID << "] Passing through edges and assigning\n";

  uint64_t localNumEdges = getNumEdges<EdgeDataTy>(localEdges);
  // determine to which host each edge will go
  galois::do_all(
      galois::iterate((uint64_t)0, localNumEdges),
      [&](uint64_t edgeIndex) {
        uint32_t src       = localEdges[edgeIndex * 3];
        uint32_t edgeOwner = findOwner(src, hostToNodes);
        uint32_t dst       = localEdges[(edgeIndex * 3) + 1];
        uint32_t localID   = src - hostToNodes[edgeOwner].first;
        uint32_t edgeData  = localEdges[(edgeIndex * 3) + 2];

        if (edgeOwner != hostID) {
          // send off to correct host
          auto& hostSendBuffer = (*(sendBuffers.getLocal()))[edgeOwner];
          auto& dstVector      = (*(dstVectors.getLocal()))[edgeOwner];
          auto& dataVector     = (*(dataVectors.getLocal()))[edgeOwner];
          auto& lastSourceSent =
              (*(lastSourceSentStorage.getLocal()))[edgeOwner];

          if (lastSourceSent == localID) {
            dstVector.emplace_back(dst);
            dataVector.emplace_back(edgeData);
          } else {
            // serialize vector if anything exists in it + send buffer if
            // reached some limit
            if (dstVector.size() > 0) {
              uint64_t globalSourceID =
                  lastSourceSent + hostToNodes[edgeOwner].first;
              galois::runtime::gSerialize(hostSendBuffer, globalSourceID,
                                          dstVector, dataVector);
              dstVector.clear();
              dataVector.clear();
              if (hostSendBuffer.size() > 1400) {
                net.sendTagged(edgeOwner, galois::runtime::evilPhase,
                               hostSendBuffer);
                hostSendBuffer.getVec().clear();
              }
            }

            dstVector.emplace_back(dst);
            dataVector.emplace_back(edgeData);
            lastSourceSent = localID;
          }
        } else {
          // save to edge dest array
          nodeLocks[localID].lock();
          localSrcToDest[localID].emplace_back(dst);
          localSrcToData[localID].emplace_back(edgeData);
          nodeLocks[localID].unlock();
        }
      },
      galois::loopname("Pass2"));

  std::cout << "[" << hostID << "] Buffer cleanup\n";

  // cleanup: each thread serialize + send out remaining stuff
  galois::on_each(
      [&](unsigned, unsigned) {
        for (unsigned h = 0; h < totalNumHosts; h++) {
          if (h == hostID)
            continue;
          auto& hostSendBuffer    = (*(sendBuffers.getLocal()))[h];
          auto& dstVector         = (*(dstVectors.getLocal()))[h];
          auto& dataVector        = (*(dataVectors.getLocal()))[h];
          uint64_t lastSourceSent = (*(lastSourceSentStorage.getLocal()))[h];

          if (dstVector.size() > 0) {
            uint64_t globalSourceID = lastSourceSent + hostToNodes[h].first;
            galois::runtime::gSerialize(hostSendBuffer, globalSourceID,
                                        dstVector, dataVector);
            dstVector.clear();
            dataVector.clear();
          }

          if (hostSendBuffer.size() > 0) {
            net.sendTagged(h, galois::runtime::evilPhase, hostSendBuffer);
            hostSendBuffer.getVec().clear();
          }
        }
      },
      galois::loopname("Pass2Cleanup"));
}

/**
 * Receive this host's assigned edges: should be called after sendAssignedEdges.
 *
 * @param edgesToReceive the number of edges we expect to receive; the function
 * will not exit until all expected edges are received
 * @param hostToNodes mapping of a host to the nodes it is assigned
 * @param localSrcToDest local mapping of LOCAL sources to destinations (we
 * may have some edges that do not need sending; they are saved here)
 * @param localSrcToData Vector of vectors: the vector at index i specifies
 * the data of edges owned by local node i; NOTE THAT THIS VECTOR BEING EMPTY
 * OR NON EMPTY DETERMINES IF THE FUNCTION EXPECTS TO RECEIVE EDGE DATA
 * @param nodeLocks Vector of mutexes (one for each local node) that are used
 * when writing to the local mapping of sources to destinations since vectors
 * are not thread safe
 */
void receiveAssignedEdges(std::atomic<uint64_t>& edgesToReceive,
                          const std::vector<Uint64Pair>& hostToNodes,
                          std::vector<std::vector<uint32_t>>& localSrcToDest,
                          std::vector<std::vector<uint32_t>>& localSrcToData,
                          std::vector<std::mutex>& nodeLocks);

/**
 * Send/receive other hosts number of assigned edges.
 *
 * @param localAssignedEdges number of edges assigned to this host
 * @returns a vector that has every hosts number of locally assigned edges
 */
std::vector<uint64_t> getEdgesPerHost(uint64_t localAssignedEdges);

/**
 * Given a vector of vectors, "flatten" it by merging them into 1 vector
 * in the order they appear the in the vector.
 *
 * @param vectorOfVectors vector of vectors to flatten. FUNCTION WILL ERASE
 * ALL DATA IN THE VECTOR.
 * @returns a flattened vector from vectorOfVectors
 */
std::vector<uint32_t>
flattenVectors(std::vector<std::vector<uint32_t>>& vectorOfVectors);

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
                        const std::vector<uint64_t>& edgePrefixSum);

/**
 * Writes the edge destination data of a galois binary graph.
 *
 * @param gr File to write to
 * @param edgeDestOffset offset into file specifying where to start writing
 * @param localSrcToDest Vector of vectors: the vector at index i specifies
 * the destinations for local src node i
 */
void writeEdgeDestData(MPI_File& gr, uint64_t edgeDestOffset,
                       std::vector<std::vector<uint32_t>>& localSrcToDest);

/**
 * Writes the edge destination data of a galois binary graph.
 * @param gr File to write to
 * @param edgeDestOffset offset into file specifying where to start writing
 * @param destVector Vector of edge destinations IN THE ORDER THAT THEY SHOULD
 * BE WRITTEN (i.e. in correct order corresponding to node order this host has)
 */
void writeEdgeDestData(MPI_File& gr, uint64_t edgeDestOffset,
                       std::vector<uint32_t>& destVector);
/**
 * Writes the edge data data of a galois binary graph.
 *
 * @param gr File to write to
 * @param edgeDataOffset offset into file specifying where to start writing
 * @param edgeDataToWrite vector of localNumEdges elements corresponding to
 * edge data that needs to be written
 */
void writeEdgeDataData(MPI_File& gr, uint64_t edgeDataOffset,
                       const std::vector<uint32_t>& edgeDataToWrite);

/**
 * Write graph data out to a V1 Galois binary graph file.
 *
 * @param outputFile name of file to write to
 * @param totalNumNodes total number of nodes in the graph
 * @param totalNumEdges total number of edges in graph
 * @param localNumNodes number of source nodes that this host was assigned to
 * write
 * @param localNodeBegin global id of first node this host was assigned
 * @param globalEdgeOffset number of edges to skip to get to the first edge
 * this host is responsible for
 * @param localSrcToDest Vector of vectors: the vector at index i specifies
 * the destinations of edges owned by local node i
 * @param localSrcToData Vector of vectors: the vector at index i specifies
 * the data of edges owned by local node i
 */
void writeToGr(const std::string& outputFile, uint64_t totalNumNodes,
               uint64_t totalNumEdges, uint64_t localNumNodes,
               uint64_t localNodeBegin, uint64_t globalEdgeOffset,
               std::vector<std::vector<uint32_t>>& localSrcToDest,
               std::vector<std::vector<uint32_t>>& localSrcToData);

/**
 * Write graph data out to a Lux binary graph file.
 *
 * @param outputFile name of file to write to
 * @param totalNumNodes total number of nodes in the graph
 * @param totalNumEdges total number of edges in graph
 * @param localNumNodes number of source nodes that this host was assigned to
 * write
 * @param localNodeBegin global id of first node this host was assigned
 * @param globalEdgeOffset number of edges to skip to get to the first edge
 * this host is responsible for
 * @param localSrcToDest Vector of vectors: the vector at index i specifies
 * the destinations of edges owned by local node i
 * @param localSrcToData Vector of vectors: the vector at index i specifies
 * the data of edges owned by local node i
 */
void writeToLux(const std::string& outputFile, uint64_t totalNumNodes,
                uint64_t totalNumEdges, uint64_t localNumNodes,
                uint64_t localNodeBegin, uint64_t globalEdgeOffset,
                std::vector<std::vector<uint32_t>>& localSrcToDest,
                std::vector<std::vector<uint32_t>>& localSrcToData);

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
Uint64Pair getLocalAssignment(uint64_t numToSplit);

/**
 * Given a host to node assignment, send the edges we have to the appropriate
 * place + receieve edges sent by other hosts.
 *
 * @param hostToNodes Vector specifying assignment of nodes to hosts
 * @param localEdges array that represents edges on this host (to keep or to
 * send)
 * @returns 2 structures: one is a vector of vectors where the vector at
 * index i has destination ids for local node i and another vector of vectors
 * similar to the former except with edge weights (if EdgeTy is non-void)
 */
template <typename EdgeTy>
PairVoVUint32
sendAndReceiveAssignedEdges(const std::vector<Uint64Pair>& hostToNodes,
                            std::vector<uint32_t>& localEdges) {
  uint32_t hostID = galois::runtime::getSystemNetworkInterface().ID;
  uint64_t localNumNodes =
      hostToNodes[hostID].second - hostToNodes[hostID].first;

  sendEdgeCounts<EdgeTy>(hostToNodes, localEdges);
  std::atomic<uint64_t> edgesToReceive;
  edgesToReceive.store(receiveEdgeCounts());

  std::cout << "[" << hostID << "] Need to receive " << edgesToReceive.load()
            << " edges\n";

  // FIXME ONLY V1 SUPPORT
  VoVUint32 localSrcToDest(localNumNodes);
  VoVUint32 localSrcToData;
  std::vector<std::mutex> nodeLocks(localNumNodes);

  sendAssignedEdges<EdgeTy>(hostToNodes, localEdges, localSrcToDest,
                            localSrcToData, nodeLocks);
  freeVector(localEdges);
  receiveAssignedEdges(edgesToReceive, hostToNodes, localSrcToDest,
                       localSrcToData, nodeLocks);
  return PairVoVUint32(localSrcToDest, localSrcToData);
}

/**
 * Given a set of disjoint edges, assign/send edges to hosts. Then, each host
 * writes the edges to the specified output file in the Galois binary graph
 * format.
 *
 * @param localEdges Array of edges this host has
 * @param totalNumNodes total number of nodes in entire graph
 * @param totalNumEdges total number of edges in entire graph
 * @param outputFile file to write new graph to
 */
template <typename EdgeTy>
void assignAndWriteEdges(std::vector<uint32_t>& localEdges,
                         uint64_t totalNumNodes, uint64_t totalNumEdges,
                         const std::string& outputFile) {
  uint32_t hostID = galois::runtime::getSystemNetworkInterface().ID;

  std::vector<Uint64Pair> hostToNodes = getEvenNodeToHostMapping<EdgeTy>(
      localEdges, totalNumNodes, totalNumEdges);

  PairVoVUint32 receivedEdgeInfo =
      sendAndReceiveAssignedEdges<EdgeTy>(hostToNodes, localEdges);
  VoVUint32 localSrcToDest = receivedEdgeInfo.first;
  VoVUint32 localSrcToData = receivedEdgeInfo.second;

  uint64_t localNodeBegin = hostToNodes[hostID].first;
  uint64_t localNumNodes  = hostToNodes[hostID].second - localNodeBegin;
  freeVector(hostToNodes);

  uint64_t totalAssignedEdges = 0;
  for (unsigned i = 0; i < localNumNodes; i++) {
    totalAssignedEdges += localSrcToDest[i].size();
  }

  std::cout << "[" << hostID << "] Will write " << totalAssignedEdges
            << " edges\n";

  // calculate global edge offset using edge counts from other hosts
  std::vector<uint64_t> edgesPerHost = getEdgesPerHost(totalAssignedEdges);
  uint64_t globalEdgeOffset          = 0;
  uint64_t totalEdgeCount            = 0;
  for (unsigned h = 0; h < hostID; h++) {
    globalEdgeOffset += edgesPerHost[h];
    totalEdgeCount += edgesPerHost[h];
  }

  uint64_t totalNumHosts = galois::runtime::getSystemNetworkInterface().Num;
  // finish off getting total edge count (note this is more of a sanity check
  // since we got total edge count near the beginning already)
  for (unsigned h = hostID; h < totalNumHosts; h++) {
    totalEdgeCount += edgesPerHost[h];
  }
  GALOIS_ASSERT(totalNumEdges == totalEdgeCount);
  freeVector(edgesPerHost);

  writeToGr(outputFile, totalNumNodes, totalEdgeCount, localNumNodes,
            localNodeBegin, globalEdgeOffset, localSrcToDest, localSrcToData);
}

/**
 * Given a set of disjoint edges, assign/send edges to hosts. Then, each host
 * writes the edges to the specified output file in the Lux binary graph
 * format.
 *
 * @todo merge this with the other assignAndWriteEdges to prevent code
 * duplication
 *
 * @param localEdges Array of edges this host has
 * @param totalNumNodes total number of nodes in entire graph
 * @param totalNumEdges total number of edges in entire graph
 * @param outputFile file to write new graph to
 */
template <typename EdgeTy>
void assignAndWriteEdgesLux(std::vector<uint32_t>& localEdges,
                            uint64_t totalNumNodes, uint64_t totalNumEdges,
                            const std::string& outputFile) {
  uint32_t hostID = galois::runtime::getSystemNetworkInterface().ID;

  std::vector<Uint64Pair> hostToNodes = getEvenNodeToHostMapping<EdgeTy>(
      localEdges, totalNumNodes, totalNumEdges);

  PairVoVUint32 receivedEdgeInfo =
      sendAndReceiveAssignedEdges<EdgeTy>(hostToNodes, localEdges);
  VoVUint32 localSrcToDest = receivedEdgeInfo.first;
  VoVUint32 localSrcToData = receivedEdgeInfo.second;

  uint64_t localNodeBegin = hostToNodes[hostID].first;
  uint64_t localNumNodes  = hostToNodes[hostID].second - localNodeBegin;
  freeVector(hostToNodes);

  uint64_t totalAssignedEdges = 0;
  for (unsigned i = 0; i < localNumNodes; i++) {
    totalAssignedEdges += localSrcToDest[i].size();
  }

  std::cout << "[" << hostID << "] Will write " << totalAssignedEdges
            << " edges\n";

  // calculate global edge offset using edge counts from other hosts
  std::vector<uint64_t> edgesPerHost = getEdgesPerHost(totalAssignedEdges);
  uint64_t globalEdgeOffset          = 0;
  uint64_t totalEdgeCount            = 0;
  for (unsigned h = 0; h < hostID; h++) {
    globalEdgeOffset += edgesPerHost[h];
    totalEdgeCount += edgesPerHost[h];
  }

  uint64_t totalNumHosts = galois::runtime::getSystemNetworkInterface().Num;
  // finish off getting total edge count (note this is more of a sanity check
  // since we got total edge count near the beginning already)
  for (unsigned h = hostID; h < totalNumHosts; h++) {
    totalEdgeCount += edgesPerHost[h];
  }
  GALOIS_ASSERT(totalNumEdges == totalEdgeCount);
  freeVector(edgesPerHost);

  writeToLux(outputFile, totalNumNodes, totalEdgeCount, localNumNodes,
             localNodeBegin, globalEdgeOffset, localSrcToDest, localSrcToData);
}
#endif
