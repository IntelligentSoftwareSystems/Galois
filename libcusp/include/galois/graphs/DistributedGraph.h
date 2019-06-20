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
 * @file DistributedGraph.h
 *
 * Contains the implementation for DistGraph. Command line argument definitions
 * are found in DistributedGraph.cpp.
 */

#ifndef _GALOIS_DIST_HGRAPH_H_
#define _GALOIS_DIST_HGRAPH_H_

#include <unordered_map>
#include <fstream>

#include "galois/graphs/LC_CSR_Graph.h"
#include "galois/graphs/BufferedGraph.h"
#include "galois/runtime/DistStats.h"
#include "galois/graphs/OfflineGraph.h"
#include "galois/DynamicBitset.h"
#include "llvm/Support/CommandLine.h"

/*
 * Headers for boost serialization
 */
//#include <boost/archive/binary_oarchive.hpp>
//#include <boost/archive/binary_iarchive.hpp>
//#include <boost/serialization/split_member.hpp>
//#include <boost/serialization/binary_object.hpp>
//#include <boost/serialization/serialization.hpp>
//#include <boost/serialization/vector.hpp>
//#include <boost/serialization/unordered_map.hpp>

namespace galois {
namespace graphs {
/**
 * Enums specifying how masters are to be distributed among hosts.
 */
enum MASTERS_DISTRIBUTION {
  //! balance nodes
  BALANCED_MASTERS,
  //! balance edges
  BALANCED_EDGES_OF_MASTERS,
  //! balance nodes and edges
  BALANCED_MASTERS_AND_EDGES
};

/**
 * Base DistGraph class that all distributed graphs extend from.
 *
 * @tparam NodeTy type of node data for the graph
 * @tparam EdgeTy type of edge data for the graph
 */
template <typename NodeTy, typename EdgeTy>
class DistGraph {
private:
  //! Graph name used for printing things
  constexpr static const char* const GRNAME = "dGraph";

  using GraphTy = galois::graphs::LC_CSR_Graph<NodeTy, EdgeTy, true>;

protected:
  //! The internal graph used by DistGraph to represent the graph
  GraphTy graph;

  //! Marks if the graph is transposed or not.
  bool transposed;

  // global graph variables
  uint64_t numGlobalNodes; //!< Total nodes in the global unpartitioned graph.
  uint64_t numGlobalEdges; //!< Total edges in the global unpartitioned graph.
  uint32_t numNodes; //!< Num nodes in this graph in total
  uint64_t numEdges; //!< Num edges in this graph in total

  const unsigned id; //!< ID of the machine.
  const uint32_t numHosts; //!< Total number of machines

  // local graph
  // size() = Number of nodes created on this host (masters + mirrors)
  uint32_t numOwned;    //!< Number of nodes owned (masters) by this host.
                        //!< size() - numOwned = mirrors on this host
  uint32_t beginMaster; //!< Local id of the beginning of master nodes.
                        //!< beginMaster + numOwned = local id of the end of
                        //!< master nodes
  uint32_t numNodesWithEdges; //!< Number of nodes (masters + mirrors) that have
                              //!< outgoing edges

  //! Information that converts host to range of nodes that host reads
  std::vector<std::pair<uint64_t, uint64_t>> gid2host;
  //! Mirror nodes from different hosts. For reduce
  std::vector<std::vector<size_t>> mirrorNodes;

  //! GID = localToGlobalVector[LID]
  std::vector<uint64_t> localToGlobalVector;
  //! LID = globalToLocalMap[GID]
  std::unordered_map<uint64_t, uint32_t> globalToLocalMap;


private:
  // vector for determining range objects for master nodes + nodes
  // with edges (which includes masters)
  //! represents split of all nodes among threads to balance edges
  std::vector<uint32_t> allNodesRanges;
  //! represents split of master nodes among threads to balance edges
  std::vector<uint32_t> masterRanges;
  //! represents split of nodes with edges (includes masters) among threads to
  //! balance edges
  std::vector<uint32_t> withEdgeRanges;
  //! represents split of all nodes among threads to balance in-edges
  std::vector<uint32_t> allNodesRangesIn;
  //! represents split of master nodes among threads to balance in-edges
  std::vector<uint32_t> masterRangesIn;

  using NodeRangeType =
      galois::runtime::SpecificRange<boost::counting_iterator<size_t>>;

  //! Vector of ranges that stores the 3 different range objects that a user is
  //! able to access
  std::vector<NodeRangeType> specificRanges;
  //! Like specificRanges, but for in edges
  std::vector<NodeRangeType> specificRangesIn;

protected:
  //! Prints graph statistics.
  void printStatistics() {
    if (id == 0) {
      galois::gPrint("Total nodes: ", numGlobalNodes, "\n");
      galois::gPrint("Total edges: ", numGlobalEdges, "\n");
    }
    galois::gPrint("[", id, "] Master nodes: ", numOwned, "\n");
    galois::gPrint("[", id, "] Mirror nodes: ", size() - numOwned, "\n");
    galois::gPrint("[", id, "] Nodes with edges: ", numNodesWithEdges, "\n");
    galois::gPrint("[", id, "] Edges: ", sizeEdges(), "\n");

    // reports the number of nodes + edge as well
    galois::runtime::reportStatCond_Tsum<MORE_DIST_STATS>(
        "dGraph", "TotalNodes", numOwned);
    galois::runtime::reportStatCond_Tsum<MORE_DIST_STATS>(
        "dGraph", "TotalEdges", sizeEdges());
  }

  //! Increments evilPhase, a phase counter used by communication.
  void inline increment_evilPhase() {
    ++galois::runtime::evilPhase;
    if (galois::runtime::evilPhase >=
        std::numeric_limits<int16_t>::max()) { // limit defined by MPI or LCI
      galois::runtime::evilPhase = 1;
    }
  }

  //! Returns evilPhase + 1, handling loop around as necessary
  unsigned inline evilPhasePlus1() {
    unsigned result = galois::runtime::evilPhase + 1;

    // limit defined by MPI or LCI
    if (result >= std::numeric_limits<int16_t>::max()) {
      return 1;
    } else {
      return result;
    }
  }

private:
  /**
   * Given an OfflineGraph, compute the masters for each node by
   * evenly (or unevenly as specified by scale factor)
   * blocking the nodes off to assign to each host. Considers
   * ONLY nodes and not edges.
   *
   * @param g The offline graph which has loaded the graph you want
   * to get the masters for
   * @param numNodes_to_divide The total number of nodes you are
   * assigning to different hosts
   * @param scalefactor A vector that specifies if a particular host
   * should have more or less than other hosts
   * @param DecomposeFactor Specifies how decomposed the blocking
   * of nodes should be. For example, a factor of 2 will make 2 blocks
   * out of 1 block had the decompose factor been set to 1.
   */
  void computeMastersBlockedNodes(galois::graphs::OfflineGraph& g,
                                  uint64_t numNodes_to_divide,
                                  const std::vector<unsigned>& scalefactor,
                                  unsigned DecomposeFactor = 1) {
    if (scalefactor.empty() || (numHosts * DecomposeFactor == 1)) {
      for (unsigned i = 0; i < numHosts * DecomposeFactor; ++i)
        gid2host.push_back(galois::block_range(0U, (unsigned)numNodes_to_divide,
                                               i, numHosts * DecomposeFactor));
    } else { // TODO: not compatible with DecomposeFactor.
      assert(scalefactor.size() == numHosts);

      unsigned numBlocks = 0;

      for (unsigned i = 0; i < numHosts; ++i) {
        numBlocks += scalefactor[i];
      }

      std::vector<std::pair<uint64_t, uint64_t>> blocks;
      for (unsigned i = 0; i < numBlocks; ++i) {
        blocks.push_back(galois::block_range(0U, (unsigned)numNodes_to_divide,
                                             i, numBlocks));
      }

      std::vector<unsigned> prefixSums;
      prefixSums.push_back(0);

      for (unsigned i = 1; i < numHosts; ++i) {
        prefixSums.push_back(prefixSums[i - 1] + scalefactor[i - 1]);
      }

      for (unsigned i = 0; i < numHosts; ++i) {
        unsigned firstBlock = prefixSums[i];
        unsigned lastBlock  = prefixSums[i] + scalefactor[i] - 1;
        gid2host.push_back(
            std::make_pair(blocks[firstBlock].first, blocks[lastBlock].second));
      }
    }
  }

  /**
   * Given an OfflineGraph, compute the masters for each node by
   * evenly (or unevenly as specified by scale factor)
   * blocking the nodes off to assign to each host while taking
   * into consideration the only edges of the node to get
   * even blocks.
   *
   * @param g The offline graph which has loaded the graph you want
   * to get the masters for
   * @param numNodes_to_divide The total number of nodes you are
   * assigning to different hosts
   * @param scalefactor A vector that specifies if a particular host
   * should have more or less than other hosts
   * @param DecomposeFactor Specifies how decomposed the blocking
   * of nodes should be. For example, a factor of 2 will make 2 blocks
   * out of 1 block had the decompose factor been set to 1.
   */
  void computeMastersBalancedEdges(galois::graphs::OfflineGraph& g,
                                   uint64_t numNodes_to_divide,
                                   const std::vector<unsigned>& scalefactor,
                                   uint32_t edgeWeight,
                                   unsigned DecomposeFactor = 1) {
    if (edgeWeight == 0) {
      edgeWeight = 1;
    }

    auto& net = galois::runtime::getSystemNetworkInterface();

    gid2host.resize(numHosts * DecomposeFactor);
    for (unsigned d = 0; d < DecomposeFactor; ++d) {
      auto r = g.divideByNode(0, edgeWeight, (id + d * numHosts),
                              numHosts * DecomposeFactor, scalefactor);
      gid2host[id + d * numHosts].first  = *(r.first.first);
      gid2host[id + d * numHosts].second = *(r.first.second);
    }

    for (unsigned h = 0; h < numHosts; ++h) {
      if (h == id)
        continue;
      galois::runtime::SendBuffer b;
      for (unsigned d = 0; d < DecomposeFactor; ++d) {
        galois::runtime::gSerialize(b, gid2host[id + d * numHosts]);
      }
      net.sendTagged(h, galois::runtime::evilPhase, b);
    }
    net.flush();
    unsigned received = 1;
    while (received < numHosts) {
      decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;
      do {
        p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
      } while (!p);
      assert(p->first != id);
      auto& b = p->second;
      for (unsigned d = 0; d < DecomposeFactor; ++d) {
        galois::runtime::gDeserialize(b, gid2host[p->first + d * numHosts]);
      }
      ++received;
    }
    increment_evilPhase();

    #ifndef NDEBUG
    for (unsigned h = 0; h < numHosts; h++) {
      if (h == 0) {
        assert(gid2host[h].first == 0);
      } else if (h == numHosts - 1) {
        assert(gid2host[h].first == gid2host[h - 1].second);
        assert(gid2host[h].second == g.size());
      } else {
        assert(gid2host[h].first == gid2host[h - 1].second);
        assert(gid2host[h].second == gid2host[h + 1].first);
      }
    }
    #endif
  }

  /**
   * Given an OfflineGraph, compute the masters for each node by
   * evenly (or unevenly as specified by scale factor)
   * blocking the nodes off to assign to each host while taking
   * into consideration the edges of the node AND the node itself.
   *
   * @param g The offline graph which has loaded the graph you want
   * to get the masters for
   * @param numNodes_to_divide The total number of nodes you are
   * assigning to different hosts
   * @param scalefactor A vector that specifies if a particular host
   * should have more or less than other hosts
   * @param DecomposeFactor Specifies how decomposed the blocking
   * of nodes should be. For example, a factor of 2 will make 2 blocks
   * out of 1 block had the decompose factor been set to 1. Ignored
   * in this function currently.
   *
   * @todo make this function work with decompose factor
   */
  void computeMastersBalancedNodesAndEdges(
      galois::graphs::OfflineGraph& g, uint64_t numNodes_to_divide,
      const std::vector<unsigned>& scalefactor, uint32_t nodeWeight,
      uint32_t edgeWeight, unsigned DecomposeFactor = 1) {
    if (nodeWeight == 0) {
      nodeWeight = g.sizeEdges() / g.size(); // average degree
    }
    if (edgeWeight == 0) {
      edgeWeight = 1;
    }

    auto& net = galois::runtime::getSystemNetworkInterface();
    gid2host.resize(numHosts);
    auto r = g.divideByNode(nodeWeight, edgeWeight, id,
                            numHosts, scalefactor);
    gid2host[id].first  = *r.first.first;
    gid2host[id].second = *r.first.second;
    for (unsigned h = 0; h < numHosts; ++h) {
      if (h == id)
        continue;
      galois::runtime::SendBuffer b;
      galois::runtime::gSerialize(b, gid2host[id]);
      net.sendTagged(h, galois::runtime::evilPhase, b);
    }
    net.flush();
    unsigned received = 1;
    while (received < numHosts) {
      decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;
      do {
        p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
      } while (!p);
      assert(p->first != id);
      auto& b = p->second;
      galois::runtime::gDeserialize(b, gid2host[p->first]);
      ++received;
    }
    increment_evilPhase();
  }

protected:
  /**
   * Wrapper call that will call into more specific compute masters
   * functions that compute masters based on nodes, edges, or both.
   *
   * @param g The offline graph which has loaded the graph you want
   * to get the masters for
   * @param scalefactor A vector that specifies if a particular host
   * should have more or less than other hosts
   * @param DecomposeFactor Specifies how decomposed the blocking
   * of nodes should be. For example, a factor of 2 will make 2 blocks
   * out of 1 block had the decompose factor been set to 1.
   */
  uint64_t computeMasters(MASTERS_DISTRIBUTION masters_distribution,
                          galois::graphs::OfflineGraph& g,
                          const std::vector<unsigned>& scalefactor,
                          uint32_t nodeWeight=0, uint32_t edgeWeight=0,
                          unsigned DecomposeFactor = 1) {
    galois::Timer timer;
    timer.start();
    g.reset_seek_counters();

    uint64_t numNodes_to_divide = 0;

    numNodes_to_divide = g.size();

    // compute masters for all nodes
    switch (masters_distribution) {
    case BALANCED_MASTERS:
      computeMastersBlockedNodes(g, numNodes_to_divide, scalefactor,
                                 DecomposeFactor);
      break;
    case BALANCED_MASTERS_AND_EDGES:
      computeMastersBalancedNodesAndEdges(g, numNodes_to_divide, scalefactor,
                                          nodeWeight, edgeWeight,
                                          DecomposeFactor);
      break;
    case BALANCED_EDGES_OF_MASTERS:
    default:
      computeMastersBalancedEdges(g, numNodes_to_divide, scalefactor,
                                  edgeWeight, DecomposeFactor);
      break;
    }

    timer.stop();

    galois::runtime::reportStatCond_Tmax<MORE_DIST_STATS>(
        GRNAME, "MasterDistTime", timer.get());

    galois::gPrint(
        "[", id, "] Master distribution time : ", timer.get_usec() / 1000000.0f,
        " seconds to read ", g.num_bytes_read(), " bytes in ", g.num_seeks(),
        " seeks (", g.num_bytes_read() / (float)timer.get_usec(), " MBPS)\n");
    return numNodes_to_divide;
  }

  uint32_t G2L(uint64_t gid) const {
    assert(isLocal(gid));
    return globalToLocalMap.at(gid);
  }

  uint64_t L2G(uint32_t lid) const {
    return localToGlobalVector[lid];
  }

public:
  //! Type representing a node in this graph
  using GraphNode = typename GraphTy::GraphNode;
  //! Expose EdgeTy to other classes
  using EdgeType = EdgeTy;
  //! iterator type over nodes
  using iterator = typename GraphTy::iterator;
  //! constant iterator type over nodes
  using const_iterator = typename GraphTy::const_iterator;
  //! iterator type over edges
  using edge_iterator = typename GraphTy::edge_iterator;

  /**
   * Constructor for DistGraph. Initializes metadata fields.
   *
   * @param host host number that this graph resides on
   * @param numHosts total number of hosts in the currently executing program
   */
  DistGraph(unsigned host, unsigned numHosts)
      : transposed(false), id(host), numHosts(numHosts) {
    mirrorNodes.resize(numHosts);
    numGlobalNodes = 0;
    numGlobalEdges = 0;

    // report edge buffer size
    //if (host == 0) {
    //  galois::runtime::reportStat_Single(GRNAME, "EdgePartitionBufferSize",
    //                                     (unsigned)edgePartitionSendBufSize);
    //}
  }

  /**
   * Return a vector of pairs denoting mirror node ranges.
   *
   * Assumes all mirror nodes occur after the masters: this invariant should be
   * held by CuSP.
   */
  std::vector<std::pair<uint32_t, uint32_t>> getMirrorRanges() const {
    std::vector<std::pair<uint32_t, uint32_t>> mirrorRangesVector;
    // order of nodes locally is masters, outgoing mirrors, incoming mirrors,
    // so just get from numOwned to end
    if (numOwned != numNodes) {
      assert(numOwned < numNodes);
      mirrorRangesVector.push_back(std::make_pair(numOwned, numNodes));
    }
    return mirrorRangesVector;
  }

  std::vector<std::vector<size_t>>& getMirrorNodes() {
    return mirrorNodes;
  }


  //! Determines which host has the master for a particular node
  //! @returns Host id of node in question
  virtual unsigned getHostID(uint64_t) const = 0;
  //! Determine if a node has a master on this host.
  //! @returns True if passed in global id has a master on this host
  virtual bool isOwned(uint64_t) const = 0;
  //! Determine if a node has a proxy on this host
  //! @returns True if passed in global id has a proxy on this host
  virtual bool isLocal(uint64_t) const = 0;
  /**
   * Returns true if current partition is a vertex cut
   * @returns true if partition being stored in this graph is a vertex cut
   */
  virtual bool is_vertex_cut() const = 0;
  /**
   * Returns Cartesian split (if it exists, else returns pair of 0s
   */
  virtual std::pair<unsigned, unsigned> cartesianGrid() const {
    return std::make_pair(0u, 0u);
  }

  bool isTransposed() { return transposed; }

  /**
   * Converts a local node id into a global node id
   *
   * @param nodeID local node id
   * @returns global node id corresponding to the local one
   */
  inline uint64_t getGID(const uint32_t nodeID) const { return L2G(nodeID); }

  /**
   * Converts a global node id into a local node id
   *
   * @param nodeID global node id
   * @returns local node id corresponding to the global one
   */
  inline uint32_t getLID(const uint64_t nodeID) const { return G2L(nodeID); }

  /**
   * Get data of a node.
   *
   * @param N node to get the data of
   * @param mflag access flag for node data
   * @returns A node data object
   */
  inline NodeTy&
  getData(GraphNode N,
          galois::MethodFlag mflag = galois::MethodFlag::UNPROTECTED) {
    auto& r = graph.getData(N, mflag);
    return r;
  }

  /**
   * Get the edge data for a particular edge in the graph.
   *
   * @param ni edge to get the data of
   * @param mflag access flag for edge data
   * @returns The edge data for the requested edge
   */
  inline typename GraphTy::edge_data_reference
  getEdgeData(edge_iterator ni,
              galois::MethodFlag mflag = galois::MethodFlag::UNPROTECTED) {
    auto& r = graph.getEdgeData(ni, mflag);
    return r;
  }

  /**
   * Gets edge destination of edge ni.
   *
   * @param ni edge id to get destination of
   * @returns Local ID of destination of edge ni
   */
  GraphNode getEdgeDst(edge_iterator ni) { return graph.getEdgeDst(ni); }

  /**
   * Gets the first edge of some node.
   *
   * @param N node to get the edge of
   * @returns iterator to first edge of N
   */
  inline edge_iterator edge_begin(GraphNode N) {
    return graph.edge_begin(N, galois::MethodFlag::UNPROTECTED);
  }

  /**
   * Gets the end edge boundary of some node.
   *
   * @param N node to get the edge of
   * @returns iterator to the end of the edges of node N, i.e. the first edge
   * of the next node (or an "end" iterator if there is no next node)
   */
  inline edge_iterator edge_end(GraphNode N) {
    return graph.edge_end(N, galois::MethodFlag::UNPROTECTED);
  }

  /**
   * Returns an iterable object over the edges of a particular node in the
   * graph.
   *
   * @param N node to get edges iterator over
   */
  inline galois::runtime::iterable<galois::NoDerefIterator<edge_iterator>>
  edges(GraphNode N) {
    return galois::graphs::internal::make_no_deref_range(edge_begin(N),
                                                         edge_end(N));
  }

  /**
   * Gets number of nodes on this (local) graph.
   *
   * @returns number of nodes present in this (local) graph
   */
  inline size_t size() const { return graph.size(); }

  /**
   * Gets number of edges on this (local) graph.
   *
   * @returns number of edges present in this (local) graph
   */
  inline size_t sizeEdges() const { return graph.sizeEdges(); }

  /**
   * Gets number of nodes on this (local) graph.
   *
   * @returns number of nodes present in this (local) graph
   */
  inline size_t numMasters() const { return numOwned; }

  /**
   * Gets number of nodes with edges (may include nodes without edges)
   * on this (local) graph.
   *
   * @returns number of nodes with edges (may include nodes without edges
   * as it measures a contiguous range)
   */
  inline size_t getNumNodesWithEdges() const { return numNodesWithEdges; }

  /**
   * Gets number of nodes on the global unpartitioned graph.
   *
   * @returns number of nodes present in the global unpartitioned graph
   */
  inline size_t globalSize() const { return numGlobalNodes; }

  /**
   * Gets number of edges on the global unpartitioned graph.
   *
   * @returns number of edges present in the global unpartitioned graph
   */
  inline size_t globalSizeEdges() const { return numGlobalEdges; }

  /**
   * Returns a range object that encapsulates all nodes of the graph.
   *
   * @returns A range object that contains all the nodes in this graph
   */
  inline const NodeRangeType& allNodesRange() const {
    assert(specificRanges.size() == 3);
    return specificRanges[0];
  }

  /**
   * Returns a range object that encapsulates only master nodes in this
   * graph.
   *
   * @returns A range object that contains the master nodes in this graph
   */
  inline const NodeRangeType& masterNodesRange() const {
    assert(specificRanges.size() == 3);
    return specificRanges[1];
  }

  /**
   * Returns a range object that encapsulates master nodes and nodes
   * with edges in this graph.
   *
   * @returns A range object that contains the master nodes and the nodes
   * with outgoing edges in this graph
   */
  inline const NodeRangeType& allNodesWithEdgesRange() const {
    assert(specificRanges.size() == 3);
    return specificRanges[2];
  }

protected:
  /**
   * Uses a pre-computed prefix sum to determine division of nodes among
   * threads.
   *
   * The call uses binary search to determine the ranges.
   */
  inline void determineThreadRanges() {
    allNodesRanges = galois::graphs::determineUnitRangesFromPrefixSum(
        galois::runtime::activeThreads, graph.getEdgePrefixSum());
  }

  /**
   * Determines the thread ranges for master nodes only and saves them to
   * the object.
   *
   * Only call after graph is constructed + only call once
   */
  inline void determineThreadRangesMaster() {
    // make sure this hasn't been called before
    assert(masterRanges.size() == 0);

    // first check if we even need to do any work; if already calculated,
    // use already calculated vector
    if (beginMaster == 0 && (beginMaster + numOwned) == size()) {
      masterRanges = allNodesRanges;
    } else if (beginMaster == 0 &&
               (beginMaster + numOwned) == numNodesWithEdges &&
               withEdgeRanges.size() != 0) {
      masterRanges = withEdgeRanges;
    } else {
      galois::gDebug("Manually det. master thread ranges");
      masterRanges = galois::graphs::determineUnitRangesFromGraph(
          graph, galois::runtime::activeThreads, beginMaster,
          beginMaster + numOwned, 0);
    }
  }

  /**
   * Determines the thread ranges for nodes with edges only and saves them to
   * the object.
   *
   * Only call after graph is constructed + only call once
   */
  inline void determineThreadRangesWithEdges() {
    // make sure not called before
    assert(withEdgeRanges.size() == 0);

    // first check if we even need to do any work; if already calculated,
    // use already calculated vector
    if (numNodesWithEdges == size()) {
      withEdgeRanges = allNodesRanges;
    } else if (beginMaster == 0 &&
               (beginMaster + numOwned) == numNodesWithEdges &&
               masterRanges.size() != 0) {
      withEdgeRanges = masterRanges;
    } else {
      galois::gDebug("Manually det. with edges thread ranges");
      withEdgeRanges = galois::graphs::determineUnitRangesFromGraph(
          graph, galois::runtime::activeThreads, 0, numNodesWithEdges,
          0);
    }
  }

  /**
   * Initializes the 3 range objects that a user can access to iterate
   * over the graph in different ways.
   */
  void initializeSpecificRanges() {
    assert(specificRanges.size() == 0);

    // TODO/FIXME assertion likely not safe if a host gets no nodes
    // make sure the thread ranges have already been calculated
    // for the 3 ranges
    assert(allNodesRanges.size() != 0);
    assert(masterRanges.size() != 0);
    assert(withEdgeRanges.size() != 0);

    // 0 is all nodes
    specificRanges.push_back(galois::runtime::makeSpecificRange(
        boost::counting_iterator<size_t>(0),
        boost::counting_iterator<size_t>(size()), allNodesRanges.data()));

    // 1 is master nodes
    specificRanges.push_back(galois::runtime::makeSpecificRange(
        boost::counting_iterator<size_t>(beginMaster),
        boost::counting_iterator<size_t>(beginMaster + numOwned),
        masterRanges.data()));

    // 2 is with edge nodes
    specificRanges.push_back(galois::runtime::makeSpecificRange(
        boost::counting_iterator<size_t>(0),
        boost::counting_iterator<size_t>(numNodesWithEdges),
        withEdgeRanges.data()));

    assert(specificRanges.size() == 3);
  }

  /**
   * Specific range editor: makes the range for edges equivalent to the range
   * for masters.
   */
  void edgesEqualMasters() { specificRanges[2] = specificRanges[1]; }

public:
  // TODO make these work again
  /**
   * Write the local LC_CSR graph to the file on a disk.
   *
   * @param localGraphFileName file name to write local graph to.
   * @todo revive this
   */
  void save_local_graph_to_file(std::string localGraphFileName = "local_graph") {
    galois::gWarn("Currently not implemented. TODO");
    //using namespace boost::archive;
    //galois::StatTimer dGraphTimerSaveLocalGraph("TimerSaveLocalGraph", GRNAME);
    //dGraphTimerSaveLocalGraph.start();

    //std::string fileName = localGraphFileName + "_" + std::to_string(id);

    //galois::gDebug("[", id, "] inside save_local_graph_to_file \n");

    //std::ofstream outputStream(fileName, std::ios::binary);
    //if (!outputStream.is_open()) {
    //  std::cerr << "ERROR: Could not open " << fileName
    //            << " to save local graph!!!\n";
    //}

    //boost::archive::binary_oarchive ar(outputStream, boost::archive::no_header);

    //// graph topology
    //ar << graph;
    //ar << numGlobalNodes;
    //ar << numGlobalEdges;

    //// bool
    //ar << transposed;

    //// Proxy information
    //// TODO: Find better way to serialize vector of vectors in boost
    //// serialization
    //for (uint32_t i = 0; i < numHosts; ++i) {
    //  ar << masterNodes[i];
    //  ar << mirrorNodes[i];
    //}

    //ar << numOwned;
    //ar << beginMaster;
    //ar << numNodesWithEdges;
    //ar << gid2host;

    //// Serialize partitioning scheme specific data structures.
    //boostSerializeLocalGraph(ar);

    //outputStream.close();
    //dGraphTimerSaveLocalGraph.stop();
  }

  /**
   * Read the local LC_CSR graph from the file on a disk.
   *
   * @param localGraphFileName file name to read local graph from.
   * @todo revive this
   */
  void
  read_local_graph_from_file(std::string localGraphFileName = "local_graph") {
    galois::gWarn("Currently not implemented. TODO");
    //using namespace boost::archive;
    //galois::StatTimer dGraphTimerReadLocalGraph("TimerReadLocalGraph", GRNAME);
    //dGraphTimerReadLocalGraph.start();

    //std::string fileName = localGraphFileName + "_" + std::to_string(id);

    //std::ifstream inputStream(fileName, std::ios::binary);
    //if (!inputStream.is_open()) {
    //  std::cerr << "ERROR: Could not open " << fileName
    //            << " to read local graph!!!\n";
    //}

    //galois::gPrint("[", id, "] inside read_local_graph_from_file \n");

    //boost::archive::binary_iarchive ar(inputStream, boost::archive::no_header);

    //// Graph topology
    //ar >> graph;
    //ar >> numGlobalNodes;
    //ar >> numGlobalEdges;

    //// bool
    //ar >> transposed;

    //// Proxy information
    //// TODO: Find better way to Deserialize vector of vectors in boost
    //// serialization
    //for (uint32_t i = 0; i < numHosts; ++i) {
    //  ar >> masterNodes[i];
    //  ar >> mirrorNodes[i];
    //}

    //ar >> numOwned;
    //ar >> beginMaster;
    //ar >> numNodesWithEdges;
    //ar >> gid2host;

    //// Serialize partitioning scheme specific data structures.
    //boostDeSerializeLocalGraph(ar);

    //allNodesRanges.clear();
    //masterRanges.clear();
    //withEdgeRanges.clear();
    //specificRanges.clear();

    //// find ranges again
    //determineThreadRanges();
    //determineThreadRangesMaster();
    //determineThreadRangesWithEdges();
    //initializeSpecificRanges();

    //// Exchange information among hosts
    //// send_info_to_host();

    //inputStream.close();
    //dGraphTimerReadLocalGraph.stop();
  }

  /**
   * Deallocates underlying LC CSR Graph
   */
  void deallocate() {
    galois::gDebug("Deallocating CSR in DistGraph");
    graph.deallocate();
  }
};

template <typename NodeTy, typename EdgeTy>
constexpr const char* const
    galois::graphs::DistGraph<NodeTy, EdgeTy>::GRNAME;
} // end namespace graphs
} // end namespace galois

#endif //_GALOIS_DIST_HGRAPH_H
