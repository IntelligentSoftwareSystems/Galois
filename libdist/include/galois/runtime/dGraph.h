/** partitioned graph wrapper -*- C++ -*-
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
 * Header file that includes base functionality for the distributed
 * graph object, including the synchronization infrastructure.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Gurbinder Gill <gurbinder533@gmail.com>
 * @author Roshan Dathathri <roshan@cs.utexas.edu>
 * @author Loc Hoang <l_hoang@utexas.edu>
 */
#ifndef _GALOIS_DIST_HGRAPH_H
#define _GALOIS_DIST_HGRAPH_H

#include <vector>
#include <set>
#include <algorithm>
#include <unordered_map>
#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>

#include "galois/gstl.h"
#include "galois/Galois.h"
#include "galois/graphs/LC_CSR_Graph.h"
#include "galois/runtime/Substrate.h"
#include "galois/runtime/DistStats.h"
#include "galois/runtime/GlobalObj.h"
#include "galois/graphs/OfflineGraph.h"
#include "galois/runtime/sync_structures.h"
#include "galois/runtime/DataCommMode.h"
#include "galois/runtime/Dynamic_bitset.h"
#include "galois/substrate/ThreadPool.h"

#ifdef __GALOIS_HET_CUDA__
#include "galois/runtime/Cuda/cuda_mtypes.h"
#endif
#ifdef __GALOIS_HET_OPENCL__
#include "galois/opencl/CL_Header.h"
#endif

#include "galois/runtime/bare_mpi.h"

#include "llvm/Support/CommandLine.h"

namespace cll = llvm::cl;

enum MASTERS_DISTRIBUTION {
  BALANCED_MASTERS, BALANCED_EDGES_OF_MASTERS, BALANCED_MASTERS_AND_EDGES
};

extern cll::opt<bool> useGidMetadata;
extern cll::opt<MASTERS_DISTRIBUTION> masters_distribution;
extern cll::opt<uint32_t> nodeWeightOfMaster;
extern cll::opt<uint32_t> edgeWeightOfMaster;
extern cll::opt<uint32_t> nodeAlphaRanges;
extern cll::opt<unsigned> numFileThreads;

// Enumerations for specifiying read/write location for sync calls
enum WriteLocation { writeSource, writeDestination, writeAny };
enum ReadLocation { readSource, readDestination, readAny };

/**
 * Base hGraph class that all distributed graphs extend from.
 *
 * @tparam NodeTy type of node data for the graph
 * @tparam EdgeTy type of edge data for the graph
 * @tparam BSPNode specifies if node is a BSP node, e.g. it has an "old"
 * and a "new" that you can switch between for bulk-synchronous parallel
 * phases
 * @tparam BSPEdge specifies if edge is a BSP edge, e.g. it has an "old"
 * and a "new" that you can switch between for bulk-synchronous parallel
 * phases
 */
template<typename NodeTy, typename EdgeTy, bool BSPNode = false,
         bool BSPEdge = false>
class hGraph: public GlobalObject {
private:
  constexpr static const char* const GRNAME = "dGraph";

  typedef typename std::conditional<
    BSPNode, std::pair<NodeTy, NodeTy>, NodeTy
  >::type realNodeTy;
  typedef typename std::conditional<
    BSPEdge && !std::is_void<EdgeTy>::value, std::pair<EdgeTy, EdgeTy>,
    EdgeTy
  >::type realEdgeTy;

  typedef typename galois::graphs::LC_CSR_Graph<realNodeTy, realEdgeTy, true>
    GraphTy; // do not use locks, use default interleaved numa allocation

  bool round;

protected:
  GraphTy graph;
  enum SyncType { syncReduce, syncBroadcast };

  bool transposed;

  // global graph
  uint64_t numGlobalNodes; // Total nodes in the global unpartitioned graph.
  uint64_t numGlobalEdges; // Total edges in the global unpartitioned graph.

  const unsigned id; // copy of net.ID
  const uint32_t numHosts; // copy of net.Num 

  // local graph
  // size() = Number of nodes created on this host (masters + mirrors)
  uint32_t numOwned; // Number of nodes owned (masters) by this host
                     // size() - numOwned = mirrors on this host
  uint32_t beginMaster; // local id of the beginning of master nodes
                        // beginMaster + numOwned = local id of the end of master nodes
  uint32_t numNodesWithEdges; // Number of nodes (masters + mirrors) that have outgoing edges 

  // Master nodes on each host.
  std::vector<std::pair<uint64_t, uint64_t>> gid2host;

  uint64_t last_nodeID_withEdges_bipartite; // used only for bipartite graphs

  // memoization optimization
  // mirror nodes from different hosts. For reduce
  std::vector<std::vector<size_t>> mirrorNodes;
  // master nodes on different hosts. For broadcast
  std::vector<std::vector<size_t>> masterNodes;

  // FIXME: pass the flag as function paramater instead
  // a pointer set during sync_on_demand calls that points to the status
  // of a bitvector with regard to where data has been synchronized
  BITVECTOR_STATUS* currentBVFlag;

private:
  // vector for determining range objects for master nodes + nodes
  // with edges (which includes masters)
  std::vector<uint32_t> masterRanges;
  std::vector<uint32_t> withEdgeRanges;

  // vector of ranges that stores the 3 different range objects that a user is
  // able to access
  std::vector<galois::runtime::SpecificRange<boost::counting_iterator<size_t>>>
    specificRanges;

#ifdef __GALOIS_BARE_MPI_COMMUNICATION__
  std::vector<MPI_Group> mpi_identity_groups;
#endif

protected:
  void printStatistics() {
    if (id == 0) {
      galois::gPrint("Total nodes: ", numGlobalNodes, "\n");
      galois::gPrint("Total edges: ", numGlobalEdges, "\n");
    }
    galois::gPrint("[", id, "] Master nodes: ", numOwned, "\n");
    galois::gPrint("[", id, "] Mirror nodes: ", size() - numOwned, "\n");
    galois::gPrint("[", id, "] Nodes with edges: ", numNodesWithEdges, "\n");
    galois::gPrint("[", id, "] Edges: ", sizeEdges(), "\n");
  }

public:
  /****** VIRTUAL FUNCTIONS *********/
  virtual uint32_t G2L(uint64_t) const = 0 ;
  virtual uint64_t L2G(uint32_t) const = 0;
  virtual bool is_vertex_cut() const = 0;
  virtual unsigned getHostID(uint64_t) const = 0;
  virtual bool isOwned(uint64_t) const = 0;
  virtual bool isLocal(uint64_t) const = 0;

  // Requirement: For all X and Y,
  // On X, nothingToSend(Y) <=> On Y, nothingToRecv(X)
  // Note: templates may not be virtual, so passing types as arguments
  /**
   * Determine if we have anything that we need to send to a particular host
   *
   * @param host Host number that we may or may not send to
   * @param syncType Synchronization type to determine which nodes on a
   * host need to be considered
   * @param writeLocation If data is being written to on source or 
   * destination (or both)
   * @param readLocation If data is being read from on source or 
   * destination (or both)
   * @returns true if there is nothing to send to a host, false otherwise
   */
  virtual bool nothingToSend(unsigned host, SyncType syncType, 
                             WriteLocation writeLocation, 
                             ReadLocation readLocation) {
     auto &sharedNodes = (syncType == syncReduce) ? mirrorNodes : masterNodes;
     return (sharedNodes[host].size() == 0);
  }

  /**
   * Determine if we have anything that we need to receive from a particular 
   * host
   *
   * @param host Host number that we may or may not receive from
   * @param syncType Synchronization type to determine which nodes on a
   * host need to be considered
   * @param writeLocation If data is being written to on source or 
   * destination (or both)
   * @param readLocation If data is being read from on source or 
   * destination (or both)
   * @returns true if there is nothing to receive from a host, false otherwise
   */
  virtual bool nothingToRecv(unsigned host, SyncType syncType, 
                             WriteLocation writeLocation, 
                             ReadLocation readLocation) {
     auto &sharedNodes = (syncType == syncReduce) ? masterNodes : mirrorNodes;
     return (sharedNodes[host].size() == 0);
  }

  virtual void reset_bitset(SyncType syncType,
                            void (*bitset_reset_range)(size_t,
                                                       size_t)) const = 0;

private:
  uint32_t num_run; //Keep track of number of runs.
  uint32_t num_iteration; //Keep track of number of iterations.

  // Stats: for rough estimate of sendBytes.

  /**
   * Get the node data for a particular node in the graph.
   *
   * This function is called if we have BSP style nodes (i.e. return
   * "new" or "old" depending on round number).
   *
   * @tparam en Specifies if this is a BSP node getData. Should be true.
   *
   * @param N node to get the data of
   * @param mflag access flag for node data
   * @returns A node data object
   */
  template<bool en, typename std::enable_if<en>::type* = nullptr>
  inline NodeTy& getDataImpl(typename GraphTy::GraphNode N,
                      galois::MethodFlag mflag = galois::MethodFlag::UNPROTECTED) {
    auto& r = graph.getData(N, mflag);
    return round ? r.first : r.second;
  }

  /**
   * Get the node data for a particular node in the graph.
   *
   * This function is called if do NOT have BSP style nodes.
   *
   * @tparam en Specifies if this is a BSP node getData. Should be false.
   *
   * @param N node to get the data of
   * @param mflag access flag for node data
   * @returns A node data object
   */
  template<bool en, typename std::enable_if<!en>::type* = nullptr>
  inline NodeTy& getDataImpl(typename GraphTy::GraphNode N,
                      galois::MethodFlag mflag = galois::MethodFlag::UNPROTECTED) {
    auto& r = graph.getData(N, mflag);
    return r;
  }

  /**
   * Get the node data for a particular node in the graph.
   *
   * This function is called if you have BSP style edges.
   *
   * @tparam en Specifies if this is a BSP node getData. Should be true.
   *
   * @param ni edge to get the data of
   * @param mflag access flag for edge data
   * @returns The edge data for the requested edge
   */
  template<bool en, typename std::enable_if<en>::type* = nullptr>
  inline typename GraphTy::edge_data_reference getEdgeDataImpl(
      typename GraphTy::edge_iterator ni,
      galois::MethodFlag mflag = galois::MethodFlag::UNPROTECTED) {
    auto& r = graph.getEdgeData(ni, mflag);
    return round ? r.first : r.second;
  }

  /**
   * Get the node data for a particular node in the graph.
   *
   * This function is called if you do not have BSP style edges.
   *
   * @tparam en Specifies if this is a BSP node getData. Should be false.
   *
   * @param ni edge to get the data of
   * @param mflag access flag for edge data
   * @returns The edge data for the requested edge
   */
  template<bool en, typename std::enable_if<!en>::type* = nullptr>
  inline typename GraphTy::edge_data_reference getEdgeDataImpl(
      typename GraphTy::edge_iterator ni,
      galois::MethodFlag mflag = galois::MethodFlag::UNPROTECTED) {
    auto& r = graph.getEdgeData(ni, mflag);
    return r;
  }

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
      uint64_t numNodes_to_divide, const std::vector<unsigned>& scalefactor,
      unsigned DecomposeFactor = 1) {
    if (scalefactor.empty() || (numHosts * DecomposeFactor == 1)) {
      for (unsigned i = 0; i < numHosts * DecomposeFactor; ++i)
        gid2host.push_back(galois::block_range(
                             0U, (unsigned)numNodes_to_divide, i,
                             numHosts * DecomposeFactor
                           ));
    } else { // TODO: not compatible with DecomposeFactor.
      assert(scalefactor.size() == numHosts);

      unsigned numBlocks = 0;

      for (unsigned i = 0; i < numHosts; ++i) {
        numBlocks += scalefactor[i];
      }

      std::vector<std::pair<uint64_t, uint64_t>> blocks;
      for (unsigned i = 0; i < numBlocks; ++i) {
        blocks.push_back(galois::block_range(
                           0U, (unsigned)numNodes_to_divide, i, numBlocks
                         ));
      }

      std::vector<unsigned> prefixSums;
      prefixSums.push_back(0);

      for (unsigned i = 1; i < numHosts; ++i) {
        prefixSums.push_back(prefixSums[i - 1] + scalefactor[i - 1]);
      }

      for (unsigned i = 0; i < numHosts; ++i) {
        unsigned firstBlock = prefixSums[i];
        unsigned lastBlock = prefixSums[i] + scalefactor[i] - 1;
        gid2host.push_back(std::make_pair(blocks[firstBlock].first,
                                          blocks[lastBlock].second));
      }
    }
  }

  // TODO:: MAKE IT WORK WITH DECOMPOSE FACTOR
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
      uint64_t numNodes_to_divide, const std::vector<unsigned>& scalefactor,
      unsigned DecomposeFactor = 1) {
    if (edgeWeightOfMaster == 0) {
      edgeWeightOfMaster = 1;
    }

    auto& net = galois::runtime::getSystemNetworkInterface();

    gid2host.resize(numHosts*DecomposeFactor);
    for(unsigned d = 0; d < DecomposeFactor; ++d){
      auto r = g.divideByNode(0, edgeWeightOfMaster, (id + d * numHosts), 
                              numHosts * DecomposeFactor, scalefactor);
      gid2host[id + d*numHosts].first = *(r.first.first);
      gid2host[id + d*numHosts].second = *(r.first.second);
    }

    for (unsigned h = 0; h < numHosts; ++h) {
      if (h == id) continue;
      galois::runtime::SendBuffer b;
      for(unsigned d = 0; d < DecomposeFactor; ++d){
        galois::runtime::gSerialize(b, gid2host[id + d*numHosts]);
      }
      net.sendTagged(h, galois::runtime::evilPhase, b);
    }
    net.flush();
    unsigned received = 1;
    while (received < numHosts) {
      decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;
      do {
        net.handleReceives();
        p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
      } while (!p);
      assert(p->first != id);
      auto& b = p->second;
      for(unsigned d = 0; d < DecomposeFactor; ++d){
        galois::runtime::gDeserialize(b, gid2host[p->first + d*numHosts]);
      }
      ++received;
    }
    ++galois::runtime::evilPhase;
  }

  //TODO:: MAKE IT WORK WITH DECOMPOSE FACTOR
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
   * out of 1 block had the decompose factor been set to 1.
   */
  void computeMastersBalancedNodesAndEdges(galois::graphs::OfflineGraph& g,
      uint64_t numNodes_to_divide, const std::vector<unsigned>& scalefactor,
      unsigned DecomposeFactor = 1) {
    if (nodeWeightOfMaster == 0) {
      nodeWeightOfMaster = g.sizeEdges() / g.size(); // average degree
    }

    if (edgeWeightOfMaster == 0) {
      edgeWeightOfMaster = 1;
    }
    auto& net = galois::runtime::getSystemNetworkInterface();
    gid2host.resize(numHosts);
    auto r = g.divideByNode(nodeWeightOfMaster, edgeWeightOfMaster, id, numHosts, scalefactor);
    gid2host[id].first = *r.first.first;
    gid2host[id].second = *r.first.second;
    for (unsigned h = 0; h < numHosts; ++h) {
      if (h == id) continue;
      galois::runtime::SendBuffer b;
      galois::runtime::gSerialize(b, gid2host[id]);
      net.sendTagged(h, galois::runtime::evilPhase, b);
    }
    net.flush();
    unsigned received = 1;
    while (received < numHosts) {
      decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;
      do {
        net.handleReceives();
        p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
      } while (!p);
      assert(p->first != id);
      auto& b = p->second;
      galois::runtime::gDeserialize(b, gid2host[p->first]);
      ++received;
    }
    ++galois::runtime::evilPhase;
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
   * @param isBipartite Specifies if the graph is a bipartite graph
   * @param DecomposeFactor Specifies how decomposed the blocking 
   * of nodes should be. For example, a factor of 2 will make 2 blocks
   * out of 1 block had the decompose factor been set to 1.
   */
  uint64_t computeMasters(galois::graphs::OfflineGraph& g,
      const std::vector<unsigned>& scalefactor,
      bool isBipartite = false, unsigned DecomposeFactor = 1) {
    galois::Timer timer;
    timer.start();
    g.reset_seek_counters();

    uint64_t numNodes_to_divide = 0;

    if (isBipartite) {
      for (uint64_t n = 0; n < g.size(); ++n){
        if(std::distance(g.edge_begin(n), g.edge_end(n))){
                ++numNodes_to_divide;
                last_nodeID_withEdges_bipartite = n;
        }
      }
    } else {
      numNodes_to_divide = g.size();
    }

    // compute masters for all nodes
    switch (masters_distribution) {
      case BALANCED_MASTERS:
        computeMastersBlockedNodes(g, numNodes_to_divide, scalefactor, 
                                   DecomposeFactor);
        break;
      case BALANCED_MASTERS_AND_EDGES:
        computeMastersBalancedNodesAndEdges(g, numNodes_to_divide, scalefactor, 
                                            DecomposeFactor);
        break;
      case BALANCED_EDGES_OF_MASTERS:
      default:
        computeMastersBalancedEdges(g, numNodes_to_divide, scalefactor, 
                                    DecomposeFactor);
        break;
    }

    timer.stop();
    galois::gPrint("[", id, "] Master distribution time : ", timer.get_usec()/1000000.0f,
        " seconds to read ", g.num_bytes_read(), " bytes in ", g.num_seeks(),
        " seeks (", g.num_bytes_read()/(float)timer.get_usec(), " MBPS)\n");
    return numNodes_to_divide;
  }

private:
  void initBareMPI() {
#ifdef __GALOIS_BARE_MPI_COMMUNICATION__
    if (bare_mpi == noBareMPI) return;

#ifdef GALOIS_USE_LWCI
    int provided;
    int rv = MPI_Init_thread (NULL, NULL, MPI_THREAD_FUNNELED, &provided);
    if (rv != MPI_SUCCESS) {
      MPI_Abort(MPI_COMM_WORLD, rv);
    }
    if(!(provided >= MPI_THREAD_FUNNELED)){
      GALOIS_DIE("MPI_THREAD_FUNNELED not supported\n");
    }
    assert(provided >= MPI_THREAD_FUNNELED);
    int taskRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &taskRank);
    if (taskRank != id) GALOIS_DIE("Mismatch in MPI rank");
    int numTasks;
    MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
    auto& net = galois::runtime::getSystemNetworkInterface();
    if (numTasks != numHosts) GALOIS_DIE("Mismatch in MPI rank");
#endif

    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    mpi_identity_groups.resize(numHosts);
    for (unsigned x = 0; x < numHosts; ++x) {
      const int g[1] = {(int)x};
      MPI_Group_incl(world_group, 1, g, &mpi_identity_groups[x]);
    }

    if (id == 0) galois::gDebug("Using bare MPI\n");
#endif
  }

  void finalizeBareMPI() {
#ifdef __GALOIS_BARE_MPI_COMMUNICATION__
    if (bare_mpi == noBareMPI) return;

#ifdef GALOIS_USE_LWCI
    int rv = MPI_Finalize();
    if (rv != MPI_SUCCESS) {
      MPI_Abort(MPI_COMM_WORLD, rv);
    }
#endif
#endif
  }

public:
  typedef typename GraphTy::GraphNode GraphNode;
  typedef typename GraphTy::iterator iterator;
  typedef typename GraphTy::const_iterator const_iterator;
  typedef typename GraphTy::local_iterator local_iterator;
  typedef typename GraphTy::const_local_iterator const_local_iterator;
  typedef typename GraphTy::edge_iterator edge_iterator;

  /**
   * Constructor for hGraph. Initializes metadata fields.
   *
   * @param host host number that this graph resides on
   * @param numHosts total number of hosts in the currently executing program
   */
  hGraph(unsigned host, unsigned numHosts) :
      GlobalObject(this), round(false), transposed(false), id(host),
      numHosts(numHosts) {
    if (useGidMetadata) {
      if (enforce_data_mode != offsetsData) {
        useGidMetadata = false;
      }
    }

    masterNodes.resize(numHosts);
    mirrorNodes.resize(numHosts);

    num_run = 0;
    num_iteration = 0;
    numGlobalEdges = 0;
    currentBVFlag = nullptr;

    initBareMPI();
  }

  ~hGraph() {
    finalizeBareMPI();
  }

protected:
  /**
   * Sets up the communication between the different hosts that contain
   * different parts of the graph by exchanging master/mirror information.
   */
  void setup_communication() {
    galois::StatTimer Tcomm_setup("COMMUNICATION_SETUP_TIME", GRNAME);

    // barrier so that all hosts start the timer together
    galois::runtime::getHostBarrier().wait();

    Tcomm_setup.start();

    // Exchange information for memoization optimization.
    exchange_info_init();

    // convert the global ids stored in the master/mirror nodes arrays to local
    // ids
    for (uint32_t h = 0; h < masterNodes.size(); ++h) {
      galois::do_all(galois::iterate(0ul, masterNodes[h].size()),
                     [&](uint32_t n) {
                       masterNodes[h][n] = G2L(masterNodes[h][n]);
                     },
                     galois::loopname(get_run_identifier("MASTER_NODES").c_str()),
                     galois::timeit(),
                     galois::steal<false>(),
                     galois::no_stats());
    }

    for (uint32_t h = 0; h < mirrorNodes.size(); ++h) {
      galois::do_all(galois::iterate(0ul, mirrorNodes[h].size()),
                     [&](uint32_t n) {
                       mirrorNodes[h][n] = G2L(mirrorNodes[h][n]);
                     },
                     galois::loopname(get_run_identifier("MIRROR_NODES").c_str()),
                     galois::timeit(),
                     galois::steal<false>(),
                     galois::no_stats());
    }

    Tcomm_setup.stop();

    // report masters/mirrors to/from other hosts as statistics
    for (auto x = 0U; x < masterNodes.size(); ++x) {
      std::string master_nodes_str = "MASTER_NODES_TO_" + std::to_string(x);
      galois::runtime::reportStat_Tsum(GRNAME, master_nodes_str, 
                                       masterNodes[x].size());
    }

    for (auto x = 0U; x < mirrorNodes.size(); ++x) {
      std::string mirror_nodes_str = "MIRROR_NODES_FROM_" + std::to_string(x);
      if (x == id) continue;
      galois::runtime::reportStat_Tsum(GRNAME, mirror_nodes_str, 
                                       mirrorNodes[x].size());
    }

    send_info_to_host();
  }

public:
  /**
   * Wrapper getData that calls into the get data that distinguishes between
   * a BSP node and a non BSP node.
   *
   * @param N node to get the data of
   * @param mflag access flag for node data
   * @returns A node data object
   */
  inline NodeTy& getData(GraphNode N, 
                  galois::MethodFlag mflag = galois::MethodFlag::UNPROTECTED) {
    auto& r = getDataImpl<BSPNode>(N, mflag);
    return r;
  }

  /**
   * Wrapper getEdgeData that calls into the get edge data that distinguishes 
   * between a BSP edge and a non BSP edge.
   *
   * @param ni edge to get the data of
   * @param mflag access flag for edge data
   * @returns The edge data for the requested edge
   */
  inline typename GraphTy::edge_data_reference getEdgeData(edge_iterator ni, 
                        galois::MethodFlag mflag = galois::MethodFlag::UNPROTECTED) {
    return getEdgeDataImpl<BSPEdge>(ni, mflag);
  }

  GraphNode getEdgeDst(edge_iterator ni) {
    return graph.getEdgeDst(ni);
  }

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
  inline size_t size() const {
    return graph.size();
  }

  /**
   * Gets number of edges on this (local) graph.
   *
   * @returns number of edges present in this (local) graph
   */
  inline size_t sizeEdges() const {
    return graph.sizeEdges();
  }

  /**
   * Gets number of nodes on the global unpartitioned graph.
   *
   * @returns number of nodes present in the global unpartitioned graph
   */
  inline size_t globalSize() const {
    return numGlobalNodes;
  }

  /**
   * Gets number of edges on the global unpartitioned graph.
   *
   * @returns number of edges present in the global unpartitioned graph
   */
  inline size_t globalSizeEdges() const {
    return numGlobalEdges;
  }

  /**
   * Returns a range object that encapsulates all nodes of the graph.
   *
   * @returns A range object that contains all the nodes in this graph
   */
  inline const galois::runtime::SpecificRange<boost::counting_iterator<size_t>>&
  allNodesRange() const {
    assert(specificRanges.size() == 3);
    return specificRanges[0];
  }

  /**
   * Returns a range object that encapsulates only master nodes in this
   * graph.
   *
   * @returns A range object that contains the master nodes in this graph
   */
  inline const galois::runtime::SpecificRange<boost::counting_iterator<size_t>>&
  masterNodesRange() const {
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
  inline const galois::runtime::SpecificRange<boost::counting_iterator<size_t>>&
  allNodesWithEdgesRange() const {
    assert(specificRanges.size() == 3);
    return specificRanges[2];
  }

protected:
  /**
   * A version of determine_thread_ranges that computes the range offsets
   * for a specific range of the graph.
   *
   * Note threadRangesEdge is not determined for this variant, meaning
   * allocateSpecified will not work if you choose to use this variant.
   *
   * @param beginNode Beginning of range
   * @param endNode End of range, non-inclusive
   * @param returnRanges Vector to store thread offsets for ranges in
   */
  inline void determine_thread_ranges(uint32_t beginNode, uint32_t endNode,
                               std::vector<uint32_t>& returnRanges) {
    graph.determineThreadRanges(beginNode, endNode, returnRanges,
                                nodeAlphaRanges);
  }

  /**
   * A version of determine_thread_ranges that uses a pre-computed prefix sum
   * to determine division of nodes among threads.
   * 
   * The call uses binary search to determine the ranges.
   *
   * @param total_nodes The total number of nodes (masters + mirrors) on this
   * partition.
   * @param edge_prefix_sum The edge prefix sum of the nodes on this partition.
   */
  inline void determine_thread_ranges(uint32_t total_nodes,
                               std::vector<uint64_t> edge_prefix_sum) {
    graph.determineThreadRangesByNode(edge_prefix_sum);
  }

  /**
   * Determines the thread ranges for master nodes only and saves them to
   * the object.
   *
   * Only call after graph is constructed + only call once
   */
  inline void determine_thread_ranges_master() {
    // make sure this hasn't been called before
    assert(masterRanges.size() == 0);

    // first check if we even need to do any work; if already calculated,
    // use already calculated vector
    if (beginMaster == 0 && (beginMaster + numOwned) == size()) {
      masterRanges = graph.getThreadRangesVector();
    } else if (beginMaster == 0 && (beginMaster + numOwned) == numNodesWithEdges &&
               withEdgeRanges.size() != 0) {
      masterRanges = withEdgeRanges;
    } else {
      galois::gDebug("Manually det. master thread ranges\n");
      graph.determineThreadRanges(beginMaster, beginMaster + numOwned, masterRanges,
                                  nodeAlphaRanges);
    }
  }

  /**
   * Determines the thread ranges for nodes with edges only and saves them to
   * the object.
   *
   * Only call after graph is constructed + only call once
   */
  inline void determine_thread_ranges_with_edges() {
    // make sure not called before
    assert(withEdgeRanges.size() == 0);

    // first check if we even need to do any work; if already calculated,
    // use already calculated vector
    if (numNodesWithEdges == size()) {
      withEdgeRanges = graph.getThreadRangesVector();
    } else if (beginMaster == 0 && (beginMaster + numOwned) == numNodesWithEdges &&
               masterRanges.size() != 0) {
      withEdgeRanges = masterRanges;
    } else {
      galois::gDebug("Manually det. with edges thread ranges");
      graph.determineThreadRanges(0, numNodesWithEdges, withEdgeRanges,
                                  nodeAlphaRanges);
    }
  }

  /**
   * Initializes the 3 range objects that a user can access to iterate
   * over the graph in different ways.
   */
  void initialize_specific_ranges() {
    assert(specificRanges.size() == 0);

    // make sure the thread ranges have already been calculated
    // for the 3 ranges
    assert(graph.getThreadRangesVector().size() != 0);
    assert(masterRanges.size() != 0);
    assert(withEdgeRanges.size() != 0);

    // 0 is all nodes
    specificRanges.push_back(
      galois::runtime::makeSpecificRange(
        boost::counting_iterator<size_t>(0),
        boost::counting_iterator<size_t>(size()),
        graph.getThreadRanges()
      )
    );

    // 1 is master nodes
    specificRanges.push_back(
      galois::runtime::makeSpecificRange(
        boost::counting_iterator<size_t>(beginMaster),
        boost::counting_iterator<size_t>(beginMaster + numOwned),
        masterRanges.data()
      )
    );

    // 2 is with edge nodes
    specificRanges.push_back(
      galois::runtime::makeSpecificRange(
        boost::counting_iterator<size_t>(0),
        boost::counting_iterator<size_t>(numNodesWithEdges),
        withEdgeRanges.data()
      )
    );

    assert(specificRanges.size() == 3);
  }

private:
  /**
   * TODO
   */
  void exchange_info_init() {
    auto& net = galois::runtime::getSystemNetworkInterface();

    // send off the mirror nodes 
    for (unsigned x = 0; x < numHosts; ++x) {
      if (x == id) continue;

      galois::runtime::SendBuffer b;
      gSerialize(b, mirrorNodes[x]);
      net.sendTagged(x, galois::runtime::evilPhase, b);
    }

    // receive the mirror nodes 
    for (unsigned x = 0; x < numHosts; ++x) {
      if(x == id) continue;

      decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;
      do {
        net.handleReceives();
        p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
      } while(!p);

      galois::runtime::gDeserialize(p->second, masterNodes[p->first]);
    }
    ++galois::runtime::evilPhase;
  }

  /**
   * Reports master/mirror stats.
   * Assumes that communication has already occured so that the host
   * calling it actually has the info required.
   * 
   * @param global_total_mirror_nodes number of mirror nodes on all hosts
   * @param global_total_owned_nodes number of "owned" nodes on all hosts
   */
  void report_master_mirror_stats(uint64_t global_total_mirror_nodes,
                                  uint64_t global_total_owned_nodes) {
    float replication_factor = (float)(global_total_mirror_nodes + numGlobalNodes) /
                               (float)numGlobalNodes;
    galois::runtime::reportStat_Single(GRNAME, 
        "REPLICATION_FACTOR_" + get_run_identifier(), replication_factor);

    galois::runtime::reportStat_Single(GRNAME, 
        "TOTAL_NODES_" + get_run_identifier(), numGlobalNodes);
    galois::runtime::reportStat_Single(GRNAME, 
        "TOTAL_GLOBAL_MIRROR_NODES_" + get_run_identifier(), 
        global_total_mirror_nodes);
  }

  /**
   * Send statistics about master/mirror nodes to each host, and
   * report the statistics.
   */
  void send_info_to_host() {
    auto& net = galois::runtime::getSystemNetworkInterface();

    // send info to host
    for (unsigned x = 0; x < numHosts; ++x) {
      if(x == id) continue;

      galois::runtime::SendBuffer b;
      gSerialize(b, size() - numOwned, numOwned);
      net.sendTagged(x, galois::runtime::evilPhase, b);
    }

    // receive
    uint64_t global_total_mirror_nodes = size() - numOwned;
    uint64_t global_total_owned_nodes = numOwned;

    for (unsigned x = 0; x < numHosts; ++x) {
      if (x == id) continue;

      decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;
      do {
        net.handleReceives();
        p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
      } while(!p);

      uint64_t total_mirror_nodes_from_others;
      uint64_t total_owned_nodes_from_others;
      galois::runtime::gDeserialize(p->second, total_mirror_nodes_from_others, 
                                    total_owned_nodes_from_others);
      global_total_mirror_nodes += total_mirror_nodes_from_others;
      global_total_owned_nodes += total_owned_nodes_from_others;
    }
    ++galois::runtime::evilPhase;

    assert(numGlobalNodes == global_total_owned_nodes);
    // report stats
    if (net.ID == 0) {
      report_master_mirror_stats(global_total_mirror_nodes, 
                                 global_total_owned_nodes);
    }
  }

  /**
   * Given a bitset, determine the indices of the bitset that are currently
   * set.
   *
   * @tparam syncType either reduce or broadcast; only used to name the timer
   * @param loopName string used to name the timer for this function
   * @param bitset_comm the bitset to get the offsets of
   * @param offsets output: the offset vector that will contain indices into
   * the bitset that are set
   * @param bit_set_count output: will be set to the number of bits set in the
   * bitset
   */
  template<SyncType syncType>
  void get_offsets_from_bitset(const std::string &loopName, 
                               const galois::DynamicBitSet &bitset_comm, 
                               std::vector<unsigned int> &offsets, 
                               size_t &bit_set_count) const {
    // timer creation
    std::string syncTypeStr = (syncType == syncReduce) ? "REDUCE" : "BROADCAST";
    std::string offsets_timer_str(syncTypeStr + "_OFFSETS_" + 
                                  get_run_identifier(loopName));
    galois::StatTimer Toffsets(offsets_timer_str.c_str(), GRNAME);

    Toffsets.start();


    auto activeThreads = galois::getActiveThreads();
    std::vector<unsigned int> t_prefix_bit_counts(activeThreads);

    // count how many bits are set on each thread
    galois::on_each(
      [&](unsigned tid, unsigned nthreads) {
        // TODO use block_range instead
        unsigned int block_size = bitset_comm.size() / nthreads;
        if ((bitset_comm.size() % nthreads) > 0) ++block_size;
        assert((block_size * nthreads) >= bitset_comm.size());

        unsigned int start = tid * block_size;
        unsigned int end = (tid + 1) * block_size;
        if (end > bitset_comm.size()) end = bitset_comm.size();

        unsigned int count = 0;
        for (unsigned int i = start; i < end; ++i) {
          if (bitset_comm.test(i)) ++count;
        }

        t_prefix_bit_counts[tid] = count;
      }
    );

    // calculate prefix sum of bits per thread
    for (unsigned int i = 1; i < activeThreads; ++i) {
      t_prefix_bit_counts[i] += t_prefix_bit_counts[i - 1];
    }
    // total num of set bits
    bit_set_count = t_prefix_bit_counts[activeThreads - 1];

    // calculate the indices of the set bits and save them to the offset
    // vector
    if (bit_set_count > 0) {
      offsets.resize(bit_set_count);
      galois::on_each(
        [&](unsigned tid, unsigned nthreads) {
          // TODO use block_range instead
          // TODO this is same calculation as above; maybe refactor it
          // into function?
          unsigned int block_size = bitset_comm.size() / nthreads;
          if ((bitset_comm.size() % nthreads) > 0) ++block_size;
          assert((block_size * nthreads) >= bitset_comm.size());

          unsigned int start = tid*block_size;
          unsigned int end = (tid+1)*block_size;
          if (end > bitset_comm.size()) end = bitset_comm.size();

          unsigned int count = 0;
          unsigned int t_prefix_bit_count;
          if (tid == 0) {
            t_prefix_bit_count = 0;
          } else {
            t_prefix_bit_count = t_prefix_bit_counts[tid - 1];
          }

          for (unsigned int i = start; i < end; ++i) {
            if (bitset_comm.test(i)) {
              offsets[t_prefix_bit_count + count] = i;
              ++count;
            }
          }
        }
      );
    }
    Toffsets.stop();
  }

  /**
   * TODO
   *
   * @tparam FnTy structure that specifies how synchronization is to be done;
   * only used to get the size of the type being synchronized in this function
   * @tparam syncType type of synchronization this function is being called 
   * for; only used to name a timer
   *
   * @param loopName loopname used to name the timer for the function
   * @param indices A vector that contains the local ids of the nodes that
   * you want to potentially synchronize
   * @param bitset_compute Contains the full bitset of all nodes in this
   * graph
   * @param bitset_comm OUTPUT: bitset that marks which indices in the passed
   * in indices array need to be synchronized
   * @param offsets OUTPUT: contains indices into bitset_comm that are set 
   * @param bit_set_count OUTPUT: contains number of bits set in bitset_comm
   * @param data_mode TODO
   */
  template<typename FnTy, SyncType syncType>
  void get_bitset_and_offsets(const std::string &loopName,
                              const std::vector<size_t> &indices,
                              const galois::DynamicBitSet &bitset_compute,
                              galois::DynamicBitSet &bitset_comm,
                              std::vector<unsigned int> &offsets,
                              size_t &bit_set_count, DataCommMode &data_mode) const {
    if (enforce_data_mode != onlyData) {
      bitset_comm.reset();
      std::string syncTypeStr = (syncType == syncReduce) ? "REDUCE" : "BROADCAST";
      std::string doall_str(syncTypeStr + "_BITSET_" + loopName);

      // determine which local nodes in the indices array need to be 
      // sychronized
      galois::do_all(galois::iterate(0ul, indices.size()),
                     [&](unsigned int n) {
                       // assumes each lid is unique as test is not thread safe
                       size_t lid = indices[n]; 
                       if (bitset_compute.test(lid)) {
                         bitset_comm.set(n);
                       }
                     },
                     galois::loopname(get_run_identifier(doall_str).c_str()),
                     galois::timeit(),
                     galois::steal<false>(),
                     galois::no_stats());

      // get the number of set bits and the offsets into the comm bitset
      get_offsets_from_bitset<syncType>(loopName, bitset_comm, offsets,
                                        bit_set_count);
    }

    data_mode = get_data_mode<typename FnTy::ValTy>(bit_set_count, indices.size());
  }

  /**
   * TODO
   */
  /* Reduction extract resets the value afterwards */
  template<typename FnTy, SyncType syncType,
           typename std::enable_if<syncType == syncReduce>::type* = nullptr>
  inline typename FnTy::ValTy extract_wrapper(size_t lid) {
    #ifdef __GALOIS_HET_OPENCL__
    CLNodeDataWrapper d = clGraph.getDataW(lid);
    auto val = FnTy::extract(lid, getData(lid, d));
    FnTy::reset(lid, d);
    #else
    auto val = FnTy::extract(lid, getData(lid));
    FnTy::reset(lid, getData(lid));
    #endif
    return val;
  }

  /**
   * TODO
   */
  /* Broadcast extract doesn't reset the value */
  template<typename FnTy, SyncType syncType,
           typename std::enable_if<syncType == syncBroadcast>::type* = nullptr>
  inline typename FnTy::ValTy extract_wrapper(size_t lid) {
    #ifdef __GALOIS_HET_OPENCL__
    CLNodeDataWrapper d = clGraph.getDataW(lid);
    return FnTy::extract(lid, getData(lid, d));
    #else
    return FnTy::extract(lid, getData(lid));
    #endif
  }

  /**
   * TODO
   */
  template<typename FnTy, SyncType syncType, bool identity_offsets = false, 
           bool parallelize = true>
  void extract_subset(const std::string &loopName,
                      const std::vector<size_t> &indices, size_t size,
                      const std::vector<unsigned int> &offsets,
                      std::vector<typename FnTy::ValTy> &val_vec,
                      size_t start = 0) {
    if (parallelize) {
      std::string syncTypeStr = (syncType == syncReduce) ? "REDUCE" : "BROADCAST";
      std::string doall_str(syncTypeStr + "_EXTRACTVAL_" + loopName);
      galois::do_all(galois::iterate(start, start + size),
                     [&](unsigned int n){
                       unsigned int offset;
                       if (identity_offsets) offset = n;
                       else offset = offsets[n];
                       size_t lid = indices[offset];
                       val_vec[n - start] = extract_wrapper<FnTy, syncType>(lid);
                     },
                     galois::loopname(get_run_identifier(doall_str).c_str()),
                     galois::timeit(),
                     galois::steal<false>(),
                     galois::no_stats());
    } else {
      for (unsigned n = start; n < start + size; ++n) {
        unsigned int offset;
        if (identity_offsets) offset = n;
        else offset = offsets[n];

        size_t lid = indices[offset];
        val_vec[n - start] = extract_wrapper<FnTy, syncType>(lid);
      }
    }
  }

  /**
   * TODO
   */
  template<typename FnTy, typename SeqTy, SyncType syncType, 
           bool identity_offsets = false, bool parallelize = true>
  void extract_subset(const std::string &loopName,
                      const std::vector<size_t> &indices, size_t size,
                      const std::vector<unsigned int> &offsets,
                      galois::runtime::SendBuffer& b, SeqTy lseq,
                      size_t start = 0) {
    if (parallelize) {
      std::string syncTypeStr = (syncType == syncReduce) ? "REDUCE" : "BROADCAST";
      std::string doall_str(syncTypeStr + "_EXTRACTVAL_" + loopName);

      galois::do_all(galois::iterate(start, start + size), 
          [&](unsigned int n) {
            unsigned int offset;

            if (identity_offsets) offset = n;
            else offset = offsets[n];

            size_t lid = indices[offset];
            gSerializeLazy(b, lseq, n-start, extract_wrapper<FnTy, syncType>(lid));
          }, 
          galois::loopname(get_run_identifier(doall_str).c_str()),
          galois::steal<false>(),
          galois::timeit(),
          galois::no_stats());
    } else {
      for (unsigned int n = start; n < start + size; ++n) {
        unsigned int offset;

        if (identity_offsets) offset = n;
        else offset = offsets[n];

        size_t lid = indices[offset];
        gSerializeLazy(b, lseq, n-start, extract_wrapper<FnTy, syncType>(lid));
      }
    }
  }

  /**
   * TODO
   */
  template<typename FnTy, SyncType syncType, 
           typename std::enable_if<syncType == syncReduce>::type* = nullptr>
  inline bool extract_batch_wrapper(unsigned x, std::vector<typename FnTy::ValTy> &v) {
    return FnTy::extract_reset_batch(x, v.data());
  }

  /**
   * TODO
   */
  template<typename FnTy, SyncType syncType, 
           typename std::enable_if<syncType == syncBroadcast>::type* = nullptr>
  inline bool extract_batch_wrapper(unsigned x, std::vector<typename FnTy::ValTy> &v) {
    return FnTy::extract_batch(x, v.data());
  }

  /**
   * TODO
   */
  template<typename FnTy, SyncType syncType, 
           typename std::enable_if<syncType == syncReduce>::type* = nullptr>
  inline bool extract_batch_wrapper(unsigned x, galois::DynamicBitSet &b, 
                             std::vector<unsigned int> &o, 
                             std::vector<typename FnTy::ValTy> &v, 
                             size_t &s, DataCommMode& data_mode) {
    return FnTy::extract_reset_batch(x, 
        (unsigned long long int *)b.get_vec().data(), o.data(), v.data(), &s, 
        &data_mode);
  }

  /**
   * TODO
   */
  template<typename FnTy, SyncType syncType, 
           typename std::enable_if<syncType == syncBroadcast>::type* = nullptr>
  inline bool extract_batch_wrapper(unsigned x, galois::DynamicBitSet &b, 
                             std::vector<unsigned int> &o, 
                             std::vector<typename FnTy::ValTy> &v, 
                             size_t &s, DataCommMode& data_mode) const {
    return FnTy::extract_batch(x, (unsigned long long int *)b.get_vec().data(), 
                               o.data(), v.data(), &s, &data_mode);
  }

  /**
   * TODO
   */
  template<typename FnTy, SyncType syncType, 
           typename std::enable_if<syncType == syncReduce>::type* = nullptr>
  inline void set_wrapper(size_t lid, typename FnTy::ValTy val, 
                   galois::DynamicBitSet& bit_set_compute) {
    #ifdef __GALOIS_HET_OPENCL__
    CLNodeDataWrapper d = clGraph.getDataW(lid);
    FnTy::reduce(lid, d, val);
    #else
    if (FnTy::reduce(lid, getData(lid), val)) {
      if (bit_set_compute.size() != 0) bit_set_compute.set(lid);
    }
    #endif
  }

  /**
   * TODO
   */
  template<typename FnTy, SyncType syncType, 
           typename std::enable_if<syncType == syncBroadcast>::type* = nullptr>
  inline void set_wrapper(size_t lid, typename FnTy::ValTy val, 
                   galois::DynamicBitSet& bit_set_compute) {
    #ifdef __GALOIS_HET_OPENCL__
    CLNodeDataWrapper d = clGraph.getDataW(lid);
    FnTy::setVal(lid, d, val_vec[n]);
    #else
    FnTy::setVal(lid, getData(lid), val);
    #endif
  }

  /**
   * TODO
   */
  template<typename FnTy, SyncType syncType, bool identity_offsets = false, 
           bool parallelize = true>
  void set_subset(const std::string &loopName, 
                  const std::vector<size_t> &indices, 
                  size_t size, 
                  const std::vector<unsigned int> &offsets, 
                  std::vector<typename FnTy::ValTy> &val_vec, 
                  galois::DynamicBitSet& bit_set_compute, 
                  size_t start = 0) {
    std::string syncTypeStr = (syncType == syncReduce) ? "REDUCE" : "BROADCAST";
    std::string doall_str(syncTypeStr + "_SETVAL_" + 
                          get_run_identifier(loopName));
    if (parallelize) {
      galois::do_all(galois::iterate(start, start + size), 
          [&](unsigned int n) {
            unsigned int offset;

            if (identity_offsets) offset = n;
            else offset = offsets[n];

            size_t lid = indices[offset];
            set_wrapper<FnTy, syncType>(lid, val_vec[n - start], bit_set_compute);
          }, 
          galois::loopname(get_run_identifier(doall_str).c_str()),
          galois::steal<false>(),
          galois::timeit(),
          galois::no_stats());
    } else {
      for (unsigned int n = start; n < start + size; ++n) {
        unsigned int offset;

        if (identity_offsets) offset = n;
        else offset = offsets[n];

        size_t lid = indices[offset];
        set_wrapper<FnTy, syncType>(lid, val_vec[n - start], bit_set_compute);
      }
    }
  }

  /**
   * TODO
   */
  template<typename FnTy, SyncType syncType, 
           typename std::enable_if<syncType == syncReduce>::type* = nullptr>
  inline bool set_batch_wrapper(unsigned x, std::vector<typename FnTy::ValTy> &v) {
    return FnTy::reduce_batch(x, v.data());
  }

  /**
   * TODO
   */
  template<typename FnTy, SyncType syncType, 
           typename std::enable_if<syncType == syncBroadcast>::type* = nullptr>
  inline bool set_batch_wrapper(unsigned x, std::vector<typename FnTy::ValTy> &v) {
    return FnTy::setVal_batch(x, v.data());
  }
  /**
   * TODO
   */
  template<typename FnTy, SyncType syncType, 
           typename std::enable_if<syncType == syncReduce>::type* = nullptr>
  inline bool set_batch_wrapper(unsigned x, galois::DynamicBitSet &b, 
                         std::vector<unsigned int> &o, 
                         std::vector<typename FnTy::ValTy> &v, size_t &s, 
                         DataCommMode& data_mode) {
    return FnTy::reduce_batch(x, (unsigned long long int *)b.get_vec().data(), 
                              o.data(), v.data(), s, data_mode);
  }
  
  /**
   * TODO
   */
  template<typename FnTy, SyncType syncType, 
           typename std::enable_if<syncType == syncBroadcast>::type* = nullptr>
  inline bool set_batch_wrapper(unsigned x, galois::DynamicBitSet &b, 
                         std::vector<unsigned int> &o, 
                         std::vector<typename FnTy::ValTy> &v, size_t &s, 
                         DataCommMode& data_mode) {
    return FnTy::setVal_batch(x, (unsigned long long int *)b.get_vec().data(), 
                              o.data(), v.data(), s, data_mode);
  }
  
  /**
   * TODO
   */
  template<SyncType syncType>
  void convert_lid_to_gid(const std::string &loopName, 
                          std::vector<unsigned int> &offsets) {
    std::string syncTypeStr = (syncType == syncReduce) ? "REDUCE" : "BROADCAST";
    std::string doall_str(syncTypeStr + "_LID2GID_" + 
                          get_run_identifier(loopName));
    galois::do_all(galois::iterate(0ul, offsets.size()), 
        [&](unsigned int n) {
          offsets[n] = static_cast<uint32_t>(getGID(offsets[n]));
        }, 
        galois::loopname(get_run_identifier(doall_str).c_str()), 
        galois::timeit(),
        galois::steal<false>(),
        galois::no_stats());
  }
  
  /**
   * TODO
   */
  template<SyncType syncType>
  void convert_gid_to_lid(const std::string &loopName, 
                          std::vector<unsigned int> &offsets) {
    std::string syncTypeStr = (syncType == syncReduce) ? "REDUCE" : "BROADCAST";
    std::string doall_str(syncTypeStr + "_GID2LID_" + 
                          get_run_identifier(loopName));

    galois::do_all(galois::iterate(0ul, offsets.size()), 
        [&](unsigned int n) {
          offsets[n] = static_cast<uint32_t>(getLID(offsets[n]));
        }, 
        galois::loopname(get_run_identifier(doall_str).c_str()), 
        galois::steal<false>(),
        galois::timeit(),
        galois::no_stats());
  }
  
  /**
   * TODO
   */
  template<SyncType syncType, typename SyncFnTy>
  void sync_extract(std::string loopName, unsigned from_id, 
                    std::vector<size_t> &indices, galois::runtime::SendBuffer &b) {
    uint32_t num = indices.size();
    static std::vector<typename SyncFnTy::ValTy> val_vec; // sometimes wasteful
    static std::vector<unsigned int> offsets;
    std::string syncTypeStr = (syncType == syncReduce) ? "REDUCE" : "BROADCAST";
    std::string extract_timer_str(syncTypeStr + "_EXTRACT_" + 
                                  get_run_identifier(loopName));
    galois::StatTimer Textract(extract_timer_str.c_str(), GRNAME);

    DataCommMode data_mode;

    Textract.start();

    if (num > 0) {
      data_mode = onlyData;
      val_vec.resize(num);

      bool batch_succeeded = extract_batch_wrapper<SyncFnTy, syncType>(from_id, 
                                                                       val_vec);

      if (!batch_succeeded) {
        gSerialize(b, onlyData);
        auto lseq = gSerializeLazySeq(b, num, 
                        (std::vector<typename SyncFnTy::ValTy>*)nullptr);
        extract_subset<SyncFnTy, decltype(lseq), syncType, true, true>(loopName, 
            indices, num, offsets, b, lseq);
      } else {
        gSerialize(b, onlyData, val_vec);
      }
    } else {
      data_mode = noData;
      gSerialize(b, noData);
    }

    Textract.stop();

    std::string metadata_str(syncTypeStr + "_METADATA_MODE" + 
                             std::to_string(data_mode) + "_" + 
                             get_run_identifier(loopName));
    galois::runtime::reportStat_Single(GRNAME, metadata_str, 1);
  }
  
  /**
   * TODO
   */
  // Bitset variant (uses bitset to determine what to sync)
  template<SyncType syncType, typename SyncFnTy, typename BitsetFnTy>
  void sync_extract(std::string loopName, unsigned from_id,
                    std::vector<size_t> &indices,
                    galois::runtime::SendBuffer &b) {
    uint32_t num = indices.size();
    static galois::DynamicBitSet bit_set_comm;
    static std::vector<typename SyncFnTy::ValTy> val_vec;
    static std::vector<unsigned int> offsets;

    std::string syncTypeStr = (syncType == syncReduce) ? "REDUCE" : "BROADCAST";
    std::string extract_timer_str(syncTypeStr + "_EXTRACT_" + 
                                  get_run_identifier(loopName));
    galois::StatTimer Textract(extract_timer_str.c_str(), GRNAME);

    DataCommMode data_mode;

    Textract.start();

    if (num > 0) {
      bit_set_comm.resize(num);
      offsets.resize(num);
      val_vec.resize(num);
      size_t bit_set_count = 0;

      bool batch_succeeded = extract_batch_wrapper<SyncFnTy, syncType>(from_id,
                               bit_set_comm, offsets, val_vec, bit_set_count,
                               data_mode);

      // GPUs have a batch function they can use; CPUs do not; therefore, 
      // CPUS always enter this if block
      if (!batch_succeeded) {
        const galois::DynamicBitSet &bit_set_compute = BitsetFnTy::get();

        get_bitset_and_offsets<SyncFnTy, syncType>(loopName, indices,
                   bit_set_compute, bit_set_comm, offsets, bit_set_count,
                   data_mode);

        // at this point indices should hold local ids of nodes that need
        // to be accessed
        if (data_mode == onlyData) {
          bit_set_count = indices.size();
          extract_subset<SyncFnTy, syncType, true, true>(loopName, indices,
            bit_set_count, offsets, val_vec);
        } else if (data_mode != noData) { // bitsetData or offsetsData
          extract_subset<SyncFnTy, syncType, false, true>(loopName, indices,
            bit_set_count, offsets, val_vec);
        }
      }

      size_t redundant_size = (num - bit_set_count) *
                                sizeof(typename SyncFnTy::ValTy);
      size_t bit_set_size = (bit_set_comm.get_vec().size() * sizeof(uint64_t));

      if (redundant_size > bit_set_size) {
        std::string statSavedBytes_str(syncTypeStr + "_SAVED_BYTES_" + 
                                       get_run_identifier(loopName));
                                     
        galois::runtime::reportStat_Tsum(GRNAME, statSavedBytes_str, 
                                         (redundant_size - bit_set_size));
      }

      if (data_mode == noData) {
        gSerialize(b, data_mode);
      } else if (data_mode == offsetsData) {
        offsets.resize(bit_set_count);

        if (useGidMetadata) {
          convert_lid_to_gid<syncType>(loopName, offsets);
        }

        val_vec.resize(bit_set_count);
        gSerialize(b, data_mode, bit_set_count, offsets, val_vec);
      } else if (data_mode == bitsetData) {
        val_vec.resize(bit_set_count);
        gSerialize(b, data_mode, bit_set_count, bit_set_comm, val_vec);
      } else { // onlyData
        gSerialize(b, data_mode, val_vec);
      }
    } else {
      data_mode = noData;
      gSerialize(b, noData);
    }

    Textract.stop();

    std::string metadata_str(syncTypeStr + "_METADATA_MODE" + 
                             std::to_string(data_mode) + 
                             get_run_identifier(loopName));
    galois::runtime::reportStat_Single(GRNAME, metadata_str, 1);
  }
  
  /**
   * TODO
   */
  template<SyncType syncType, typename SyncFnTy, typename BitsetFnTy>
  void get_send_buffer(std::string loopName, unsigned x, galois::runtime::SendBuffer &b) {
    auto& sharedNodes = (syncType == syncReduce) ? mirrorNodes : masterNodes;

    if (BitsetFnTy::is_valid()) {
      sync_extract<syncType, SyncFnTy, BitsetFnTy>(loopName, x,
                                                   sharedNodes[x], b);
    } else {
      sync_extract<syncType, SyncFnTy>(loopName, x, sharedNodes[x], b);
    }

    std::string syncTypeStr = (syncType == syncReduce) ? "REDUCE" : "BROADCAST";
    std::string statSendBytes_str(syncTypeStr + "_SEND_BYTES_" + 
                                  get_run_identifier(loopName));

    galois::runtime::reportStat_Tsum(GRNAME, statSendBytes_str, b.size());
  }
  
#ifdef __GALOIS_BARE_MPI_COMMUNICATION__
  /**
   * TODO
   */
  template<WriteLocation writeLocation, ReadLocation readLocation,
           SyncType syncType, typename SyncFnTy, typename BitsetFnTy>
  void sync_mpi_send(std::string loopName) {
    static std::vector<galois::runtime::SendBuffer> b;
    static std::vector<MPI_Request> request;
    b.resize(numHosts);
    request.resize(numHosts, MPI_REQUEST_NULL);

    for (unsigned h = 1; h < numHosts; ++h) {
      unsigned x = (id + h) % numHosts;

      if (nothingToSend(x, syncType, writeLocation, readLocation)) continue;

      int ready = 0;
      MPI_Test(&request[x], &ready, MPI_STATUS_IGNORE);
      if (!ready) {
        assert(b[x].size() > 0);
        MPI_Wait(&request[x], MPI_STATUS_IGNORE);
      }
      if (b[x].size() > 0) {
        b[x].getVec().clear();
      }

      get_send_buffer<syncType, SyncFnTy, BitsetFnTy>(loopName, x, b[x]);

      MPI_Isend((uint8_t *)b[x].linearData(), b[x].size(), MPI_BYTE, x, 32767, 
                MPI_COMM_WORLD, &request[x]);
    }

    if (BitsetFnTy::is_valid()) {
      reset_bitset(syncType, &BitsetFnTy::reset_range);
    }
  }

  /**
   * TODO
   */
  template<WriteLocation writeLocation, ReadLocation readLocation,
           SyncType syncType, typename SyncFnTy, typename BitsetFnTy>
  void sync_mpi_put(std::string loopName,
      const std::vector<MPI_Win>& window) {
    for (unsigned h = 1; h < numHosts; ++h) {
      unsigned x = (id + h) % numHosts;

      if (nothingToSend(x, syncType, writeLocation, readLocation)) continue;

      galois::runtime::SendBuffer b;

      get_send_buffer<syncType, SyncFnTy, BitsetFnTy>(loopName, x, b);

      MPI_Win_start(mpi_identity_groups[x], 0, window[id]);
      size_t size = b.size();
      MPI_Put((uint8_t *)&size, sizeof(size_t), MPI_BYTE, 
          x, 0, sizeof(size_t), MPI_BYTE,
          window[id]);
      MPI_Put((uint8_t *)b.linearData(), size, MPI_BYTE, 
          x, sizeof(size_t), size, MPI_BYTE,
          window[id]);
      MPI_Win_complete(window[id]);
    }

    if (BitsetFnTy::is_valid()) {
      reset_bitset(syncType, &BitsetFnTy::reset_range);
    }
  }
#endif
  
  /**
   * TODO
   */
  template<WriteLocation writeLocation, ReadLocation readLocation,
           SyncType syncType, typename SyncFnTy, typename BitsetFnTy>
  void sync_net_send(std::string loopName) {
    auto& net = galois::runtime::getSystemNetworkInterface();

    for (unsigned h = 1; h < numHosts; ++h) {
      unsigned x = (id + h) % numHosts;

      if (nothingToSend(x, syncType, writeLocation, readLocation)) continue;

      galois::runtime::SendBuffer b;

      get_send_buffer<syncType, SyncFnTy, BitsetFnTy>(loopName, x, b);

      net.sendTagged(x, galois::runtime::evilPhase, b);
    }
    // Will force all messages to be processed before continuing
    net.flush();

    if (BitsetFnTy::is_valid()) {
      reset_bitset(syncType, &BitsetFnTy::reset_range);
    }
  }
  
  /**
   * TODO
   */
  template<WriteLocation writeLocation, ReadLocation readLocation,
           SyncType syncType, typename SyncFnTy, typename BitsetFnTy>
  void sync_send(std::string loopName) {
    std::string syncTypeStr = (syncType == syncReduce) ? "REDUCE" : "BROADCAST";
    galois::StatTimer TSendTime((syncTypeStr + "_SEND_" + 
                                         get_run_identifier(loopName)).c_str(), GRNAME);

    TSendTime.start();
    sync_net_send<writeLocation, readLocation, syncType, SyncFnTy, BitsetFnTy>(loopName);
    TSendTime.stop();
  }
  
  /**
   * TODO
   */
  template<SyncType syncType, typename SyncFnTy, typename BitsetFnTy,
           bool parallelize = true>
  size_t syncRecvApply(uint32_t from_id, galois::runtime::RecvBuffer& buf,
                       std::string loopName) {
    std::string syncTypeStr = (syncType == syncReduce) ? "REDUCE" : "BROADCAST";
    std::string set_timer_str(syncTypeStr + "_SET_" + 
                              get_run_identifier(loopName));
    galois::StatTimer Tset(set_timer_str.c_str(), GRNAME);

    Tset.start();

    static galois::DynamicBitSet bit_set_comm;
    static std::vector<typename SyncFnTy::ValTy> val_vec;
    static std::vector<unsigned int> offsets;
    auto& sharedNodes = (syncType == syncReduce) ? masterNodes : mirrorNodes;

    uint32_t num = sharedNodes[from_id].size();
    size_t buf_start = 0;
    size_t retval = 0;

    if (num > 0) {
      DataCommMode data_mode;
      galois::runtime::gDeserialize(buf, data_mode);

      if (data_mode != noData) {
        size_t bit_set_count = num;

        if (data_mode != onlyData) {
          galois::runtime::gDeserialize(buf, bit_set_count);

          if (data_mode == offsetsData) {
            //offsets.resize(bit_set_count);
            galois::runtime::gDeserialize(buf, offsets);
            if (useGidMetadata) {
              convert_gid_to_lid<syncType>(loopName, offsets);
            }
          } else if (data_mode == bitsetData) {
            bit_set_comm.resize(num);
            galois::runtime::gDeserialize(buf, bit_set_comm);
          } else if (data_mode == dataSplit) {
            galois::runtime::gDeserialize(buf, buf_start);
          } else if (data_mode == dataSplitFirst) {
            galois::runtime::gDeserialize(buf, retval);
          }
        }

        //val_vec.resize(bit_set_count);
        galois::runtime::gDeserialize(buf, val_vec);

        bool batch_succeeded = set_batch_wrapper<SyncFnTy, syncType>(from_id, 
                                   bit_set_comm, offsets, val_vec, 
                                   bit_set_count, data_mode);

        if (!batch_succeeded) {
          galois::DynamicBitSet &bit_set_compute = BitsetFnTy::get();

          if (data_mode == bitsetData) {
            size_t bit_set_count2;
            get_offsets_from_bitset<syncType>(loopName, bit_set_comm, offsets, bit_set_count2);
            assert(bit_set_count ==  bit_set_count2);
          }

          if (data_mode == onlyData) {
            set_subset<SyncFnTy, syncType, true, true>(loopName, 
                sharedNodes[from_id], bit_set_count, offsets, val_vec, 
                bit_set_compute);
          } else if (data_mode == dataSplit || data_mode == dataSplitFirst) {
            set_subset<SyncFnTy, syncType, true, true>(loopName, 
                sharedNodes[from_id], bit_set_count, offsets, val_vec, 
                bit_set_compute, buf_start);
          } else {
            set_subset<SyncFnTy, syncType, false, true>(loopName, 
                sharedNodes[from_id], bit_set_count, offsets, val_vec, 
                bit_set_compute);
          }
          // TODO: reduce could update the bitset, so it needs to be copied 
          // back to the device
        }
      }
    }

    Tset.stop();

    return retval;
  }
  
#ifdef __GALOIS_BARE_MPI_COMMUNICATION__
  /**
   * TODO
   */
  template<WriteLocation writeLocation, ReadLocation readLocation,
           SyncType syncType, typename SyncFnTy, typename BitsetFnTy>
  void sync_mpi_recv_post(std::string loopName,
      std::vector<MPI_Request>& request,
      const std::vector<std::vector<uint8_t>>& rb) {
    for (unsigned h = 1; h < numHosts; ++h) {
      unsigned x = (id + numHosts - h) % numHosts;
      if (nothingToRecv(x, syncType, writeLocation, readLocation)) continue;

      MPI_Irecv((uint8_t *)rb[x].data(), rb[x].size(), MPI_BYTE, x, 32767, 
                MPI_COMM_WORLD, &request[x]);
    }
  }

  /**
   * TODO
   */
  template<WriteLocation writeLocation, ReadLocation readLocation,
           SyncType syncType, typename SyncFnTy, typename BitsetFnTy>
  void sync_mpi_recv_wait(std::string loopName,
      std::vector<MPI_Request>& request,
      const std::vector<std::vector<uint8_t>>& rb) {
    for (unsigned h = 1; h < numHosts; ++h) {
      unsigned x = (id + numHosts - h) % numHosts;
      if (nothingToRecv(x, syncType, writeLocation, readLocation)) continue;

      MPI_Status status;
      MPI_Wait(&request[x], &status);

      int size = 0;
      MPI_Get_count(&status, MPI_BYTE, &size);

      galois::runtime::RecvBuffer rbuf(rb[x].begin(), 
          rb[x].begin() + size);

      syncRecvApply<syncType, SyncFnTy, BitsetFnTy>(x, rbuf, loopName);
    }
  }
  
  /**
   * TODO
   */
  template<WriteLocation writeLocation, ReadLocation readLocation,
           SyncType syncType, typename SyncFnTy, typename BitsetFnTy>
  void sync_mpi_get(std::string loopName,
      const std::vector<MPI_Win>& window,
      const std::vector<std::vector<uint8_t>>& rb) {
    for (unsigned h = 1; h < numHosts; ++h) {
      unsigned x = (id + numHosts - h) % numHosts;
      if (nothingToRecv(x, syncType, writeLocation, readLocation)) continue;

      MPI_Win_wait(window[x]);

      size_t size = 0;
      memcpy(&size, rb[x].data(), sizeof(size_t));

      galois::runtime::RecvBuffer rbuf(rb[x].begin() + sizeof(size_t), 
          rb[x].begin() + sizeof(size_t) + size);

      syncRecvApply<syncType, SyncFnTy, BitsetFnTy>(x, rbuf, loopName);

      MPI_Win_post(mpi_identity_groups[x], 0, window[x]);
    }
  }
#endif
  
  /**
   * TODO
   */
  template<WriteLocation writeLocation, ReadLocation readLocation,
           SyncType syncType, typename SyncFnTy, typename BitsetFnTy>
  void sync_net_recv(std::string loopName) {
    auto& net = galois::runtime::getSystemNetworkInterface();

    for (unsigned x = 0; x < numHosts; ++x) {
      if (x == id) continue;
      if (nothingToRecv(x, syncType, writeLocation, readLocation)) continue;

      decltype(net.recieveTagged(galois::runtime::evilPhase,nullptr)) p;
      do {
        net.handleReceives();
        p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
      } while (!p);

      syncRecvApply<syncType, SyncFnTy, BitsetFnTy>(p->first, p->second, 
                                                    loopName);
    }
    ++galois::runtime::evilPhase;
  }
  
  /**
   * TODO
   */
  template<WriteLocation writeLocation, ReadLocation readLocation,
           SyncType syncType, typename SyncFnTy, typename BitsetFnTy>
  void sync_recv(std::string loopName) {
    std::string syncTypeStr = (syncType == syncReduce) ? "REDUCE" : "BROADCAST";
    galois::StatTimer TRecvTime((syncTypeStr + "_RECV_" + 
                                         get_run_identifier(loopName)).c_str(), GRNAME);

    TRecvTime.start();
    sync_net_recv<writeLocation, readLocation, syncType, SyncFnTy, BitsetFnTy>(loopName);
    TRecvTime.stop();
  }
  
#ifdef __GALOIS_BARE_MPI_COMMUNICATION__
  /**
   * TODO
   */
  template<WriteLocation writeLocation, ReadLocation readLocation,
           SyncType syncType, typename SyncFnTy, typename BitsetFnTy>
  void sync_nonblocking_mpi(std::string loopName, bool use_bitset_to_send = true) {
    std::string syncTypeStr = (syncType == syncReduce) ? "REDUCE" : "BROADCAST";
    galois::StatTimer TSendTime((syncTypeStr + "_SEND_" + 
                                         get_run_identifier(loopName)).c_str(), GRNAME);
    galois::StatTimer TRecvTime((syncTypeStr + "_RECV_" + 
                                         get_run_identifier(loopName)).c_str(), GRNAME);

    static std::vector<std::vector<uint8_t>> rb;
    static std::vector<MPI_Request> request;

    if (rb.size() == 0) { // create the receive buffers
      TRecvTime.start();
      auto& sharedNodes = (syncType == syncReduce) ? masterNodes : mirrorNodes;
      rb.resize(numHosts);
      request.resize(numHosts, MPI_REQUEST_NULL);

      for (unsigned h = 1; h < numHosts; ++h) {
        unsigned x = (id + numHosts - h) % numHosts;
        if (nothingToRecv(x, syncType, writeLocation, readLocation)) continue;

        size_t size = (sharedNodes[x].size() * sizeof(typename SyncFnTy::ValTy));
        size += sizeof(size_t); // vector size
        size += sizeof(DataCommMode); // data mode

        rb[x].resize(size);
      }
      TRecvTime.stop();
    }

    TRecvTime.start();
    sync_mpi_recv_post<writeLocation, readLocation, syncType, SyncFnTy, 
      BitsetFnTy>(loopName, request, rb);
    TRecvTime.stop();

    TSendTime.start();
    if (use_bitset_to_send) {
      sync_mpi_send<writeLocation, readLocation, syncType, SyncFnTy, 
        BitsetFnTy>(loopName);
    } else {
      sync_mpi_send<writeLocation, readLocation, syncType, SyncFnTy, 
        galois::InvalidBitsetFnTy>(loopName);
    }
    TSendTime.stop();

    TRecvTime.start();
    sync_mpi_recv_wait<writeLocation, readLocation, syncType, SyncFnTy, 
      BitsetFnTy>(loopName, request, rb);
    TRecvTime.stop();
  }

  /**
   * TODO
   */
  template<WriteLocation writeLocation, ReadLocation readLocation,
           SyncType syncType, typename SyncFnTy, typename BitsetFnTy>
  void sync_onesided_mpi(std::string loopName, bool use_bitset_to_send = true) {
    std::string syncTypeStr = (syncType == syncReduce) ? "REDUCE" : "BROADCAST";
    galois::StatTimer TSendTime((syncTypeStr + "_SEND_" + 
                                         get_run_identifier(loopName)).c_str(), GRNAME);
    galois::StatTimer TRecvTime((syncTypeStr + "_RECV_" + 
                                         get_run_identifier(loopName)).c_str(), GRNAME);

    static std::vector<MPI_Win> window;
    static std::vector<std::vector<uint8_t>> rb;

    if (window.size() == 0) { // create the windows
      TRecvTime.start();
      auto& sharedNodes = (syncType == syncReduce) ? masterNodes : mirrorNodes;
      window.resize(numHosts);
      rb.resize(numHosts);

      for (unsigned x = 0; x < numHosts; ++x) {
        size_t size = (sharedNodes[x].size() * sizeof(typename SyncFnTy::ValTy));
        size += sizeof(size_t); // vector size
        size += sizeof(DataCommMode); // data mode
        size += sizeof(size_t); // buffer size

        rb[x].resize(size);

        // TODO add no_locks and same_disp_unit
        MPI_Win_create(rb[x].data(), size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &window[x]);
      }

      for (unsigned h = 1; h < numHosts; ++h) {
        unsigned x = (id + numHosts - h) % numHosts;
        if (nothingToRecv(x, syncType, writeLocation, readLocation)) continue;
        MPI_Win_post(mpi_identity_groups[x], 0, window[x]);
      }
      TRecvTime.stop();
    }

    TSendTime.start();
    if (use_bitset_to_send) {
      sync_mpi_put<writeLocation, readLocation, syncType, SyncFnTy, 
        BitsetFnTy>(loopName, window);
    } else {
      sync_mpi_put<writeLocation, readLocation, syncType, SyncFnTy, 
        galois::InvalidBitsetFnTy>(loopName, window);
    }
    TSendTime.stop();

    TRecvTime.start();
    sync_mpi_get<writeLocation, readLocation, syncType, SyncFnTy, 
      BitsetFnTy>(loopName, window, rb);
    TRecvTime.stop();
  }
#endif
  
  /**
   * TODO
   */
  // reduce from mirrors to master
  template<WriteLocation writeLocation, ReadLocation readLocation,
           typename ReduceFnTy, typename BitsetFnTy>
  inline void reduce(std::string loopName) {
    std::string timer_str("REDUCE_" + get_run_identifier(loopName));
    galois::StatTimer TsyncReduce(timer_str.c_str(), GRNAME);
    TsyncReduce.start();

#ifdef __GALOIS_BARE_MPI_COMMUNICATION__
    switch (bare_mpi) {
      case noBareMPI:
#endif
        sync_send<writeLocation, readLocation, syncReduce, ReduceFnTy, 
                  BitsetFnTy>(loopName);
        sync_recv<writeLocation, readLocation, syncReduce, ReduceFnTy, 
                  BitsetFnTy>(loopName);
#ifdef __GALOIS_BARE_MPI_COMMUNICATION__
        break;
      case nonBlockingBareMPI:
        sync_nonblocking_mpi<writeLocation, readLocation, syncReduce, ReduceFnTy, 
                  BitsetFnTy>(loopName);
        break;
      case oneSidedBareMPI:
        sync_onesided_mpi<writeLocation, readLocation, syncReduce, ReduceFnTy, 
                  BitsetFnTy>(loopName);
        break;
      default:
        GALOIS_DIE("Unsupported bare MPI");
    }
#endif

    TsyncReduce.stop();
  }
  
  /**
   * TODO
   */
  // broadcast from master to mirrors
  template<WriteLocation writeLocation, ReadLocation readLocation,
           typename BroadcastFnTy, typename BitsetFnTy>
  inline void broadcast(std::string loopName) {
    std::string timer_str("BROADCAST_" + get_run_identifier(loopName));
    galois::StatTimer TsyncBroadcast(timer_str.c_str(), GRNAME);

    TsyncBroadcast.start();

    bool use_bitset = true;

    if (currentBVFlag != nullptr) {
      if (readLocation == readSource && src_invalid(*currentBVFlag)) {
        use_bitset = false;
        *currentBVFlag = BITVECTOR_STATUS::NONE_INVALID;
        currentBVFlag = nullptr;
      } else if (readLocation == readDestination &&
                 dst_invalid(*currentBVFlag)) {
        use_bitset = false;
        *currentBVFlag = BITVECTOR_STATUS::NONE_INVALID;
        currentBVFlag = nullptr;
      } else if (readLocation == readAny &&
                 *currentBVFlag != BITVECTOR_STATUS::NONE_INVALID) {
        // the bitvector flag being non-null means this call came from
        // sync on demand; sync on demand will NEVER use readAny
        // if location is read Any + one of src or dst is invalid
        GALOIS_DIE("readAny + use of bitvector flag without none_invalid "
                   "should never happen");
      }
    }

#ifdef __GALOIS_BARE_MPI_COMMUNICATION__
    switch (bare_mpi) {
      case noBareMPI:
#endif
        if (use_bitset) {
          sync_send<writeLocation, readLocation, syncBroadcast, BroadcastFnTy,
                    BitsetFnTy>(loopName);
        } else {
          sync_send<writeLocation, readLocation, syncBroadcast, BroadcastFnTy,
                    galois::InvalidBitsetFnTy>(loopName);
        }
        sync_recv<writeLocation, readLocation, syncBroadcast, BroadcastFnTy,
                  BitsetFnTy>(loopName);
#ifdef __GALOIS_BARE_MPI_COMMUNICATION__
        break;
      case nonBlockingBareMPI:
        sync_nonblocking_mpi<writeLocation, readLocation, syncBroadcast, BroadcastFnTy,
                  BitsetFnTy>(loopName, use_bitset);
        break; 
      case oneSidedBareMPI:
        sync_onesided_mpi<writeLocation, readLocation, syncBroadcast, BroadcastFnTy,
                  BitsetFnTy>(loopName, use_bitset);
        break; 
      default:
        GALOIS_DIE("Unsupported bare MPI");
    }
#endif

    TsyncBroadcast.stop();
  }

  // OEC - outgoing edge-cut : source of any edge is master
  // IEC - incoming edge-cut : destination of any edge is master
  // CVC - cartesian vertex-cut : if source of an edge is mirror, 
  //                              then destination is not, and vice-versa
  // UVC - unconstrained vertex-cut

  // reduce - mirrors to master
  // broadcast - master to mirrors
  
  /**
   * TODO
   */
  template<typename ReduceFnTy, typename BroadcastFnTy, typename BitsetFnTy>
  inline void sync_src_to_src(std::string loopName) {
    // do nothing for OEC
    // reduce and broadcast for IEC, CVC, UVC
    if (transposed || is_vertex_cut()) {
      reduce<writeSource, readSource, ReduceFnTy, BitsetFnTy>(loopName);
      broadcast<writeSource, readSource, BroadcastFnTy, BitsetFnTy>(loopName);
    }
  }
  
  /**
   * TODO
   */
  template<typename ReduceFnTy, typename BroadcastFnTy, typename BitsetFnTy>
  inline void sync_src_to_dst(std::string loopName) {
    // only broadcast for OEC
    // only reduce for IEC
    // reduce and broadcast for CVC, UVC
    if (transposed) {
      reduce<writeSource, readDestination, ReduceFnTy, BitsetFnTy>(loopName);
      if (is_vertex_cut()) {
        broadcast<writeSource, readDestination, BroadcastFnTy, 
                  BitsetFnTy>(loopName);
      }
    } else {
      if (is_vertex_cut()) {
        reduce<writeSource, readDestination, ReduceFnTy, BitsetFnTy>(loopName);
      }
      broadcast<writeSource, readDestination, BroadcastFnTy, 
                BitsetFnTy>(loopName);
    }
  }
  
  /**
   * TODO
   */
  template<typename ReduceFnTy, typename BroadcastFnTy, typename BitsetFnTy>
  inline void sync_src_to_any(std::string loopName) {
    // only broadcast for OEC
    // reduce and broadcast for IEC, CVC, UVC
    if (transposed || is_vertex_cut()) {
      reduce<writeSource, readAny, ReduceFnTy, BitsetFnTy>(loopName);
    }
    broadcast<writeSource, readAny, BroadcastFnTy, BitsetFnTy>(loopName);
  }
  
  /**
   * TODO
   */
  template<typename ReduceFnTy, typename BroadcastFnTy, typename BitsetFnTy>
  inline void sync_dst_to_src(std::string loopName) {
    // only reduce for OEC
    // only broadcast for IEC
    // reduce and broadcast for CVC, UVC
    if (transposed) {
      if (is_vertex_cut()) {
        reduce<writeDestination, readSource, ReduceFnTy, BitsetFnTy>(loopName);
      }
      broadcast<writeDestination, readSource, BroadcastFnTy, 
                BitsetFnTy>(loopName);
    } else {
      reduce<writeDestination, readSource, ReduceFnTy, BitsetFnTy>(loopName);
      if (is_vertex_cut()) {
        broadcast<writeDestination, readSource, BroadcastFnTy, 
                  BitsetFnTy>(loopName);
      }
    }
  }
  
  /**
   * TODO
   */
  template<typename ReduceFnTy, typename BroadcastFnTy, typename BitsetFnTy>
  inline void sync_dst_to_dst(std::string loopName) {
    // do nothing for IEC
    // reduce and broadcast for OEC, CVC, UVC
    if (!transposed || is_vertex_cut()) {
      reduce<writeDestination, readDestination, ReduceFnTy, 
             BitsetFnTy>(loopName);
      broadcast<writeDestination, readDestination, BroadcastFnTy, 
                BitsetFnTy>(loopName);
    }
  }
  
  /**
   * TODO
   */
  template<typename ReduceFnTy, typename BroadcastFnTy, typename BitsetFnTy>
  inline void sync_dst_to_any(std::string loopName) {
    // only broadcast for IEC
    // reduce and broadcast for OEC, CVC, UVC
    if (!transposed || is_vertex_cut()) {
      reduce<writeDestination, readAny, ReduceFnTy, BitsetFnTy>(loopName);
    }
    broadcast<writeDestination, readAny, BroadcastFnTy, BitsetFnTy>(loopName);
  }
  
  /**
   * TODO
   */
  template<typename ReduceFnTy, typename BroadcastFnTy, typename BitsetFnTy>
  inline void sync_any_to_src(std::string loopName) {
    // only reduce for OEC
    // reduce and broadcast for IEC, CVC, UVC
    reduce<writeAny, readSource, ReduceFnTy, BitsetFnTy>(loopName);
    if (transposed || is_vertex_cut()) {
      broadcast<writeAny, readSource, BroadcastFnTy, BitsetFnTy>(loopName);
    }
  }
  
  /**
   * TODO
   */
  template<typename ReduceFnTy, typename BroadcastFnTy, typename BitsetFnTy>
  inline void sync_any_to_dst(std::string loopName) {
    // only reduce for IEC
    // reduce and broadcast for OEC, CVC, UVC
    reduce<writeAny, readDestination, ReduceFnTy, BitsetFnTy>(loopName);

    if (!transposed || is_vertex_cut()) {
      broadcast<writeAny, readDestination, BroadcastFnTy, BitsetFnTy>(loopName);
    }
  }
  
  /**
   * TODO
   */
  template<typename ReduceFnTy, typename BroadcastFnTy, typename BitsetFnTy>
  inline void sync_any_to_any(std::string loopName) {
    // reduce and broadcast for OEC, IEC, CVC, UVC
    reduce<writeAny, readAny, ReduceFnTy, BitsetFnTy>(loopName);
    broadcast<writeAny, readAny, BroadcastFnTy, BitsetFnTy>(loopName);
  }

public:
  /**
   * TODO
   */
  template<WriteLocation writeLocation, ReadLocation readLocation,
           typename ReduceFnTy, typename BroadcastFnTy,
           typename BitsetFnTy = galois::InvalidBitsetFnTy>
  inline void sync(std::string loopName) {
    std::string timer_str("SYNC_" + loopName + "_" + get_run_identifier());
    galois::StatTimer Tsync(timer_str.c_str(), GRNAME);
    Tsync.start();

    if (writeLocation == writeSource) {
      if (readLocation == readSource) {
        sync_src_to_src<ReduceFnTy, BroadcastFnTy, BitsetFnTy>(loopName);
      } else if (readLocation == readDestination) {
        sync_src_to_dst<ReduceFnTy, BroadcastFnTy, BitsetFnTy>(loopName);
      } else { // readAny
        sync_src_to_any<ReduceFnTy, BroadcastFnTy, BitsetFnTy>(loopName);
      }
    } else if (writeLocation == writeDestination) {
      if (readLocation == readSource) {
        sync_dst_to_src<ReduceFnTy, BroadcastFnTy, BitsetFnTy>(loopName);
      } else if (readLocation == readDestination) {
        sync_dst_to_dst<ReduceFnTy, BroadcastFnTy, BitsetFnTy>(loopName);
      } else { // readAny
        sync_dst_to_any<ReduceFnTy, BroadcastFnTy, BitsetFnTy>(loopName);
      }
    } else { // writeAny
      if (readLocation == readSource) {
        sync_any_to_src<ReduceFnTy, BroadcastFnTy, BitsetFnTy>(loopName);
      } else if (readLocation == readDestination) {
        sync_any_to_dst<ReduceFnTy, BroadcastFnTy, BitsetFnTy>(loopName);
      } else { // readAny
        sync_any_to_any<ReduceFnTy, BroadcastFnTy, BitsetFnTy>(loopName);
      }
    }

    Tsync.stop();
  }

private:
  /**
   * Generic Sync on demand handler. Should NEVER get to this (hence
   * the galois die).
   */
  template<ReadLocation rl, typename ReduceFnTy, typename BroadcastFnTy,
           typename BitsetFnTy>
  struct SyncOnDemandHandler {
    // note this call function signature is diff. from specialized versions:
    // will cause compile time error if this struct is used (which is what
    // we want)
    void call() {
      GALOIS_DIE("Invalid read location for sync on demand");
    }
  };

  /**
   * Sync on demand handler specialized for read source.
   *
   * @tparam ReduceFnTy specify how to do reductions
   * @tparam BroadcastFnTy specify how to do broadcasts
   * @tparam BitsetFnTy tells program what data needs to be sync'd
   */
  template<typename ReduceFnTy, typename BroadcastFnTy,
           typename BitsetFnTy>
  struct SyncOnDemandHandler<readSource, ReduceFnTy, BroadcastFnTy, 
                             BitsetFnTy> {
    /**
     * Based on sync flags, handles syncs for cases when you need to read
     * at source
     *
     * @param g The graph to sync
     * @param fieldFlags the flags structure specifying what needs to be 
     * sync'd
     * @param loopName loopname used to name timers
     * @param bvFlag Copy of the bitvector status (valid/invalid at particular
     * locations)
     */
    static inline void call(hGraph* g, FieldFlags& fieldFlags, std::string loopName,
                     const BITVECTOR_STATUS& bvFlag) {
      if (fieldFlags.src_to_src() && fieldFlags.dst_to_src()) {
        g->sync_any_to_src<ReduceFnTy, BroadcastFnTy, BitsetFnTy>(loopName);
      } else if (fieldFlags.src_to_src()) {
        g->sync_src_to_src<ReduceFnTy, BroadcastFnTy, BitsetFnTy>(loopName);
      } else if (fieldFlags.dst_to_src()) {
        g->sync_dst_to_src<ReduceFnTy, BroadcastFnTy, BitsetFnTy>(loopName);
      }
  
      fieldFlags.clear_read_src();
    }
  };

  /**
   * Sync on demand handler specialized for read destination.
   *
   * @tparam ReduceFnTy specify how to do reductions
   * @tparam BroadcastFnTy specify how to do broadcasts
   * @tparam BitsetFnTy tells program what data needs to be sync'd
   */
  template<typename ReduceFnTy, typename BroadcastFnTy,
           typename BitsetFnTy>
  struct SyncOnDemandHandler<readDestination, ReduceFnTy, BroadcastFnTy,
                             BitsetFnTy> {
    /**
     * Based on sync flags, handles syncs for cases when you need to read
     * at destination
     *
     * @param g The graph to sync
     * @param fieldFlags the flags structure specifying what needs to be 
     * sync'd
     * @param loopName loopname used to name timers
     * @param bvFlag Copy of the bitvector status (valid/invalid at particular
     * locations)
     */
    static inline void call(hGraph* g, FieldFlags& fieldFlags, std::string loopName,
                     const BITVECTOR_STATUS& bvFlag) {
      if (fieldFlags.src_to_dst() && fieldFlags.dst_to_dst()) {
        g->sync_any_to_dst<ReduceFnTy, BroadcastFnTy, BitsetFnTy>(loopName);
      } else if (fieldFlags.src_to_dst()) {
        g->sync_src_to_dst<ReduceFnTy, BroadcastFnTy, BitsetFnTy>(loopName);
      } else if (fieldFlags.dst_to_dst()) {
        g->sync_dst_to_dst<ReduceFnTy, BroadcastFnTy, BitsetFnTy>(loopName);
      }
  
      fieldFlags.clear_read_dst();
    }
  };

  /**
   * Sync on demand handler specialized for read any.
   *
   * @tparam ReduceFnTy specify how to do reductions
   * @tparam BroadcastFnTy specify how to do broadcasts
   * @tparam BitsetFnTy tells program what data needs to be sync'd
   */
  template<typename ReduceFnTy, typename BroadcastFnTy,
           typename BitsetFnTy>
  struct SyncOnDemandHandler<readAny, ReduceFnTy, BroadcastFnTy,
                             BitsetFnTy> {
    /**
     * Based on sync flags, handles syncs for cases when you need to read
     * at both source and destination
     *
     * @param g The graph to sync
     * @param fieldFlags the flags structure specifying what needs to be 
     * sync'd
     * @param loopName loopname used to name timers
     * @param bvFlag Copy of the bitvector status (valid/invalid at particular
     * locations)
     */
    static inline void call(hGraph* g, FieldFlags& fieldFlags, std::string loopName,
                     const BITVECTOR_STATUS& bvFlag) {
      bool src_write = fieldFlags.src_to_src() || fieldFlags.src_to_dst();
      bool dst_write = fieldFlags.dst_to_src() || fieldFlags.dst_to_dst();

      if (!(src_write && dst_write)) {
        // src or dst write flags aren't set (potentially both are not set),
        // but it's NOT the case that both are set, meaning "any" isn't
        // required in the "from"; can work at granularity of just src
        // write or dst wrte

        if (src_write) {
          if (fieldFlags.src_to_src() && fieldFlags.src_to_dst()) {
            if (bvFlag == BITVECTOR_STATUS::NONE_INVALID) {
              g->sync_src_to_any<ReduceFnTy, BroadcastFnTy, BitsetFnTy>(loopName);
            } else if (src_invalid(bvFlag)) {
              // src invalid bitset; sync individually so it can be called
              // without bitset
              g->sync_src_to_dst<ReduceFnTy, BroadcastFnTy, BitsetFnTy>(loopName);
              g->sync_src_to_src<ReduceFnTy, BroadcastFnTy, BitsetFnTy>(loopName);
            } else if (dst_invalid(bvFlag)) {
              // dst invalid bitset; sync individually so it can be called
              // without bitset
              g->sync_src_to_src<ReduceFnTy, BroadcastFnTy, BitsetFnTy>(loopName);
              g->sync_src_to_dst<ReduceFnTy, BroadcastFnTy, BitsetFnTy>(loopName);
            } else {
              GALOIS_DIE("Invalid bitvector flag setting in sync_on_demand");
            }
          } else if (fieldFlags.src_to_src()) {
            g->sync_src_to_src<ReduceFnTy, BroadcastFnTy, BitsetFnTy>(loopName);
          } else { // src to dst is set
            g->sync_src_to_dst<ReduceFnTy, BroadcastFnTy, BitsetFnTy>(loopName);
          }
        } else if (dst_write) {
          if (fieldFlags.dst_to_src() && fieldFlags.dst_to_dst()) {
            if (bvFlag == BITVECTOR_STATUS::NONE_INVALID) {
              g->sync_dst_to_any<ReduceFnTy, BroadcastFnTy, BitsetFnTy>(loopName);
            } else if (src_invalid(bvFlag)) {
              g->sync_dst_to_dst<ReduceFnTy, BroadcastFnTy, BitsetFnTy>(loopName);
              g->sync_dst_to_src<ReduceFnTy, BroadcastFnTy, BitsetFnTy>(loopName);
            } else if (dst_invalid(bvFlag)) {
              g->sync_dst_to_src<ReduceFnTy, BroadcastFnTy, BitsetFnTy>(loopName);
              g->sync_dst_to_dst<ReduceFnTy, BroadcastFnTy, BitsetFnTy>(loopName);
            } else {
              GALOIS_DIE("Invalid bitvector flag setting in sync_on_demand");
            }
          } else if (fieldFlags.dst_to_src()) {
            g->sync_dst_to_src<ReduceFnTy, BroadcastFnTy, BitsetFnTy>(loopName);
          } else { // dst to dst is set
            g->sync_dst_to_dst<ReduceFnTy, BroadcastFnTy, BitsetFnTy>(loopName);
          }
        }

        // note the "no flags are set" case will enter into this block
        // as well, and it is correctly handled by doing nothing since
        // both src/dst_write will be false
      } else {
        // it is the case that both src/dst write flags are set, so "any"
        // is required in the "from"; what remains to be determined is
        // the use of src, dst, or any for the destination of the sync
        bool src_read = fieldFlags.src_to_src() || fieldFlags.dst_to_src();
        bool dst_read = fieldFlags.src_to_dst() || fieldFlags.dst_to_dst();

        if (src_read && dst_read) {
          if (bvFlag == BITVECTOR_STATUS::NONE_INVALID) {
            g->sync_any_to_any<ReduceFnTy, BroadcastFnTy, BitsetFnTy>(loopName);
          } else if (src_invalid(bvFlag)) {
            g->sync_any_to_dst<ReduceFnTy, BroadcastFnTy, BitsetFnTy>(loopName);
            g->sync_any_to_src<ReduceFnTy, BroadcastFnTy, BitsetFnTy>(loopName);
          } else if (dst_invalid(bvFlag)) {
            g->sync_any_to_src<ReduceFnTy, BroadcastFnTy, BitsetFnTy>(loopName);
            g->sync_any_to_dst<ReduceFnTy, BroadcastFnTy, BitsetFnTy>(loopName);
          } else {
            GALOIS_DIE("Invalid bitvector flag setting in sync_on_demand");
          }
        } else if (src_read) {
          g->sync_any_to_src<ReduceFnTy, BroadcastFnTy, BitsetFnTy>(loopName);
        } else { // dst_read
          g->sync_any_to_dst<ReduceFnTy, BroadcastFnTy, BitsetFnTy>(loopName);
        }
      }

      fieldFlags.clear_read_src();
      fieldFlags.clear_read_dst();
    }
  };

public:
  /**
   * Given a structure that contains flags signifying what needs to be
   * synchronized, sync_on_demand will synchronize what is necessary based 
   * on the read location of the * field. 
   *
   * @tparam readLocation Location in which field will need to be read
   * @tparam ReduceFnTy reduce sync structure for the field
   * @tparam BroadcastFnTy broadcast sync structure for the field
   * @tparam BitsetFnTy struct which holds a bitset which can be used
   * to control synchronization at a more fine grain level
   * @param fieldFlags structure for field you are syncing
   * @param loopName Name of loop this sync is for for naming timers
   */
  template<ReadLocation readLocation,
           typename ReduceFnTy, typename BroadcastFnTy,
           typename BitsetFnTy = galois::InvalidBitsetFnTy>
  inline void sync_on_demand(FieldFlags& fieldFlags, std::string loopName) {
    std::string timer_str("SYNC_" + get_run_identifier(loopName));
    galois::StatTimer Tsync(timer_str.c_str(), GRNAME);
    Tsync.start();

    currentBVFlag = &(fieldFlags.bitvectorStatus);

    // call a template-specialized function depending on the read location
    SyncOnDemandHandler<readLocation, ReduceFnTy, BroadcastFnTy, BitsetFnTy>::
          call(this, fieldFlags, loopName, *currentBVFlag);

    currentBVFlag = nullptr;

    Tsync.stop();
  }

#ifdef __GALOIS_CHECKPOINT__
private:
  /****** checkpointing **********/
  galois::runtime::RecvBuffer checkpoint_recvBuffer;

  static void syncRecv(uint32_t src, galois::runtime::RecvBuffer& buf) {
    uint32_t oid;
    void (hGraph::*fn)(galois::runtime::RecvBuffer&);
    galois::runtime::gDeserialize(buf, oid, fn);
    hGraph* obj = reinterpret_cast<hGraph*>(ptrForObj(oid));
    (obj->*fn)(buf);
  }

  void exchange_info_landingPad(galois::runtime::RecvBuffer& buf) {
    uint32_t hostID;
    uint64_t numItems;
    std::vector<uint64_t> items;
    galois::runtime::gDeserialize(buf, hostID, numItems);
    galois::runtime::gDeserialize(buf, masterNodes[hostID]);
  }

  template<typename FnTy>
  void syncRecvApply_ck(uint32_t from_id, galois::runtime::RecvBuffer& buf,
                        std::string loopName) {
    std::string set_timer_str("SYNC_SET_" + get_run_identifier(loopName));
    std::string doall_str("LAMBDA::REDUCE_RECV_APPLY_" + get_run_identifier(loopName));
    galois::StatTimer Tset(set_timer_str.c_str(), GRNAME);
    Tset.start();

    uint32_t num = masterNodes[from_id].size();
    std::vector<typename FnTy::ValTy> val_vec(num);
    galois::runtime::gDeserialize(buf, val_vec);

    if (num > 0) {
      if (!FnTy::reduce_batch(from_id, val_vec.data())) {
        galois::do_all(galois::iterate(0u, num),
                       [&](uint32_t n) {
                         uint32_t lid = masterNodes[from_id][n];
                         #ifdef __GALOIS_HET_OPENCL__
                         CLNodeDataWrapper d = clGraph.getDataW(lid);
                         FnTy::reduce(lid, d, val_vec[n]);
                         #else
                         FnTy::reduce(lid, getData(lid), val_vec[n]);
                         #endif
                       },
                       galois::loopname(get_run_identifier(doall_str).c_str()),
                       galois::no_stats());
      }
    }

    if (id == (from_id + 1) % numHosts) {
      checkpoint_recvBuffer = std::move(buf);
    }

    Tset.stop();
  }

  template<typename FnTy>
  void reduce_ck(std::string loopName) {
    std::string extract_timer_str("REDUCE_EXTRACT_" + 
                                  get_run_identifier(loopName));
    std::string timer_str("REDUCE_" + get_run_identifier(loopName));
    std::string timer_barrier_str("REDUCE_BARRIER_" + 
                                  get_run_identifier(loopName));
    std::string statSendBytes_str("SEND_BYTES_REDUCE_" + 
                                  get_run_identifier(loopName));
    std::string doall_str("LAMBDA::REDUCE_" + get_run_identifier(loopName));

    galois::StatTimer TsyncReduce(timer_str.c_str(), GRNAME);
    galois::StatTimer StatTimerBarrier_syncReduce(timer_barrier_str.c_str(), GRNAME);
    galois::StatTimer Textract(extract_timer_str.c_str(), GRNAME);

    std::string statChkPtBytes_str("CHECKPOINT_BYTES_REDUCE_" + 
                                   get_run_identifier(loopName));

    // TODO/FIXME this didn't include loop name originally
    std::string checkpoint_timer_str("TIME_CHECKPOINT_REDUCE_MEM_" +
                                     get_run_identifier(loopName));

    galois::StatTimer Tcheckpoint(checkpoint_timer_str.c_str(), GRNAME);

    TsyncReduce.start();
    auto& net = galois::runtime::getSystemNetworkInterface();

    size_t SyncReduce_send_bytes = 0;
    size_t checkpoint_bytes = 0;
    for (unsigned h = 1; h < numHosts; ++h) {
      unsigned x = (id + h) % numHosts;
      uint32_t num = mirrorNodes[x].size();

      galois::runtime::SendBuffer b;

      Textract.start();
      std::vector<typename FnTy::ValTy> val_vec(num);

      if (num > 0) {
        if (!FnTy::extract_reset_batch(x, val_vec.data())) {
          galois::do_all(galois::iterate(0u, num), 
              [&](uint32_t n) {
                uint32_t lid = mirrorNodes[x][n];
                #ifdef __GALOIS_HET_OPENCL__
                CLNodeDataWrapper d = clGraph.getDataW(lid);
                auto val = FnTy::extract(lid, getData(lid, d));
                FnTy::reset(lid, d);
                #else
                auto val = FnTy::extract(lid, getData(lid));
                FnTy::reset(lid, getData(lid));
                #endif
                val_vec[n] = val;
              }, 
              galois::loopname(get_run_identifier(doall_str).c_str()),
              galois::no_stats());
        }
      }

      gSerialize(b, val_vec);

      SyncReduce_send_bytes += b.size();
      auto send_bytes = b.size();

      Tcheckpoint.start();
      if (x == (id + 1) % numHosts) {
        // checkpoint owned nodes.
        std::vector<typename FnTy::ValTy> checkpoint_val_vec(numOwned);
        galois::do_all(galois::iterate(0u, numOwned), 
            [&](uint32_t n) {
             auto val = FnTy::extract(n, getData(n));
             checkpoint_val_vec[n] = val;
            }, 
            galois::loopname(get_run_identifier(doall_str).c_str()), 
            galois::no_stats());

        gSerialize(b, checkpoint_val_vec);
        checkpoint_bytes += (b.size() - send_bytes);
      }

      Tcheckpoint.stop();
      Textract.stop();

      net.sendTagged(x, galois::runtime::evilPhase, b);
    }

    // Will force all messages to be processed before continuing
    net.flush();

    galois::runtime::reportStat_Tsum(GRNAME, statSendBytes_str, 
        SyncReduce_send_bytes);
    galois::runtime::reportStat_Tsum(GRNAME, statChkPtBytes_str, 
        checkpoint_bytes);

    // receive
    for (unsigned x = 0; x < numHosts; ++x) {
      if ((x == id)) continue;

      decltype(net.recieveTagged(galois::runtime::evilPhase,nullptr)) p;

      do {
        net.handleReceives();
        p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
      } while (!p);

      syncRecvApply_ck<FnTy>(p->first, p->second, loopName);
    }
    ++galois::runtime::evilPhase;

    TsyncReduce.stop();
  }

 /****************************************
  * Fault Tolerance
  * 1. CheckPointing
  ***************************************/
public:
  template <typename FnTy>
  void checkpoint(std::string loopName) {
    auto& net = galois::runtime::getSystemNetworkInterface();
    std::string doall_str("LAMBDA::CHECKPOINT_" + get_run_identifier(loopName));
    std::string checkpoint_timer_str("TIME_CHECKPOINT_" + get_run_identifier());
    std::string checkpoint_fsync_timer_str("TIME_CHECKPOINT_FSYNC_" + 
                                           get_run_identifier());
    galois::StatTimer Tcheckpoint(checkpoint_timer_str.c_str(), GRNAME);
    galois::StatTimer Tcheckpoint_fsync(checkpoint_fsync_timer_str.c_str(), GRNAME);

    Tcheckpoint.start();

    std::string statChkPtBytes_str("CHECKPOINT_BYTES_" + 
                                   get_run_identifier(loopName));

    // checkpoint owned nodes.
    std::vector<typename FnTy::ValTy> val_vec(numOwned);
    galois::do_all(galois::iterate(0u, numOwned), 
        [&](uint32_t n) {
          auto val = FnTy::extract(n, getData(n));
          val_vec[n] = val;
        }, 
        galois::loopname(get_run_identifier(doall_str).c_str()), 
        galois::no_stats());

    galois::runtime::reportStat_Tsum(GRNAME, statChkPtBytes_str, 
        val_vec.size() * sizeof(typename FnTy::ValTy));

    //std::string chkPt_fileName = "/scratch/02982/ggill0/Checkpoint_" + loopName + "_" + FnTy::field_name() + "_" + std::to_string(net.ID);
    //std::string chkPt_fileName = "Checkpoint_" + loopName + "_" + FnTy::field_name() + "_" + std::to_string(net.ID);
    //std::string chkPt_fileName = "CheckPointFiles_" + std::to_string(numHosts) + "/Checkpoint_" + loopName + "_" + FnTy::field_name() + "_" + std::to_string(net.ID);

    #ifdef __TMPFS__
    #ifdef __CHECKPOINT_NO_FSYNC__
    std::string chkPt_fileName = "/dev/shm/CheckPointFiles_no_fsync_" + 
                                 std::to_string(numHosts) + "/Checkpoint_" + 
                                 loopName + "_" + FnTy::field_name() + "_" + 
                                 std::to_string(net.ID);
    galois::runtime::reportParam(GRNAME, "CHECKPOINT_FILE_LOC_", chkPt_fileName);
    #else
    std::string chkPt_fileName = "/dev/shm/CheckPointFiles_fsync_" + 
                                 std::to_string(numHosts) + "/Checkpoint_" + 
                                 loopName + "_" + FnTy::field_name() + "_" + 
                                 std::to_string(net.ID);
    galois::runtime::reportParam(GRNAME, "CHECKPOINT_FILE_LOC_", chkPt_fileName);
    #endif
    galois::runtime::reportParam(GRNAME, "CHECKPOINT_FILE_LOC_", chkPt_fileName);
    #else

    #ifdef __CHECKPOINT_NO_FSYNC__
    std::string chkPt_fileName = "CheckPointFiles_no_fsync_" + 
                                 std::to_string(numHosts) + "/Checkpoint_" + 
                                 loopName + "_" + FnTy::field_name() + "_" + 
                                 std::to_string(net.ID);
    galois::runtime::reportParam(GRNAME, "CHECKPOINT_FILE_LOC_", chkPt_fileName);
    #else
    std::string chkPt_fileName = "CheckPointFiles_fsync_" + 
                                 std::to_string(numHosts) + "/Checkpoint_" + 
                                 loopName + "_" + FnTy::field_name() + "_" + 
                                 std::to_string(net.ID);
    galois::runtime::reportParam(GRNAME, "CHECKPOINT_FILE_LOC_", chkPt_fileName);
    #endif
    #endif

    //std::ofstream chkPt_file(chkPt_fileName, std::ios::out | std::ofstream::binary | std::ofstream::trunc);
    #if __TMPFS__
    int fd = shm_open(chkPt_fileName.c_str(),O_CREAT|O_RDWR|O_TRUNC, 0666);
    #else
    int fd = open(chkPt_fileName.c_str(),O_CREAT|O_RDWR|O_TRUNC, 0666);
    #endif
    if (fd == -1) {
      GALOIS_DIE("File could not be created. file name : ", chkPt_fileName, " fd : ", fd, "\n");
    }

    write(fd, reinterpret_cast<char*>(val_vec.data()), 
          val_vec.size() * sizeof(typename FnTy::ValTy));
    //chkPt_file.write(reinterpret_cast<char*>(val_vec.data()), val_vec.size()*sizeof(uint32_t));
    Tcheckpoint_fsync.start();
    #ifdef __CHECKPOINT_NO_FSYNC__
    #else
    fsync(fd);
    #endif
    Tcheckpoint_fsync.stop();

    close(fd);
    //chkPt_file.close();
    Tcheckpoint.stop();
  }

  template<typename FnTy>
  void checkpoint_apply(std::string loopName){
    auto& net = galois::runtime::getSystemNetworkInterface();
    std::string doall_str("LAMBDA::CHECKPOINT_APPLY_" + 
                          get_run_identifier(loopName));
    // checkpoint owned nodes.
    std::vector<typename FnTy::ValTy> val_vec(numOwned);
    // read val_vec from disk.
    // std::string chkPt_fileName = "/scratch/02982/ggill0/Checkpoint_" + loopName + "_" + FnTy::field_name() + "_" + std::to_string(net.ID);
    std::string chkPt_fileName = "Checkpoint_" + loopName + "_" + 
        FnTy::field_name() + "_" + std::to_string(net.ID);
    std::ifstream chkPt_file(chkPt_fileName, std::ios::in | std::ofstream::binary);

    if (!chkPt_file.is_open()) {
      GALOIS_DIE("Unable to open checkpoint file ", chkPt_fileName, " ! Exiting!\n");
    }

    chkPt_file.read(reinterpret_cast<char*>(val_vec.data()), numOwned * sizeof(uint32_t));

    if (id == 0) {
      for (auto k = 0; k < 10; ++k) {
        galois::gPrint("AFTER : val_vec[", k, "] : ", val_vec[k], "\n");
      }
    }

    galois::do_all(galois::iterate(0u, numOwned), 
      [&](uint32_t n) {
        FnTy::setVal(n, getData(n), val_vec[n]);
      }, 
      galois::loopname(get_run_identifier(doall_str).c_str()), 
      galois::no_stats());
  }

 /*************************************************
  * Fault Tolerance
  * 1. CheckPointing in the memory of another node
  ************************************************/
  template<typename FnTy>
  void saveCheckPoint(galois::runtime::RecvBuffer& b){
    checkpoint_recvBuffer = std::move(b);
  }

  template<typename FnTy>
  void checkpoint_mem(std::string loopName) {
    auto& net = galois::runtime::getSystemNetworkInterface();
    std::string doall_str("LAMBDA::CHECKPOINT_MEM_" + 
        get_run_identifier(loopName));

    std::string statChkPtBytes_str("CHECKPOINT_BYTES_" + 
        get_run_identifier(loopName));

    std::string checkpoint_timer_str("TIME_CHECKPOINT_TOTAL_MEM_" + 
        get_run_identifier());
    galois::StatTimer Tcheckpoint(checkpoint_timer_str.c_str(), GRNAME);

    std::string checkpoint_timer_send_str("TIME_CHECKPOINT_TOTAL_MEM_SEND_" + 
        get_run_identifier());
    galois::StatTimer Tcheckpoint_send(checkpoint_timer_send_str.c_str(), GRNAME);

    std::string checkpoint_timer_recv_str("TIME_CHECKPOINT_TOTAL_MEM_recv_" + 
        get_run_identifier());
    galois::StatTimer Tcheckpoint_recv(checkpoint_timer_recv_str.c_str(), GRNAME);

    Tcheckpoint.start();

    Tcheckpoint_send.start();
    // checkpoint owned nodes.
    std::vector<typename FnTy::ValTy> val_vec(numOwned);
    galois::do_all(galois::iterate(0u, numOwned), 
        [&](uint32_t n) {
          auto val = FnTy::extract(n, getData(n));
          val_vec[n] = val;
        }, 
        galois::loopname(get_run_identifier(doall_str).c_str()), 
        galois::no_stats());

    galois::runtime::SendBuffer b;
    gSerialize(b, val_vec);

    galois::runtime::reportStat_Tsum(GRNAME, statChkPtBytes_str, b.size());
    // send to your neighbor on your left.
    net.sendTagged((net.ID + 1) % numHosts, galois::runtime::evilPhase, b);

    Tcheckpoint_send.stop();

    net.flush();

    Tcheckpoint_recv.start();
    // receiving the checkpointed data.
    decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;

    do {
      net.handleReceives();
      p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
    } while (!p);

    checkpoint_recvBuffer = std::move(p->second);

    ++galois::runtime::evilPhase;
    Tcheckpoint_recv.stop();
    Tcheckpoint.stop();
  }

  template<typename FnTy>
  void checkpoint_mem_apply(galois::runtime::RecvBuffer& b) {
    auto& net = galois::runtime::getSystemNetworkInterface();
    std::string doall_str("LAMBDA::CHECKPOINT_MEM_APPLY_" +
        get_run_identifier());

    std::string checkpoint_timer_str("TIME_CHECKPOINT_MEM_APPLY" + 
        get_run_identifier());
    galois::StatTimer Tcheckpoint(checkpoint_timer_str.c_str(), GRNAME);
    Tcheckpoint.start();

    uint32_t from_id;
    galois::runtime::RecvBuffer recv_checkpoint_buf;
    gDeserialize(b, from_id);
    recv_checkpoint_buf = std::move(b);

    //gDeserialize(b, recv_checkpoint_buf);

    std::vector<typename FnTy::ValTy> val_vec(numOwned);
    gDeserialize(recv_checkpoint_buf, val_vec);

    if (net.ID == 0) {
      for (auto k = 0; k < 10; ++k) {
        galois::gPrint("AFTER : val_vec[", k, "] : ", val_vec[k], "\n");
      }
    }

    galois::do_all(galois::iterate(0u, numOwned), 
        [&](uint32_t n) {
          FnTy::setVal(n, getData(n), val_vec[n]);
        }, 
        galois::loopname(get_run_identifier(doall_str).c_str()), 
        galois::no_stats());
  }

private:
  template<typename FnTy>
  void recovery_help_landingPad(galois::runtime::RecvBuffer& buff) {
    void (hGraph::*fn)(galois::runtime::RecvBuffer&) = 
      &hGraph::checkpoint_mem_apply<FnTy>;
    auto& net = galois::runtime::getSystemNetworkInterface();
    uint32_t from_id;
    std::string help_str;

    gDeserialize(buff, from_id, help_str);
    galois::runtime::SendBuffer b;
    gSerialize(b, idForSelf(), fn, net.ID, checkpoint_recvBuffer);
    net.sendMsg(from_id, syncRecv, b);
    //send back the checkpointed nodes for from_id.
  }


  template<typename FnTy>
  void recovery_send_help(std::string loopName) {
    void (hGraph::*fn)(galois::runtime::RecvBuffer&) = 
      &hGraph::recovery_help_landingPad<FnTy>;

    auto& net = galois::runtime::getSystemNetworkInterface();
    galois::runtime::SendBuffer b;
    std::string help_str = "recoveryHelp";

    gSerialize(b, idForSelf(), fn, net.ID, help_str);

    // jsend help message to the host that is keeping checkpoint for you.
    net.sendMsg((net.ID + 1) % numHosts, syncRecv, b);
  }


  /*****************************************************/

 /****************************************
  * Fault Tolerance
  * 1. Zorro
  ***************************************/
  #if 0
  void recovery_help_landingPad(galois::runtime::RecvBuffer& b) {
    uint32_t from_id;
    std::string help_str;
    gDeserialize(b, from_id, help_str);

    //send back the mirrorNode for from_id.

  }

  template<typename FnTy>
  void recovery_send_help(std::string loopName) {
    void (hGraph::*fn)(galois::runtime::RecvBuffer&) = &hGraph::recovery_help<FnTy>;
    auto& net = galois::runtime::getSystemNetworkInterface();
    galois::runtime::SendBuffer b;
    std::string help_str = "recoveryHelp";

    gSerialize(b, idForSelf(), help_str);

    for(auto i = 0; i < numHosts; ++i){
      net.sendMsg(i, syncRecv, b);
    }
  }
  #endif

  /*************************************/
#endif

public:
  /**
   * Converts a local node id into a global node id
   *
   * @param nodeID local node id
   * @returns global node id corresponding to the local one
   */
  inline uint64_t getGID(const uint32_t nodeID) const {
     return L2G(nodeID);
  }

  /**
   * Converts a global node id into a local node id
   *
   * @param nodeID global node id
   * @returns local node id corresponding to the global one
   */
  inline uint32_t getLID(const uint64_t nodeID) const {
     return G2L(nodeID);
  }

#ifdef __GALOIS_HET_CUDA__
private:
  // Code that handles getting the graph onto the GPU
  template<bool isVoidType, typename std::enable_if<isVoidType>::type* = nullptr>
  inline void setMarshalEdge(MarshalGraph &m, const size_t index, const edge_iterator &e) {
     // do nothing
  }

  template<bool isVoidType, typename std::enable_if<!isVoidType>::type* = nullptr>
  inline void setMarshalEdge(MarshalGraph &m, const size_t index, const edge_iterator &e) {
     m.edge_data[index] = getEdgeData(e);
  }

public:
  void getMarshalGraph(MarshalGraph& m) {
    m.nnodes = size();
    m.nedges = sizeEdges();
    m.nowned = numOwned;
    assert(m.nowned > 0);
    m.id = id;
    m.row_start = (index_type*) calloc(m.nnodes + 1, sizeof(index_type));
    m.edge_dst = (index_type*) calloc(m.nedges, sizeof(index_type));

    // initialize node_data with localID-to-globalID mapping
    m.node_data = (index_type *) calloc(m.nnodes, sizeof(node_data_type));

    for (index_type i = 0; i < m.nnodes; ++i) {
      m.node_data[i] = getGID(i);
    }

    if (std::is_void<EdgeTy>::value) {
      m.edge_data = NULL;
    } else {
      if (!std::is_same<EdgeTy, edge_data_type>::value) {
        galois::gWarn("Edge data type mismatch between CPU and GPU\n");
      }

      m.edge_data = (edge_data_type *) calloc(m.nedges, sizeof(edge_data_type));
    }

    // pinched from Rashid's LC_LinearArray_Graph.h
    size_t edge_counter = 0, node_counter = 0;
    for (auto n = graph.begin(); 
         n != graph.end() && *n != m.nnodes; 
         n++, node_counter++) {
      m.row_start[node_counter] = edge_counter;
      if (*n < m.nowned) {
        for (auto e = edge_begin(*n); e != edge_end(*n); e++) {
           if (getEdgeDst(e) < m.nnodes) {
              setMarshalEdge<std::is_void<EdgeTy>::value>(m, edge_counter, e);
              m.edge_dst[edge_counter++] = getEdgeDst(e);
           }
        }
      }
    }

    m.row_start[node_counter] = edge_counter;
    m.nedges = edge_counter;

    // copy memoization meta-data
    m.num_master_nodes = (unsigned int *)calloc(masterNodes.size(), 
        sizeof(unsigned int));;
    m.master_nodes = (unsigned int **) calloc(masterNodes.size(), 
        sizeof(unsigned int *));;

    for (uint32_t h = 0; h < masterNodes.size(); ++h) {
      m.num_master_nodes[h] = masterNodes[h].size();

      if (masterNodes[h].size() > 0) {
        m.master_nodes[h] = (unsigned int *) calloc(masterNodes[h].size(), 
            sizeof(unsigned int));;
        std::copy(masterNodes[h].begin(), masterNodes[h].end(), m.master_nodes[h]);
      } else {
        m.master_nodes[h] = NULL;
      }
    }

    m.num_mirror_nodes = (unsigned int *) calloc(mirrorNodes.size(), 
        sizeof(unsigned int));;
    m.mirror_nodes = (unsigned int **) calloc(mirrorNodes.size(), 
        sizeof(unsigned int *));;
    for (uint32_t h = 0; h < mirrorNodes.size(); ++h) {
      m.num_mirror_nodes[h] = mirrorNodes[h].size();

      if (mirrorNodes[h].size() > 0) {
        m.mirror_nodes[h] = (unsigned int *) calloc(mirrorNodes[h].size(), 
            sizeof(unsigned int));;
        std::copy(mirrorNodes[h].begin(), mirrorNodes[h].end(), m.mirror_nodes[h]);
      } else {
        m.mirror_nodes[h] = NULL;
      }
    }
  }
#endif

#ifdef __GALOIS_HET_OPENCL__
  typedef galois::opencl::Graphs::CL_LC_Graph<NodeTy, EdgeTy> CLGraphType;
  typedef typename CLGraphType::NodeDataWrapper CLNodeDataWrapper;
  typedef typename CLGraphType::NodeIterator CLNodeIterator;
  CLGraphType clGraph;

  const cl_mem & device_ptr() const {
    return clGraph.device_ptr();
  }
  CLNodeDataWrapper getDataW(GraphNode N, 
      galois::MethodFlag mflag = galois::MethodFlag::UNPROTECTED) {
    return clGraph.getDataW(N);
  }
  const CLNodeDataWrapper getDataR(GraphNode N,
      galois::MethodFlag mflag = galois::MethodFlag::UNPROTECTED) {
    return clGraph.getDataR(N);
  }
#endif

  /**
   * Set the run number. (contrary to what the function name is)
   *
   * TODO rename, then fix across apps
   *
   * @param runNum Number to set the run to
   */
  inline void set_num_run(const uint32_t runNum) {
     num_run = runNum;
  }

  /**
   * Get the set run number.
   *
   * @returns The set run number saved in the graph
   */
  inline uint32_t get_run_num() const {
    return num_run;
  }

  /**
   * Set the iteration number for use in the run identifier.
   *
   * @param iteration Iteration number to set to
   */
  inline void set_num_iter(const uint32_t iteration) {
    num_iteration = iteration;
  }

  /**
   * Get a run identifier using the set run and set iteration.
   * Deprecated: use the other one below where possible
   *
   * @returns a string run identifier
   */
  inline std::string get_run_identifier() const {
    return std::string(std::to_string(num_run) + "_" +
                       std::to_string(num_iteration));
  }

  /**
   * Get a run identifier using the set run and set iteration and
   * append to the passed in string.
   *
   * @param loop_name String to append the run identifier
   * @returns String with run identifier appended to passed in loop name
   */
  inline std::string get_run_identifier(std::string loop_name) const {
    return std::string(std::string(loop_name) + "_" + std::to_string(num_run) +
                       "_" + std::to_string(num_iteration));
  }

};
template<typename NodeTy, typename EdgeTy, bool BSPNode, bool BSPEdge>
constexpr const char* const hGraph<NodeTy, EdgeTy, BSPNode, BSPEdge>::GRNAME;
#endif //_GALOIS_DIST_HGRAPH_H
