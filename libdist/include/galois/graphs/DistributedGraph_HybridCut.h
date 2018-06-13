/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
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
 * @file DistributedGraph_HybridCut.h
 * Implements the hybrid-cut partitioning scheme for DistGraph.
 */

#ifndef _GALOIS_DIST_HGRAPHHYBRID_H
#define _GALOIS_DIST_HGRAPHHYBRID_H

#include "galois/graphs/DistributedGraph.h"

namespace galois {
namespace graphs {

/**
 * Distributed graph that implements the hybrid-vertex cut partitioning scheme.
 *
 * @tparam NodeTy type of node data for the graph
 * @tparam EdgeTy type of edge data for the graph
 */
template<typename NodeTy, typename EdgeTy>
class DistGraphHybridCut : public DistGraph<NodeTy, EdgeTy> {
  constexpr static const char* const GRNAME = "dGraph_hybridCut";

  // TODO doc; to send edges to different hosts: #Src #Dst
  std::vector<std::vector<uint64_t>> assigned_edges_perhost;
  // TODO doc
  uint64_t num_total_edges_to_receive;

public:
  //! typedef to base DistGraph class
  using base_DistGraph = DistGraph<NodeTy, EdgeTy>;

  //! GID of a node = localToGlobalVector[LID]
  std::vector<uint64_t> localToGlobalVector;
  //! LID of a node = globalToLocalMap[GID]
  std::unordered_map<uint64_t, uint32_t> globalToLocalMap;

  //! @copydoc DistGraphEdgeCut::gid2host_withoutEdges
  std::vector<std::pair<uint64_t, uint64_t>> gid2host_withoutEdges;

  //! offset to first node that is a master on this host
  uint64_t globalOffset;
  //! number of nodes that exist locally in this graph
  uint32_t numNodes;
  //! number of edges that exist locally in this graph
  uint64_t numEdges;
  //! says if graph is bipartite
  //! @warning this option isn't maintained; use at own risk
  bool isBipartite;

  //! @copydoc DistGraphEdgeCut::getHostID
  unsigned getHostID(uint64_t gid) const {
    return find_hostID(gid);
  }

  //! @copydoc DistGraphEdgeCut::isOwned
  bool isOwned(uint64_t gid) const {
    if (gid >= base_DistGraph::gid2host[base_DistGraph::id].first && 
        gid < base_DistGraph::gid2host[base_DistGraph::id].second)
      return true;
    else
      return false;
  }

  //! @copydoc DistGraphEdgeCut::isLocal
  virtual bool isLocal(uint64_t gid) const {
    assert(gid < base_DistGraph::numGlobalNodes);
    if (isOwned(gid))
      return true;
    return (globalToLocalMap.find(gid) != globalToLocalMap.end());
  }

  //! @copydoc DistGraphEdgeCut::G2L
  virtual uint32_t G2L(uint64_t gid) const {
    assert(isLocal(gid));
    return globalToLocalMap.at(gid);
  }

  //! @copydoc DistGraphEdgeCut::L2G
  virtual uint64_t L2G(uint32_t lid) const {
    return localToGlobalVector[lid];
  }

  /**
   * Constructor for hybrid vertex cut: partitions the graph across
   * hosts accordingly.
   *
   * @param filename Graph to load (must be in Galois binary format)
   * @param host Host id of the caller
   * @param _numHosts total number of hosts currently running
   * @param scalefactor Unsupported for vertex cuts; dummy variable
   * @param transpose Indicates if the read graph needs to be transposed
   * @param VCutThreshold Threshold for determining how to assign edges:
   * if greater than threshold, edges go to the host that has the destination
   * @param bipartite Indicates if the graph is bipartite
   * @param readFromFile if true, reads the local graph structure from a file 
   * instead of doing partitioning
   * @param localGraphFileName local graph structure file to read (if readFromFile
   * is set to true)
   *
   * @todo split constructor into more parts? quite long at the moment
   */
  DistGraphHybridCut(const std::string& filename, 
             const std::string&,
             unsigned host, unsigned _numHosts, 
             std::vector<unsigned>& scalefactor, 
             bool transpose = false, 
             uint32_t VCutThreshold = 100, 
             bool bipartite = false, 
             bool readFromFile = false, 
             std::string localGraphFileName = "local_graph") 
        : base_DistGraph(host, _numHosts) {
    if (!scalefactor.empty()) {
      if (base_DistGraph::id == 0) {
        galois::gWarn("Scalefactor not supported for PowerLyra (hybrid) "
                      "vertex-cuts");
      }
      scalefactor.clear();
    }

    galois::CondStatTimer<MORE_DIST_STATS> Tgraph_construct(
      "GraphPartitioningTime", GRNAME
    );

    Tgraph_construct.start();

    if (readFromFile) {
      galois::gPrint("[", base_DistGraph::id, "] Reading local graph from file : ", 
                     localGraphFileName, "\n");
      base_DistGraph::read_local_graph_from_file(localGraphFileName);
      Tgraph_construct.stop();
      return;
    }

    galois::graphs::OfflineGraph g(filename);
    isBipartite = bipartite;

    base_DistGraph::numGlobalNodes = g.size();
    base_DistGraph::numGlobalEdges = g.sizeEdges();

    uint64_t numNodes_to_divide = base_DistGraph::computeMasters(g, scalefactor, 
                                                              isBipartite);

    // at this point gid2Host has pairs for how to split nodes among
    // hosts; pair has begin and end
    uint64_t nodeBegin = base_DistGraph::gid2host[base_DistGraph::id].first;
    typename galois::graphs::OfflineGraph::edge_iterator edgeBegin = 
      g.edge_begin(nodeBegin);

    uint64_t nodeEnd = base_DistGraph::gid2host[base_DistGraph::id].second;
    typename galois::graphs::OfflineGraph::edge_iterator edgeEnd = 
      g.edge_begin(nodeEnd);

    // TODO currently not used; may not be updated
    if (isBipartite) {
  	  uint64_t numNodes_without_edges = (g.size() - numNodes_to_divide);
  	  for (unsigned i = 0; i < base_DistGraph::numHosts; ++i) {
  	    auto p = galois::block_range(0U, 
                   (unsigned)numNodes_without_edges, 
                   i, base_DistGraph::numHosts);
  	    gid2host_withoutEdges.push_back(
          std::make_pair(base_DistGraph::last_nodeID_withEdges_bipartite + p.first + 1, 
                         base_DistGraph::last_nodeID_withEdges_bipartite + p.second + 1));
      }
    }

    uint64_t numEdges_distribute = edgeEnd - edgeBegin; 

    /********************************************
     * Assign edges to the hosts using heuristics
     * and send/recv from other hosts.
     * ******************************************/

    galois::Timer edgeInspectionTimer;
    edgeInspectionTimer.start();

    galois::graphs::BufferedGraph<EdgeTy> mpiGraph;
    mpiGraph.loadPartialGraph(filename, nodeBegin, nodeEnd, *edgeBegin, 
                              *edgeEnd, base_DistGraph::numGlobalNodes,
                              base_DistGraph::numGlobalEdges);

    std::vector<uint64_t> prefixSumOfEdges = assignEdges(mpiGraph, 
        numEdges_distribute, VCutThreshold, base_DistGraph::mirrorNodes, 
        edgeInspectionTimer);

    base_DistGraph::numNodesWithEdges = numNodes;

    if (base_DistGraph::numOwned > 0) {
      base_DistGraph::beginMaster = 
        G2L(base_DistGraph::gid2host[base_DistGraph::id].first);
    } else {
      base_DistGraph::beginMaster = 0;
    }

    // at this point, we know where each edge belongs

    /******************************************
     * Allocate and construct the graph
     *****************************************/

    base_DistGraph::graph.allocateFrom(numNodes, numEdges);
    base_DistGraph::graph.constructNodes();

    auto& base_graph = base_DistGraph::graph;
    galois::do_all(
      galois::iterate((uint32_t)0, numNodes),
      [&] (auto n) {
        base_graph.fixEndEdge(n, prefixSumOfEdges[n]);
      },
      #if MORE_DIST_STATS
      galois::loopname("EdgeLoading"),
      #endif
      galois::no_stats()
    );

    base_DistGraph::printStatistics();

    // load the edges + send them to their assigned hosts if needed
    loadEdges(base_DistGraph::graph, mpiGraph, VCutThreshold);

    mpiGraph.resetAndFree();

    /*******************************************/

    galois::runtime::getHostBarrier().wait();

    if (transpose && (numNodes > 0)) {
      base_DistGraph::graph.transpose(GRNAME);
      base_DistGraph::transposed = true;
    }

    galois::CondStatTimer<MORE_DIST_STATS> Tthread_ranges("ThreadRangesTime",
                                                          GRNAME);

    Tthread_ranges.start();
    base_DistGraph::determineThreadRanges();
    Tthread_ranges.stop();

    base_DistGraph::determineThreadRangesMaster();
    base_DistGraph::determineThreadRangesWithEdges();
    base_DistGraph::initializeSpecificRanges();

    Tgraph_construct.stop();

    /*****************************************
     * Communication PreProcessing:
     * Exchange mirrors and master nodes among
     * hosts
     ****************************************/
    galois::CondStatTimer<MORE_DIST_STATS> Tgraph_construct_comm(
      "GraphCommSetupTime", GRNAME
    );

    Tgraph_construct_comm.start();
    base_DistGraph::setup_communication();
    Tgraph_construct_comm.stop();
  }
private:
  /**
   * Read the edges that this host is responsible for and send them off to
   * the host that they are assigned to if necessary.
   *
   * @param graph In memory representation of the graph: to be used in the
   * program once loading is done
   * @param mpiGraph In memory graph rep of the Galois binary graph: not the
   * final representation used by the program
   * @param VCutThreshold If number of edges on a vertex are over the 
   * threshold, the edges go to the host that contains the destination
   */
  template<typename GraphTy>
  void loadEdges(GraphTy& graph, galois::graphs::BufferedGraph<EdgeTy>& mpiGraph, 
                 uint32_t VCutThreshold) {
    if (base_DistGraph::id == 0) {
      if (std::is_void<typename GraphTy::edge_data_type>::value) {
        galois::gPrint("Loading void edge-data while creating edges\n");
      } else {
        galois::gPrint("Loading edge-data while creating edges\n");
      }
    }

    galois::Timer timer;
    timer.start();
    mpiGraph.resetReadCounters();

    assigned_edges_perhost.resize(base_DistGraph::numHosts);

    std::atomic<uint64_t> edgesToReceive;
    edgesToReceive.store(num_total_edges_to_receive);

    readAndSendEdges(graph, mpiGraph, VCutThreshold, edgesToReceive);

    galois::on_each(
      [&](unsigned tid, unsigned nthreads) {
        receiveAssignedEdges(graph, edgesToReceive);
      });

    base_DistGraph::increment_evilPhase();

    timer.stop();
    galois::gPrint("[", base_DistGraph::id, "] Edge loading time: ", 
                   timer.get_usec()/1000000.0f, " seconds to read ", 
                   mpiGraph.getBytesRead(), " bytes (",
                   mpiGraph.getBytesRead()/(float)timer.get_usec(), " MBPS)\n");
  }

  /**
   * Given an assignment of edges to hosts, send messages to each host
   * informing them of assigned edges/outgoing edges
   *
   * @param assignedEdgesPerHost how many edges from this host are going to be
   * sent to each host
   * @param numOutgoingEdges numOutgoingEdges[hostNum] is an array that tells
   * you how many edges on each node on THIS host (i.e. the caller) that should
   * go to host "hostNum"
   */
  void exchangeAssignedEdgeInfo(
     std::vector<galois::GAccumulator<uint64_t>>& assignedEdgesPerHost,
     std::vector<std::vector<uint64_t>>& numOutgoingEdges
  ) {
    auto& net = galois::runtime::getSystemNetworkInterface();
    num_total_edges_to_receive = 0; 

    /****** Exchange numOutgoingEdges sets *********/
    // send and clear assigned_edges_perhost to receive from other hosts
    for (unsigned x = 0; x < net.Num; ++x) {
      if (x == base_DistGraph::id) continue;
      galois::runtime::SendBuffer b;
      galois::runtime::gSerialize(b, assignedEdgesPerHost[x].reduce());
      galois::runtime::gSerialize(b, numOutgoingEdges[x]);
      net.sendTagged(x, galois::runtime::evilPhase, b);
    }

    net.flush();

    // receive
    for (unsigned x = 0; x < net.Num; ++x) {
      if(x == base_DistGraph::id) continue;

      decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;
      do {
        p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
      } while(!p);

      uint64_t num_edges_from_host = 0;
      galois::runtime::gDeserialize(p->second, num_edges_from_host);
      galois::runtime::gDeserialize(p->second, numOutgoingEdges[p->first]);
      num_total_edges_to_receive += num_edges_from_host;
    }
    base_DistGraph::increment_evilPhase();
  }

  /**
   * Given information about what edges we have been assigned, create the 
   * master and mirror mappings.
   *
   * @param numOutgoingEdges TODO doc
   * @param ghosts bitset that is marked with ghost nodes (output)
   * @param mirrorNodes Edited by this function to have information about which 
   * host the mirror nodes on this host belongs to
   * @returns a prefix sum of the edges for the nodes this partition owns
   */
  std::vector<uint64_t> createMasterMirrorNodes(
      const std::vector<std::vector<uint64_t>>& numOutgoingEdges,
      galois::DynamicBitSet& ghosts,
      std::vector<std::vector<size_t>>& mirrorNodes
  ) {
    numNodes = 0;
    numEdges = 0;

    localToGlobalVector.reserve(base_DistGraph::gid2host[base_DistGraph::id].second - 
                                base_DistGraph::gid2host[base_DistGraph::id].first);
    globalToLocalMap.reserve(base_DistGraph::gid2host[base_DistGraph::id].second - 
                             base_DistGraph::gid2host[base_DistGraph::id].first);

    std::vector<uint64_t> prefixSumOfEdges;

    uint64_t src = 0;
    for (uint32_t i = 0; i < base_DistGraph::numHosts; ++i) {
      for (unsigned j = 0; j < numOutgoingEdges[i].size(); ++j) {
        bool createNode = false;
        if (numOutgoingEdges[i][j] > 0) {
          createNode = true;
          numEdges += numOutgoingEdges[i][j];
        } else if (isOwned(src)) {
          createNode = true;
        }

        if (createNode) {
          localToGlobalVector.push_back(src);
          globalToLocalMap[src] = numNodes++;
          prefixSumOfEdges.push_back(numEdges);
          if(!isOwned(src))
            ghosts.set(src);
        } else if (ghosts.test(src)) {
          localToGlobalVector.push_back(src);
          globalToLocalMap[src] = numNodes++;
          prefixSumOfEdges.push_back(numEdges);
        }
        ++src;
      }
    }

    for (uint64_t x = 0; x < base_DistGraph::numGlobalNodes; ++x){
      if (ghosts.test(x) && !isOwned(x)){
        auto h = find_hostID(x);
        mirrorNodes[h].push_back(x);
      }
    }

    return prefixSumOfEdges;
  }


  /**
   * Goes over the nodes that this host is responsible for loading and 
   * determine which host edges should be assigned to.
   *
   * @param mpiGraph In memory representation of the Galois binary graph
   * file (not the final underlying representation we will use)
   * @param numEdges_distribute Number of edges this host should distributed/
   * accout for; used only for sanity checking
   * @param VCutThreshold If number of edges on a vertex are over the 
   * threshold, the edges go to the host that contains the destination
   * @param mirrorNodes TODO doc
   * @param edgeInspectionTimer
   * @returns a prefix sum of edges for the nodes this host is assigned
   */
  // Go over assigned nodes and determine where edges should go.
  std::vector<uint64_t> assignEdges(
      galois::graphs::BufferedGraph<EdgeTy>& mpiGraph, 
      uint64_t numEdges_distribute, 
      uint32_t VCutThreshold, 
      std::vector<std::vector<size_t>>& mirrorNodes,
      galois::Timer& edgeInspectionTimer
  ) {
    // number of outgoing edges for each node on a particular host that this
    // host is aware of
    std::vector<std::vector<uint64_t>> numOutgoingEdges(base_DistGraph::numHosts);

    // how many edges we will give to a particular host
    std::vector<galois::GAccumulator<uint64_t>> 
        num_assigned_edges_perhost(base_DistGraph::numHosts);

    base_DistGraph::numOwned = base_DistGraph::gid2host[base_DistGraph::id].second -
                            base_DistGraph::gid2host[base_DistGraph::id].first;

    for (uint32_t i = 0; i < base_DistGraph::numHosts; ++i) {
      numOutgoingEdges[i].assign(base_DistGraph::numOwned, 0);
    }

    mpiGraph.resetReadCounters();

    uint64_t globalOffset = base_DistGraph::gid2host[base_DistGraph::id].first;
    auto& id = base_DistGraph::id;

    galois::DynamicBitSet ghosts;
    ghosts.resize(base_DistGraph::numGlobalNodes);

    // Assign edges to hosts
    galois::do_all(
      galois::iterate(base_DistGraph::gid2host[base_DistGraph::id].first,
                      base_DistGraph::gid2host[base_DistGraph::id].second),
      [&] (auto src) {
        auto ee = mpiGraph.edgeBegin(src);
        auto ee_end = mpiGraph.edgeEnd(src);
        auto num_edges = std::distance(ee, ee_end);
        // Assign edges for high degree nodes to the destination
        if (num_edges > VCutThreshold) {
          for(; ee != ee_end; ++ee){
            auto gdst = mpiGraph.edgeDestination(*ee);
            auto h = this->find_hostID(gdst);
            numOutgoingEdges[h][src - globalOffset]++;
            num_assigned_edges_perhost[h] += 1;
          }
        } else {
        // otherwise if not high degree keep all edges with the source node
          for (; ee != ee_end; ++ee) {
            numOutgoingEdges[id][src - globalOffset]++;
            num_assigned_edges_perhost[id] += 1;
            auto gdst = mpiGraph.edgeDestination(*ee);
            if(!this->isOwned(gdst))
              ghosts.set(gdst);
          }
        }
      },
      #if MORE_DIST_STATS
      galois::loopname("EdgeInspection"),
      #endif
      galois::no_stats()
    );

    edgeInspectionTimer.stop();
    galois::gPrint("[", base_DistGraph::id, "] Edge inspection time: ",
                   edgeInspectionTimer.get_usec()/1000000.0f, " seconds to read ",
                   mpiGraph.getBytesRead(), " bytes (",
                   mpiGraph.getBytesRead()/(float)edgeInspectionTimer.get_usec(),
                   " MBPS)\n");

    #ifndef NDEBUG
    uint64_t check_numEdges = 0;
    for (uint32_t h = 0; h < base_DistGraph::numHosts; ++h) {
      check_numEdges += num_assigned_edges_perhost[h].reduce();
    }
    assert(check_numEdges == numEdges_distribute);
    #endif

    // send off messages letting hosts know which edges they were assigned
    exchangeAssignedEdgeInfo(num_assigned_edges_perhost, numOutgoingEdges);
    // create the master/mirror node mapping based on the information
    // received; returns a prefix sum of edges
    return createMasterMirrorNodes(numOutgoingEdges, ghosts, mirrorNodes);
  }


  /**
   * Read/construct edges we are responsible for and send off edges we don't 
   * own to the correct hosts. Non-void variant (i.e. edge data exists).
   *
   * @param graph Final underlying in-memory graph representation to save to
   * @param mpiGraph Graph object that currently contains the in memory
   * representation of the Galois binary
   * @param VCutThreshold If number of edges on a vertex are over the 
   * threshold, the edges go to the host that contains the destination
   * @param edgesToReceive Number of edges that this host needs to receive
   * from other hosts; should have been determined in edge assigning
   * phase
   */
  template<typename GraphTy, 
           typename std::enable_if<
             !std::is_void<typename GraphTy::edge_data_type>::value
           >::type* = nullptr>
  void readAndSendEdges(GraphTy& graph, 
                        galois::graphs::BufferedGraph<EdgeTy>& mpiGraph, 
                        uint32_t VCutThreshold,
                        std::atomic<uint64_t>& edgesToReceive) {
    typedef std::vector<std::vector<uint64_t>> DstVecType;
    galois::substrate::PerThreadStorage<DstVecType> 
        gdst_vecs(base_DistGraph::numHosts);

    typedef std::vector<std::vector<typename GraphTy::edge_data_type>>
        DataVecType;
    galois::substrate::PerThreadStorage<DataVecType> 
        gdata_vecs(base_DistGraph::numHosts);

    typedef std::vector<galois::runtime::SendBuffer> SendBufferVecTy;
    galois::substrate::PerThreadStorage<SendBufferVecTy> 
      sendBuffers(base_DistGraph::numHosts);

    auto& net = galois::runtime::getSystemNetworkInterface();

    const unsigned& id = this->base_DistGraph::id;
    const unsigned& numHosts = this->base_DistGraph::numHosts;

    // TODO find a way to refactor into functions; can reuse in other
    // version as well; too long
    // Go over assigned nodes and distribute edges to other hosts.
    galois::do_all(
      galois::iterate(base_DistGraph::gid2host[base_DistGraph::id].first,
                      base_DistGraph::gid2host[base_DistGraph::id].second),
      [&] (auto src) {
        auto ee = mpiGraph.edgeBegin(src);
        auto ee_end = mpiGraph.edgeEnd(src);

        auto num_edges = std::distance(ee, ee_end);

        uint32_t lsrc = 0;
        uint64_t cur = 0;

        if (this->isLocal(src)) {
          lsrc = this->G2L(src);
          cur = *graph.edge_begin(lsrc, galois::MethodFlag::UNPROTECTED);
        }

        std::vector<std::vector<uint64_t>>& gdst_vec = *gdst_vecs.getLocal();
        auto& gdata_vec = *gdata_vecs.getLocal();

        for (unsigned i = 0; i < numHosts; i++) {
          gdst_vec[i].clear();
          gdata_vec[i].clear();
          //gdst_vec[i].reserve(std::distance(ee, ee_end));
          //gdata_vec[i].reserve(std::distance(ee, ee_end));
        }

        if (num_edges > VCutThreshold) {
          // Assign edges for high degree nodes to the destination
          for (; ee != ee_end; ++ee) {
            auto gdst = mpiGraph.edgeDestination(*ee);
            auto gdata = mpiGraph.edgeData(*ee);
            auto h = this->find_hostID(gdst);
            gdst_vec[h].push_back(gdst);
            gdata_vec[h].push_back(gdata);
          }
        } else {
          // keep all edges with the source node
          for (; ee != ee_end; ++ee) {
            auto gdst = mpiGraph.edgeDestination(*ee);
            auto gdata = mpiGraph.edgeData(*ee);
            assert(this->isLocal(src));
            uint32_t ldst = this->G2L(gdst);
            graph.constructEdge(cur++, ldst, gdata);
          }
          if (this->isLocal(src)) {
            assert(cur == (*graph.edge_end(lsrc)));
          }
        }

        // construct edges for nodes with greater than threashold edges but 
        // assigned to local host
        uint32_t i = 0;
        for (uint64_t gdst : gdst_vec[id]) {
          uint32_t ldst = this->G2L(gdst);
          auto gdata = gdata_vec[id][i++];
          graph.constructEdge(cur++, ldst, gdata);
        }

        // send 
        for (uint32_t h = 0; h < numHosts; ++h) {
          if (h == id) continue;
          if (gdst_vec[h].size()) {
            auto& sendBuffer = (*sendBuffers.getLocal())[h];
            galois::runtime::gSerialize(sendBuffer, src, gdst_vec[h], 
                                        gdata_vec[h]);
            if (sendBuffer.size() > edgePartitionSendBufSize) {
              net.sendTagged(h, galois::runtime::evilPhase, sendBuffer);
              sendBuffer.getVec().clear();
            }
          }
        }

        // Make sure the outgoing edges for this src are constructed
        if (this->isLocal(src)) {
          assert(cur == (*graph.edge_end(lsrc)));
        }

        // TODO don't have to receive every iteration
        auto buffer = net.recieveTagged(galois::runtime::evilPhase, nullptr);
        this->processReceivedEdgeBuffer(buffer, graph, edgesToReceive);
      },
      #if MORE_DIST_STATS
      galois::loopname("EdgeLoading"),
      #endif
      galois::no_stats()
    );

    // flush buffers
    for (unsigned threadNum = 0; threadNum < sendBuffers.size(); ++threadNum) {
      auto& sbr = *sendBuffers.getRemote(threadNum);
      for (unsigned h = 0; h < this->base_DistGraph::numHosts; ++h) {
        if (h == this->base_DistGraph::id) continue;
        auto& sendBuffer = sbr[h];
        if (sendBuffer.size() > 0) {
          net.sendTagged(h, galois::runtime::evilPhase, sendBuffer);
          sendBuffer.getVec().clear();
        }
      }
    }

    net.flush();
  }

  /**
   * Read/construct edges we are responsible for and send off edges we don't 
   * own to the correct hosts. Void variant (i.e. no edge data).
   *
   * @param graph Final underlying in-memory graph representation to save to
   * @param mpiGraph Graph object that currently contains the in memory
   * representation of the Galois binary
   * @param VCutThreshold If number of edges on a vertex are over the 
   * threshold, the edges go to the host that contains the destination
   * @param edgesToReceive Number of edges that this host needs to receive
   * from other hosts; should have been determined in edge assigning
   * phase
   */
  template<typename GraphTy, 
           typename std::enable_if<
             std::is_void<typename GraphTy::edge_data_type>::value
           >::type* = nullptr>
  void readAndSendEdges(GraphTy& graph, 
                        galois::graphs::BufferedGraph<EdgeTy>& mpiGraph, 
                        uint32_t VCutThreshold,
                        std::atomic<uint64_t>& edgesToReceive) {
    auto& net = galois::runtime::getSystemNetworkInterface();

    typedef std::vector<std::vector<uint64_t>> DstVecType;
    galois::substrate::PerThreadStorage<DstVecType> 
        gdst_vecs(base_DistGraph::numHosts);

    const unsigned& id = this->base_DistGraph::id;
    const unsigned& numHosts = this->base_DistGraph::numHosts;
    typedef std::vector<galois::runtime::SendBuffer> SendBufferVecTy;
    galois::substrate::PerThreadStorage<SendBufferVecTy> 
      sendBuffers(base_DistGraph::numHosts);

    // TODO find a way to refactor into functions; can reuse in other
    // version as well; too long
    // Go over assigned nodes and distribute edges to other hosts.
    galois::do_all(
      galois::iterate(base_DistGraph::gid2host[base_DistGraph::id].first,
                      base_DistGraph::gid2host[base_DistGraph::id].second),
      [&] (auto src) {
        auto ee = mpiGraph.edgeBegin(src);
        auto ee_end = mpiGraph.edgeEnd(src);

        auto num_edges = std::distance(ee, ee_end);

        uint32_t lsrc = 0;
        uint64_t cur = 0;

        if (this->isLocal(src)) {
          lsrc = this->G2L(src);
          cur = *graph.edge_begin(lsrc, galois::MethodFlag::UNPROTECTED);
        }

        std::vector<std::vector<uint64_t>>& gdst_vec = *gdst_vecs.getLocal();
        for (unsigned i = 0; i < numHosts; i++) {
          gdst_vec[i].clear();
          //gdst_vec[i].reserve(std::distance(ee, ee_end));
        }

        if (num_edges > VCutThreshold) {
          // Assign edges for high degree nodes to the destination
          for (; ee != ee_end; ++ee) {
            auto gdst = mpiGraph.edgeDestination(*ee);
            auto h = this->find_hostID(gdst);
            gdst_vec[h].push_back(gdst);
          }
        } else {
          // keep all edges with the source node
          for (; ee != ee_end; ++ee) {
            auto gdst = mpiGraph.edgeDestination(*ee);
            assert(this->isLocal(src));
            uint32_t ldst = this->G2L(gdst);
            graph.constructEdge(cur++, ldst);
          }
          if (this->isLocal(src)) {
            assert(cur == (*graph.edge_end(lsrc)));
          }
        }

        // construct edges for nodes with greater than threashold edges but 
        // assigned to local host
        for (uint64_t gdst : gdst_vec[id]) {
          uint32_t ldst = this->G2L(gdst);
          graph.constructEdge(cur++, ldst);
        }

        // send if reached the batch limit
        for (uint32_t h = 0; h < numHosts; ++h) {
          if (h == id) continue;
          if (gdst_vec[h].size()) {
            auto& sendBuffer = (*sendBuffers.getLocal())[h];
            galois::runtime::gSerialize(sendBuffer, src, gdst_vec[h]);
            if (sendBuffer.size() > edgePartitionSendBufSize) {
              net.sendTagged(h, galois::runtime::evilPhase, sendBuffer);
              sendBuffer.getVec().clear();
            }
          }
        }

        // Make sure the outgoing edges for this src are constructed
        if (this->isLocal(src)) {
          assert(cur == (*graph.edge_end(lsrc)));
        }

        // TODO don't have to receive every iteration
        auto buffer = net.recieveTagged(galois::runtime::evilPhase, nullptr);
        this->processReceivedEdgeBuffer(buffer, graph, edgesToReceive);
      },
      #if MORE_DIST_STATS
      galois::loopname("EdgeLoading"),
      #endif
      galois::no_stats()
    );

    // flush buffers
    for (unsigned threadNum = 0; threadNum < sendBuffers.size(); ++threadNum) {
      auto& sbr = *sendBuffers.getRemote(threadNum);
      for (unsigned h = 0; h < this->base_DistGraph::numHosts; ++h) {
        if (h == this->base_DistGraph::id) continue;
        auto& sendBuffer = sbr[h];
        if (sendBuffer.size() > 0) {
          net.sendTagged(h, galois::runtime::evilPhase, sendBuffer);
          sendBuffer.getVec().clear();
        }
      }
    }

    net.flush();
  }

  // optional typedef used below
  template<typename T>
  #if __GNUC__ > 5 || (__GNUC__ == 5 && __GNUC_MINOR__ > 1)
  using optional_t = std::experimental::optional<T>;
  #else
  using optional_t = boost::optional<T>;
  #endif
  /**
   * Given a (possible) buffer, grab the edge data from it and save it
   * to the in-memory graph representation.
   */
  template<typename GraphTy>
  void processReceivedEdgeBuffer(
      optional_t<std::pair<uint32_t, galois::runtime::RecvBuffer>>& buffer,
      GraphTy& graph, std::atomic<uint64_t>& edgesToReceive
  ) {
    if (buffer) {
      auto& receiveBuffer = buffer->second;

      while (receiveBuffer.r_size() > 0) {
        uint64_t _src;
        std::vector<uint64_t> _gdst_vec;
        galois::runtime::gDeserialize(receiveBuffer, _src, _gdst_vec);
        edgesToReceive -= _gdst_vec.size();
        assert(isLocal(_src));
        uint32_t lsrc = G2L(_src);
        uint64_t cur = *graph.edge_begin(lsrc, galois::MethodFlag::UNPROTECTED);
        uint64_t cur_end = *graph.edge_end(lsrc);
        assert((cur_end - cur) == _gdst_vec.size());
        deserializeEdges(graph, receiveBuffer, _gdst_vec, cur, cur_end);
      }
    }
  }

  /**
   * Receive the edge dest/data assigned to this host from other hosts
   * that are responsible for reading it.
   */
  template<typename GraphTy>
  void receiveAssignedEdges(GraphTy& graph, 
                            std::atomic<uint64_t>& edgesToReceive) {
    auto& net = galois::runtime::getSystemNetworkInterface();

    // receive the edges from other hosts
    while (edgesToReceive) {
      decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;
      p = net.recieveTagged(galois::runtime::evilPhase, nullptr);

      processReceivedEdgeBuffer(p, graph, edgesToReceive);
    }
  }

  /**
   * Deserialize received edges and constructs them in our graph. Edge-data
   * variant.
   */
  template<typename GraphTy, 
           typename std::enable_if<
             !std::is_void<typename GraphTy::edge_data_type>::value
           >::type* = nullptr>
  void deserializeEdges(GraphTy& graph, galois::runtime::RecvBuffer& b, 
      const std::vector<uint64_t>& gdst_vec, uint64_t& cur, uint64_t& cur_end) {
    std::vector<typename GraphTy::edge_data_type> gdata_vec;
    galois::runtime::gDeserialize(b, gdata_vec);
    uint64_t i = 0;
    while (cur < cur_end) {
      auto gdata = gdata_vec[i];
      uint64_t gdst = gdst_vec[i++];
      uint32_t ldst = G2L(gdst);
      graph.constructEdge(cur++, ldst, gdata);
    }
  }

  /**
   * Deserialize received edges and constructs them in our graph. No edge-data
   * variant.
   */
  template<typename GraphTy, 
           typename std::enable_if<
             std::is_void<typename GraphTy::edge_data_type>::value
           >::type* = nullptr>
  void deserializeEdges(GraphTy& graph, galois::runtime::RecvBuffer& b, 
      const std::vector<uint64_t>& gdst_vec, uint64_t& cur, uint64_t& cur_end) {
    uint64_t i = 0;
    while (cur < cur_end) {
      uint64_t gdst = gdst_vec[i++];
      uint32_t ldst = G2L(gdst);
      graph.constructEdge(cur++, ldst);
    }
  }

  /**
   * Returns host that a particular node belongs to.
   */
  uint32_t find_hostID(const uint64_t gid) const {
    for (uint32_t h = 0; h < base_DistGraph::numHosts; ++h) {
      if (gid >= base_DistGraph::gid2host[h].first && 
          gid < base_DistGraph::gid2host[h].second) {
        return h;
      } else if (isBipartite && 
                 (gid >= gid2host_withoutEdges[h].first && 
                  gid < gid2host_withoutEdges[h].second)) {
        return h;
      } else {
        continue;
      }
    }
    return -1;
  }

  /**
   * Called by sync: resets the correct portion of the bitset depending on
   * which method of synchronization was used.
   *
   * @param syncType Specifies the synchronization type
   * @param bitset_reset_range Function that resets the bitset given the
   * range to reset
   */
  void reset_bitset(typename base_DistGraph::SyncType syncType, 
                    void (*bitset_reset_range)(size_t, size_t)) const {
    if (base_DistGraph::numOwned != 0) {
      auto endMaster = base_DistGraph::beginMaster + base_DistGraph::numOwned;
      if (syncType == base_DistGraph::syncBroadcast) { // reset masters
        bitset_reset_range(base_DistGraph::beginMaster, endMaster-1);
      } else { // reset mirrors
        assert(syncType == base_DistGraph::syncReduce);
        if (base_DistGraph::beginMaster > 0) {
          bitset_reset_range(0, base_DistGraph::beginMaster - 1);
        }
        if (endMaster < numNodes) {
          bitset_reset_range(endMaster, numNodes - 1);
        }
      }
    }
  }

public:
  /**
   * Indicates that this cut is a vertex cut when called.
   *
   * @returns true as this is a vertex cut
   */
  bool is_vertex_cut() const {
    return true;
  }

  std::vector<std::pair<uint32_t,uint32_t>> getMirrorRanges() const {
    std::vector<std::pair<uint32_t, uint32_t>> mirrorRanges_vec;
    if(base_DistGraph::beginMaster > 0)
      mirrorRanges_vec.push_back(std::make_pair(0, base_DistGraph::beginMaster));
    auto endMaster = base_DistGraph::beginMaster + base_DistGraph::numOwned;
    if (endMaster < numNodes) {
      mirrorRanges_vec.push_back(std::make_pair(endMaster, numNodes));
    }
    return mirrorRanges_vec;
  }

  virtual void boostSerializeLocalGraph(boost::archive::binary_oarchive& ar, 
                                        const unsigned int version = 0) const {
    // unsigned ints
    ar << numNodes;
    // maps and vectors
    ar << localToGlobalVector;
    ar << globalToLocalMap;
  }

  virtual void boostDeSerializeLocalGraph(boost::archive::binary_iarchive& ar, 
                                          const unsigned int version = 0) {
    // unsigned ints
    ar >> numNodes;
    // maps and vectors
    ar >> localToGlobalVector;
    ar >> globalToLocalMap;
  }
};

} // end namespace graphs
} // end namespace galois
#endif
