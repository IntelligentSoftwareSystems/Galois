/** partitioned graph wrapper for vertexCut -*- C++ -*-
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
 * @section Contains the vertex cut functionality to be used in dGraph.
 *
 * @author Gurbinder Gill <gurbinder533@gmail.com>
 * @author Loc Hoang <l_hoang@utexas.edu>
 */

#ifndef _GALOIS_DIST_HGRAPHHYBRID_H
#define _GALOIS_DIST_HGRAPHHYBRID_H

#include "galois/graphs/DistributedGraph.h"
#include <sstream>

#define BATCH_MSG_SIZE 1000
template<typename NodeTy, typename EdgeTy, bool BSPNode = false, 
         bool BSPEdge = false>
class hGraph_vertexCut : public hGraph<NodeTy, EdgeTy, BSPNode, BSPEdge> {
private:
  constexpr static const char* const GRNAME = "dGraph_hybridCut";

public:
  typedef hGraph<NodeTy, EdgeTy, BSPNode, BSPEdge> base_hGraph;
  /** Utilities for reading partitioned graphs. **/
  struct NodeInfo {
    NodeInfo() 
      : local_id(0), global_id(0), owner_id(0) {}
    NodeInfo(size_t l, size_t g, size_t o) 
      : local_id(l), global_id(g), owner_id(o) {}
    size_t local_id;
    size_t global_id;
    size_t owner_id;
  };

  // To send edges to different hosts: #Src #Dst
  std::vector<std::vector<uint64_t>> assigned_edges_perhost;
  uint64_t num_total_edges_to_receive;

  // GID = localToGlobalVector[LID]
  std::vector<uint64_t> localToGlobalVector;
  // LID = globalToLocalMap[GID]
  std::unordered_map<uint64_t, uint32_t> globalToLocalMap;

  std::vector<uint64_t> numEdges_per_host;
  std::vector<std::pair<uint64_t, uint64_t>> gid2host_withoutEdges;

  uint64_t globalOffset;
  uint32_t numNodes;
  bool isBipartite;
  uint64_t numEdges;

  std::vector<NodeInfo> localToGlobalMap_meta;

  unsigned getHostID(uint64_t gid) const {
    return find_hostID(gid);
  }

  bool isOwned(uint64_t gid) const {
    if (gid >= base_hGraph::gid2host[base_hGraph::id].first && 
        gid < base_hGraph::gid2host[base_hGraph::id].second)
      return true;
    else
      return false;
  }

  virtual bool isLocal(uint64_t gid) const {
    assert(gid < base_hGraph::numGlobalNodes);
    if (isOwned(gid))
      return true;
    return (globalToLocalMap.find(gid) != globalToLocalMap.end());
  }

  virtual uint32_t G2L(uint64_t gid) const {
    assert(isLocal(gid));
    return globalToLocalMap.at(gid);
  }

  virtual uint64_t L2G(uint32_t lid) const {
    return localToGlobalVector[lid];
  }

  bool readMetaFile(const std::string& metaFileName, 
                    std::vector<NodeInfo>& localToGlobalMap_meta) {
    std::ifstream meta_file(metaFileName, std::ifstream::binary);
    if (!meta_file.is_open()) {
      std::cerr << "Unable to open file " << metaFileName << "! Exiting!\n";
      return false;
    }
    size_t num_entries;
    meta_file.read(reinterpret_cast<char*>(&num_entries), sizeof(num_entries));
    galois::gPrint("Partition :: ", " Number of nodes :: ", num_entries, "\n");
    for (size_t i = 0; i < num_entries; ++i) {
      std::pair<size_t, size_t> entry;
      size_t owner;
      meta_file.read(reinterpret_cast<char*>(&entry.first), sizeof(entry.first));
      meta_file.read(reinterpret_cast<char*>(&entry.second), sizeof(entry.second));
      meta_file.read(reinterpret_cast<char*>(&owner), sizeof(owner));
      localToGlobalMap_meta.push_back(NodeInfo(entry.second, entry.first, owner));
    }
    return true;
  }

  std::string getMetaFileName(const std::string & basename, unsigned hostID, 
                              unsigned num_hosts){
    std::string result = basename;
    result+= ".META.";
    result+=std::to_string(hostID);
    result+= ".OF.";
    result+=std::to_string(num_hosts);
    return result;
  }

  std::string getPartitionFileName(const std::string & basename, 
                                   unsigned hostID, unsigned num_hosts) {
    std::string result = basename;
    result+= ".PART.";
    result+=std::to_string(hostID);
    result+= ".OF.";
    result+=std::to_string(num_hosts);
    return result;
  }

  /**
   * Constructor for Vertex Cut
   */
  hGraph_vertexCut(const std::string& filename, 
             const std::string& partitionFolder,
             unsigned host, unsigned _numHosts, 
             std::vector<unsigned>& scalefactor, 
             bool transpose = false, 
             uint32_t VCutThreshold = 100, 
             bool bipartite = false) : base_hGraph(host, _numHosts) {
    if (!scalefactor.empty()) {
      if (base_hGraph::id == 0) {
        galois::gWarn("Scalefactor not supported for PowerLyra (hybrid) vertex-cuts\n");
      }
      scalefactor.clear();
    }

    galois::StatTimer Tgraph_construct(
      "TIME_GRAPH_CONSTRUCT", GRNAME);
    galois::StatTimer Tgraph_construct_comm(
      "TIME_GRAPH_CONSTRUCT_COMM", GRNAME);

    Tgraph_construct.start();

    galois::graphs::OfflineGraph g(filename);
    isBipartite = bipartite;

    base_hGraph::numGlobalNodes = g.size();
    base_hGraph::numGlobalEdges = g.sizeEdges();

    uint64_t numNodes_to_divide = base_hGraph::computeMasters(g, scalefactor, isBipartite);

    // at this point gid2Host has pairs for how to split nodes among
    // hosts; pair has begin and end
    uint64_t nodeBegin = base_hGraph::gid2host[base_hGraph::id].first;
    typename galois::graphs::OfflineGraph::edge_iterator edgeBegin = 
      g.edge_begin(nodeBegin);

    uint64_t nodeEnd = base_hGraph::gid2host[base_hGraph::id].second;
    typename galois::graphs::OfflineGraph::edge_iterator edgeEnd = 
      g.edge_begin(nodeEnd);

    // TODO
    // currently not used; may not be updated
    if (isBipartite) {
  	  uint64_t numNodes_without_edges = (g.size() - numNodes_to_divide);
  	  for (unsigned i = 0; i < base_hGraph::numHosts; ++i) {
  	    auto p = galois::block_range(0U, 
                   (unsigned)numNodes_without_edges, 
                   i, base_hGraph::numHosts);
  	    gid2host_withoutEdges.push_back(
          std::make_pair(base_hGraph::last_nodeID_withEdges_bipartite + p.first + 1, base_hGraph::last_nodeID_withEdges_bipartite + p.second + 1));
      }
    }

    uint64_t numEdges_distribute = edgeEnd - edgeBegin; 

    /********************************************
     * Assign edges to the hosts using heuristics
     * and send/recv from other hosts.
     * ******************************************/
    std::vector<uint64_t> prefixSumOfEdges;

    galois::Timer edgeInspectionTimer;
    edgeInspectionTimer.start();

    galois::graphs::MPIGraph<EdgeTy> mpiGraph;
    mpiGraph.loadPartialGraph(filename, nodeBegin, nodeEnd, *edgeBegin, 
                              *edgeEnd, base_hGraph::numGlobalNodes,
                              base_hGraph::numGlobalEdges);

    assignEdges(mpiGraph, numEdges_distribute, VCutThreshold, prefixSumOfEdges,
                base_hGraph::mirrorNodes, edgeInspectionTimer);

    base_hGraph::numNodesWithEdges = numNodes;

    if (base_hGraph::numOwned > 0) {
      base_hGraph::beginMaster = 
        G2L(base_hGraph::gid2host[base_hGraph::id].first);
    } else {
      base_hGraph::beginMaster = 0;
    }

    // at this point, we know where each edge belongs

    /******************************************
     * Allocate and construct the graph
     *****************************************/

    base_hGraph::graph.allocateFrom(numNodes, numEdges);
    base_hGraph::graph.constructNodes();

    auto& base_graph = base_hGraph::graph;
    galois::do_all(
      galois::iterate((uint32_t)0, numNodes),
      [&] (auto n) {
        base_graph.fixEndEdge(n, prefixSumOfEdges[n]);
      },
      galois::loopname("EdgeLoading"),
      galois::timeit(),
      galois::no_stats()
    );

    base_hGraph::printStatistics();

    loadEdges(base_hGraph::graph, mpiGraph, numEdges_distribute, VCutThreshold);

    /*******************************************/

    galois::runtime::getHostBarrier().wait();

    if (transpose && (numNodes > 0)) {
      base_hGraph::graph.transpose(GRNAME);
      base_hGraph::transposed = true;
    } else {
      // else because transpose will find thread ranges for you
      galois::StatTimer Tthread_ranges("TIME_THREAD_RANGES", GRNAME);

      Tthread_ranges.start();
      base_hGraph::determine_thread_ranges(numNodes, prefixSumOfEdges);
      Tthread_ranges.stop();
    }

    base_hGraph::determine_thread_ranges_master();
    base_hGraph::determine_thread_ranges_with_edges();
    base_hGraph::initialize_specific_ranges();

    Tgraph_construct.stop();

    /*****************************************
     * Communication PreProcessing:
     * Exchange mirrors and master nodes among
     * hosts
     ****************************************/
    Tgraph_construct_comm.start();
    base_hGraph::setup_communication();
    Tgraph_construct_comm.stop();
  }
private:
  /**
   * Read the edges that this host is responsible for and send them off to
   * the host that they are assigned to if necessary.
   *
   * TODO params
   */
  template<typename GraphTy>
  void loadEdges(GraphTy& graph, galois::graphs::MPIGraph<EdgeTy>& mpiGraph, 
                 uint64_t numEdges_distribute, uint32_t VCutThreshold) {
    if (base_hGraph::id == 0) {
      if (std::is_void<typename GraphTy::edge_data_type>::value) {
        galois::gPrint("Loading void edge-data while creating edges\n");
      } else {
        galois::gPrint("Loading edge-data while creating edges\n");
      }
    }

    galois::Timer timer;
    timer.start();
    mpiGraph.resetReadCounters();

    assigned_edges_perhost.resize(base_hGraph::numHosts);

    readAndSendEdges(graph, mpiGraph, numEdges_distribute, VCutThreshold);
    std::atomic<uint64_t> edgesToReceive;
    edgesToReceive.store(num_total_edges_to_receive);

    galois::on_each(
      [&](unsigned tid, unsigned nthreads) {
        receiveAssignedEdges(graph, edgesToReceive);
      },
      galois::no_stats()
    );

    ++galois::runtime::evilPhase;

    timer.stop();
    galois::gPrint("[", base_hGraph::id, "] Edge loading time: ", 
                   timer.get_usec()/1000000.0f, " seconds to read ", 
                   mpiGraph.getBytesRead(), " bytes (",
                   mpiGraph.getBytesRead()/(float)timer.get_usec(), " MBPS)\n");
  }

  /**
   * Given an assignment of edges to hosts, send messages to each host
   * informing them of assigned edges/outgoing edges
   *
   * TODO params
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
      if (x == base_hGraph::id) continue;
      galois::runtime::SendBuffer b;
      galois::runtime::gSerialize(b, assignedEdgesPerHost[x].reduce());
      galois::runtime::gSerialize(b, numOutgoingEdges[x]);
      net.sendTagged(x, galois::runtime::evilPhase, b);
    }

    net.flush();

    // receive
    for (unsigned x = 0; x < net.Num; ++x) {
      if(x == base_hGraph::id) continue;

      decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;
      do {
        net.handleReceives();
        p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
      } while(!p);

      uint64_t num_edges_from_host = 0;
      galois::runtime::gDeserialize(p->second, num_edges_from_host);
      galois::runtime::gDeserialize(p->second, numOutgoingEdges[p->first]);
      num_total_edges_to_receive += num_edges_from_host;
    }
    ++galois::runtime::evilPhase;
  }

  /**
   * Given information about what edges we have been assigned, create the 
   * master and mirror mappings.
   *
   * TODO params
   */
  void createMasterMirrorNodes(
      std::vector<std::vector<uint64_t>>& numOutgoingEdges,
      galois::DynamicBitSet& ghosts,
      std::vector<uint64_t>& prefixSumOfEdges,
      std::vector<std::vector<size_t>>& mirrorNodes
  ) {
    numNodes = 0;
    numEdges = 0;

    localToGlobalVector.reserve(base_hGraph::gid2host[base_hGraph::id].second - 
                                base_hGraph::gid2host[base_hGraph::id].first);
    globalToLocalMap.reserve(base_hGraph::gid2host[base_hGraph::id].second - 
                             base_hGraph::gid2host[base_hGraph::id].first);
    uint64_t src = 0;
    for (uint32_t i = 0; i < base_hGraph::numHosts; ++i) {
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

    for (uint64_t x = 0; x < base_hGraph::numGlobalNodes; ++x){
      if (ghosts.test(x) && !isOwned(x)){
        auto h = find_hostID(x);
        mirrorNodes[h].push_back(x);
      }
    }
  }

  // Go over assigned nodes and determine where edges should go.
  void assignEdges(galois::graphs::MPIGraph<EdgeTy>& mpiGraph, 
                   uint64_t numEdges_distribute, 
                   uint32_t VCutThreshold, 
                   std::vector<uint64_t>& prefixSumOfEdges, 
                   std::vector<std::vector<size_t>>& mirrorNodes,
                   galois::Timer& edgeInspectionTimer) {
    galois::DynamicBitSet ghosts;
    ghosts.resize(base_hGraph::numGlobalNodes);

    // number of outgoing edges for each node on a particular host that this
    // host is aware of
    std::vector<std::vector<uint64_t>> numOutgoingEdges(base_hGraph::numHosts);

    // how many edges we will give to a particular host
    std::vector<galois::GAccumulator<uint64_t>> 
        num_assigned_edges_perhost(base_hGraph::numHosts);

    base_hGraph::numOwned = base_hGraph::gid2host[base_hGraph::id].second -
                            base_hGraph::gid2host[base_hGraph::id].first;

    for (uint32_t i = 0; i < base_hGraph::numHosts; ++i) {
      numOutgoingEdges[i].assign(base_hGraph::numOwned, 0);
    }

    mpiGraph.resetReadCounters();

    uint64_t globalOffset = base_hGraph::gid2host[base_hGraph::id].first;
    auto& id = base_hGraph::id;
    // Assign edges to hosts
    galois::do_all(
      galois::iterate(base_hGraph::gid2host[base_hGraph::id].first,
                      base_hGraph::gid2host[base_hGraph::id].second),
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
          for(; ee != ee_end; ++ee) {
            numOutgoingEdges[id][src - globalOffset]++;
            num_assigned_edges_perhost[id] += 1;
            auto gdst = mpiGraph.edgeDestination(*ee);
            if(!this->isOwned(gdst))
              ghosts.set(gdst);
          }
        }
      },
      galois::loopname("EdgeInspection"),
      galois::timeit(),
      galois::no_stats()
    );

    edgeInspectionTimer.stop();
    galois::gPrint("[", base_hGraph::id, "] Edge inspection time: ",
                   edgeInspectionTimer.get_usec()/1000000.0f, " seconds to read ",
                   mpiGraph.getBytesRead(), " bytes (",
                   mpiGraph.getBytesRead()/(float)edgeInspectionTimer.get_usec(),
                   " MBPS)\n");

    #ifndef NDEBUG
    uint64_t check_numEdges = 0;
    for (uint32_t h = 0; h < base_hGraph::numHosts; ++h) {
      check_numEdges += num_assigned_edges_perhost[h].reduce();
    }
    assert(check_numEdges == numEdges_distribute);
    #endif

    // send off messages letting hosts know which edges they were assigned
    exchangeAssignedEdgeInfo(num_assigned_edges_perhost, numOutgoingEdges);
    // create the master/mirror node mapping based on the information
    // received
    createMasterMirrorNodes(numOutgoingEdges, ghosts, prefixSumOfEdges, 
                            mirrorNodes);
  }


  /**
   * Read/construct edges we are responsible for and send off edges we don't 
   * own to the correct hosts. Non-void variant (i.e. edge data exists).
   *
   * TODO params
   */
  template<typename GraphTy, 
           typename std::enable_if<
             !std::is_void<typename GraphTy::edge_data_type>::value
           >::type* = nullptr>
  void readAndSendEdges(GraphTy& graph, 
                        galois::graphs::MPIGraph<EdgeTy>& mpiGraph, 
                        uint64_t numEdges_distribute, 
                        uint32_t VCutThreshold) {
    typedef std::vector<std::vector<uint64_t>> DstVecType;
    galois::substrate::PerThreadStorage<DstVecType> 
        gdst_vecs(base_hGraph::numHosts);

    typedef std::vector<std::vector<typename GraphTy::edge_data_type>>
        DataVecType;
    galois::substrate::PerThreadStorage<DataVecType> 
        gdata_vecs(base_hGraph::numHosts);


    auto& net = galois::runtime::getSystemNetworkInterface();

    const unsigned& id = this->base_hGraph::id;
    const unsigned& numHosts = this->base_hGraph::numHosts;
    // Go over assigned nodes and distribute edges to other hosts.
    galois::do_all(
      galois::iterate(base_hGraph::gid2host[base_hGraph::id].first,
                      base_hGraph::gid2host[base_hGraph::id].second),
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
          gdst_vec[i].reserve(std::distance(ee, ee_end));
          gdata_vec[i].reserve(std::distance(ee, ee_end));
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
            galois::runtime::SendBuffer b;
            galois::runtime::gSerialize(b, src, gdst_vec[h], gdata_vec[h]);
            net.sendTagged(h, galois::runtime::evilPhase, b);
          }
        }

        // Make sure the outgoing edges for this src are constructed
        if (this->isLocal(src)) {
          assert(cur == (*graph.edge_end(lsrc)));
        }
      },
      galois::loopname("EdgeLoading"),
      galois::no_stats(),
      galois::timeit()
    );
    net.flush();
  }

  /**
   * Read/construct edges we are responsible for and send off edges we don't 
   * own to the correct hosts. Void variant (i.e. no edge data).
   *
   * TODO params
   */
  template<typename GraphTy, 
           typename std::enable_if<
             std::is_void<typename GraphTy::edge_data_type>::value
           >::type* = nullptr>
  void readAndSendEdges(GraphTy& graph, 
                        galois::graphs::MPIGraph<EdgeTy>& mpiGraph, 
                        uint64_t numEdges_distribute, 
                        uint32_t VCutThreshold) {
    auto& net = galois::runtime::getSystemNetworkInterface();

    typedef std::vector<std::vector<uint64_t>> DstVecType;
    galois::substrate::PerThreadStorage<DstVecType> 
        gdst_vecs(base_hGraph::numHosts);

    const unsigned& id = this->base_hGraph::id;
    const unsigned& numHosts = this->base_hGraph::numHosts;
    // Go over assigned nodes and distribute edges to other hosts.
    galois::do_all(
      galois::iterate(base_hGraph::gid2host[base_hGraph::id].first,
                      base_hGraph::gid2host[base_hGraph::id].second),
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
          gdst_vec[i].reserve(std::distance(ee, ee_end));
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
            galois::runtime::SendBuffer b;
            galois::runtime::gSerialize(b, src, gdst_vec[h]);
            net.sendTagged(h, galois::runtime::evilPhase, b);
          }
        }

        // Make sure the outgoing edges for this src are constructed
        if (this->isLocal(src)) {
          assert(cur == (*graph.edge_end(lsrc)));
        }
      },
      galois::loopname("EdgeLoading"),
      galois::no_stats(),
      galois::timeit()
    );

    net.flush();
  }


  /**
   * Receive the edge dest/data assigned to this host from other hosts
   * that are responsible for reading it.
   *
   * TODO params
   */
  template<typename GraphTy>
  void receiveAssignedEdges(GraphTy& graph, 
                            std::atomic<uint64_t>& edgesToReceive) {
    auto& net = galois::runtime::getSystemNetworkInterface();

    // receive the edges from other hosts
    while (edgesToReceive) {
      decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;
      net.handleReceives();
      p = net.recieveTagged(galois::runtime::evilPhase, nullptr);

      if (p) {
        std::vector<uint64_t> _gdst_vec;
        uint64_t _src;
        galois::runtime::gDeserialize(p->second, _src, _gdst_vec);
        edgesToReceive -= _gdst_vec.size();
        assert(isLocal(_src));
        uint32_t lsrc = G2L(_src);
        uint64_t cur = *graph.edge_begin(lsrc, galois::MethodFlag::UNPROTECTED);
        uint64_t cur_end = *graph.edge_end(lsrc);
        assert((cur_end - cur) == _gdst_vec.size());
        deserializeEdges(graph, p->second, _gdst_vec, cur, cur_end);
      }
    }
  }

  template<typename GraphTy, 
           typename std::enable_if<
             !std::is_void<typename GraphTy::edge_data_type>::value
           >::type* = nullptr>
  void deserializeEdges(GraphTy& graph, galois::runtime::RecvBuffer& b, 
            std::vector<uint64_t>& gdst_vec, uint64_t& cur, uint64_t& cur_end) {
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

  template<typename GraphTy, 
           typename std::enable_if<
             std::is_void<typename GraphTy::edge_data_type>::value
           >::type* = nullptr>
  void deserializeEdges(GraphTy& graph, galois::runtime::RecvBuffer& b, 
            std::vector<uint64_t>& gdst_vec, uint64_t& cur, uint64_t& cur_end) {
    uint64_t i = 0;
    while (cur < cur_end) {
      uint64_t gdst = gdst_vec[i++];
      uint32_t ldst = G2L(gdst);
      graph.constructEdge(cur++, ldst);
    }
  }

  uint32_t find_hostID(const uint64_t gid) const {
    for (uint32_t h = 0; h < base_hGraph::numHosts; ++h) {
      if (gid >= base_hGraph::gid2host[h].first && 
          gid < base_hGraph::gid2host[h].second) {
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

public:
  bool is_vertex_cut() const {
    return true;
  }

  void reset_bitset(typename base_hGraph::SyncType syncType, 
                    void (*bitset_reset_range)(size_t, size_t)) const {
    if (base_hGraph::numOwned != 0) {
      auto endMaster = base_hGraph::beginMaster + base_hGraph::numOwned;
      if (syncType == base_hGraph::syncBroadcast) { // reset masters
        bitset_reset_range(base_hGraph::beginMaster, endMaster-1);
      } else { // reset mirrors
        assert(syncType == base_hGraph::syncReduce);
        if (base_hGraph::beginMaster > 0) {
          bitset_reset_range(0, base_hGraph::beginMaster - 1);
        }
        if (endMaster < numNodes) {
          bitset_reset_range(endMaster, numNodes - 1);
        }
      }
    }
  }
};
#endif
