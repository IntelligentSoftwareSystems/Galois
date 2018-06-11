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
 * @file DistributedGraph_EdgeCut.h
 *
 * Implements the edge cut partitioning scheme for DistributedGraph. 
 */
#ifndef _GALOIS_DIST_HGRAPHEC_H
#define _GALOIS_DIST_HGRAPHEC_H

#include "galois/graphs/DistributedGraph.h"

namespace galois {
namespace graphs {

/**
 * Distributed graph that partitions using an edge-cut scheme.
 *
 * @tparam NodeTy type of node data for the graph
 * @tparam EdgeTy type of edge data for the graph
 * @tparam WithInEdges controls whether or not it is possible to store in-edges
 * in addition to outgoing edges in this graph
 * @tparam isBipartite specifies if graph is bipartite
 *
 * @warning isBipartite isn't maintained; use at own risk
 *
 */
template<typename NodeTy, typename EdgeTy, bool WithInEdges=false, 
         bool isBipartite=false>
class DistGraph_edgeCut 
    : public DistGraph<NodeTy, EdgeTy, WithInEdges> {
  constexpr static const char* const GRNAME = "dGraph_edgeCut";

  public:
    //! typedef to base DistGraph class
    using base_DistGraph = DistGraph<NodeTy, EdgeTy, WithInEdges>;
    //! Maps ghosts to gids. GID = ghostMap[LID - numOwned]
    std::vector<uint64_t> ghostMap;
    //! Maps global id to local id (mirror nodes). 
    //! LID = GlobalToLocalGhostMap[GID]
    std::unordered_map<uint64_t, uint32_t> GlobalToLocalGhostMap;
    //! Stores range of ghost nodes from each host.
    std::vector<std::pair<uint32_t, uint32_t>> hostNodes;
    //! used only with bipartite graphs, which aren't actively maintained
    //! @warning bipartite support isn't maintained; use at own risk
    std::vector<std::pair<uint64_t, uint64_t>> gid2host_withoutEdges;

    //! used only with bipartite graphs, which aren't actively maintained
    //! @warning bipartite support isn't maintained; use at own risk
    uint32_t numOwned_withoutEdges;
    //! used only with bipartite graphs, which aren't actively maintained
    //! @warning bipartite support isn't maintained; use at own risk
    uint64_t globalOffset_bipartite;

    //! offset to first node that is a master on this host
    uint64_t globalOffset;
    //! number of nodes that exist locally in this graph
    uint32_t numNodes;

    /**
     * Gets the range of local nodes that belongs to a certain host.
     *
     * @param host host to get local offsets for
     * @returns local offsets for the nodes to host
     */
    std::pair<uint32_t, uint32_t> nodes_by_host(uint32_t host) const {
      return hostNodes[host];
    }

    /**
     * Gets the range of nodes (global ids) that belongs to a certain host.
     *
     * @param host host to get gid offsets for
     * @return the gid offsets assigned to the hosts.
     */
    std::pair<uint64_t, uint64_t> nodes_by_host_G(uint32_t host) const {
      return base_DistGraph::gid2host[host];
    }

    //! Given a global id, return the host where the master is.
    //! @param gid GID of node to get host of
    //! @returns host that owns node with GID
    unsigned getHostID(uint64_t gid) const {
      for (auto i = 0U; i < hostNodes.size(); ++i) {
        uint64_t start, end;
        std::tie(start, end) = nodes_by_host_G(i);
        if (gid >= start && gid < end) {
          return i;
        }

        if (isBipartite) {
          if (gid >= globalOffset_bipartite && 
              gid < globalOffset_bipartite + numOwned_withoutEdges)
            return i;
        }
      }
      return -1;
    }

    //! @copydoc DistGraph::isOwned
    //! @param gid id of node to check ownership of
    bool isOwned(uint64_t gid) const {
      return gid >= globalOffset && gid < globalOffset + base_DistGraph::numOwned;
      if (isBipartite) {
        if (gid >= globalOffset_bipartite && gid < globalOffset_bipartite + numOwned_withoutEdges)
          return true;
      }
    }

    //! @copydoc DistGraph::isLocal
    //! @param gid id of node to check locality of
    bool isLocal(uint64_t gid) const {
      if (isOwned(gid)) return true;
      return (GlobalToLocalGhostMap.find(gid) != GlobalToLocalGhostMap.end());
    }

    /**
     * Constructor for DistGraph_edgeCut.
     *
     * Determines which nodes should be read by this local host and reads
     * only those nodes into memory. Setups communication with other hosts
     * as well.
     *
     * @param filename Graph file to read
     * @param host the host id of the caller
     * @param _numHosts total number of hosts in the system
     * @param scalefactor Specifies if certain hosts should get more nodes
     * than others
     * @param transpose true if graph being read needs to have an in-memory
     * transpose done after reading
     * @param readFromFile true if you want to read the local graph from a file
     * @param localGraphFileName the local file to read if readFromFile is set
     * to true
     *
     * @todo get rid of second argument (the string)
     */
    DistGraph_edgeCut(const std::string& filename, 
                      const std::string&, 
                      unsigned host, 
                      unsigned _numHosts, 
                      std::vector<unsigned>& scalefactor, 
                      bool transpose = false, 
                      bool readFromFile = false,
                      std::string localGraphFileName = "local_graph") 
        : base_DistGraph(host, _numHosts) {
      galois::CondStatTimer<MORE_DIST_STATS> Tgraph_construct(
        "GraphPartitioningTime", GRNAME
      );
      Tgraph_construct.start();
      // if local graph is saved, read from there
      if (readFromFile) {
        galois::gPrint("[", base_DistGraph::id, 
                       "] Reading local graph from file : ", 
                       localGraphFileName, "\n");
        base_DistGraph::read_local_graph_from_file(localGraphFileName);
        Tgraph_construct.stop();
        return;
      }
      uint32_t _numNodes;
      uint64_t _numEdges;

      // only used to determine node splits among hosts; abandonded later
      // for the BufferedGraph
      galois::graphs::OfflineGraph g(filename);
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

      base_DistGraph::numOwned = (nodeEnd - nodeBegin);
      
      // currently not being used, may not be updated
      if (isBipartite) {
        uint64_t numNodes_without_edges = (g.size() - numNodes_to_divide);
        for (unsigned i = 0; i < base_DistGraph::numHosts; ++i) {
          auto p = galois::block_range(
                     0U, (unsigned)numNodes_without_edges, i, 
                     base_DistGraph::numHosts);

          gid2host_withoutEdges.push_back(std::make_pair(base_DistGraph::last_nodeID_withEdges_bipartite + p.first + 1, base_DistGraph::last_nodeID_withEdges_bipartite + p.second + 1));
          globalOffset_bipartite = gid2host_withoutEdges[base_DistGraph::id].first;
        }

        numOwned_withoutEdges = (gid2host_withoutEdges[base_DistGraph::id].second - 
                                 gid2host_withoutEdges[base_DistGraph::id].first);
        base_DistGraph::numOwned = (nodeEnd - nodeBegin) + 
                                   (gid2host_withoutEdges[base_DistGraph::id].second - 
                                    gid2host_withoutEdges[base_DistGraph::id].first);
      }

      globalOffset = nodeBegin;
      _numEdges = edgeEnd - edgeBegin;
            
      galois::DynamicBitSet ghosts;
      ghosts.resize(g.size());

      galois::Timer timer;
      timer.start();

      galois::graphs::BufferedGraph<EdgeTy> bGraph;
      bGraph.loadPartialGraph(filename, nodeBegin, nodeEnd, *edgeBegin, 
                              *edgeEnd, base_DistGraph::numGlobalNodes,
                              base_DistGraph::numGlobalEdges);
      bGraph.resetReadCounters();

      // vector to hold a prefix sum for use in thread work distribution
      std::vector<uint64_t> prefixSumOfEdges(base_DistGraph::numOwned);

      // loop through all nodes we own and determine ghosts (note a node
      // we own can also be marked a ghost here if there's an outgoing edge to 
      // it)
      // Also determine prefix sums
      auto edgeOffset = bGraph.edgeBegin(nodeBegin);

      galois::do_all(galois::iterate(nodeBegin, nodeEnd),
        [&] (auto n) {
          auto ii = bGraph.edgeBegin(n);
          auto ee = bGraph.edgeEnd(n);
          for (; ii < ee; ++ii) {
            ghosts.set(bGraph.edgeDestination(*ii));
          }
          prefixSumOfEdges[n - nodeBegin] = std::distance(edgeOffset, ee);
        },
        #if MORE_DIST_STATS
        galois::loopname("EdgeInspection"),
        #endif
        galois::no_stats()
      );
      timer.stop();

      galois::gPrint("[", base_DistGraph::id, "] Edge inspection time: ", 
                     timer.get_usec()/1000000.0f, " seconds to read ", 
                     bGraph.getBytesRead(), " bytes (", 
                     bGraph.getBytesRead()/(float)timer.get_usec(), " MBPS)\n");

      // only nodes we do not own are actual ghosts (i.e. filter the "ghosts"
      // found above)
      for (uint64_t x = 0; x < g.size(); ++x) {
        if (ghosts.test(x) && !isOwned(x)) {
          ghostMap.push_back(x);
        }
      }

      hostNodes.resize(base_DistGraph::numHosts, std::make_pair(~0, ~0));

      // determine on which hosts each ghost nodes resides and store ranges
      // (e.g. hostNodes[1] will tell me the first node and last node 
      // locally whose masters are on node 1)
      GlobalToLocalGhostMap.reserve(ghostMap.size());
      for (unsigned ln = 0; ln < ghostMap.size(); ++ln) {
        unsigned lid = ln + base_DistGraph::numOwned;
        auto gid = ghostMap[ln];
        GlobalToLocalGhostMap[gid] = lid;

        for (auto h = 0U; h < base_DistGraph::gid2host.size(); ++h) {
          auto& p = base_DistGraph::gid2host[h];
          if (gid >= p.first && gid < p.second) {
            hostNodes[h].first = std::min(hostNodes[h].first, lid);
            hostNodes[h].second = lid + 1;
            break;
          } else if (isBipartite) {
            auto& p2 = gid2host_withoutEdges[h];
            if(gid >= p2.first && gid < p2.second) {
              hostNodes[h].first = std::min(hostNodes[h].first, lid);
              hostNodes[h].second = lid + 1;
              break;
             }
          }
        }
      }

      numNodes =_numNodes = base_DistGraph::numOwned + ghostMap.size();
      assert((uint64_t)base_DistGraph::numOwned + (uint64_t)ghostMap.size() == 
             (uint64_t)numNodes);
      prefixSumOfEdges.resize(_numNodes, prefixSumOfEdges.back());

      // transpose is usually used for incoming edge cuts: this makes it
      // so you consider ghosts as having edges as well (since in IEC ghosts
      // have outgoing edges)
      if (transpose) {
        base_DistGraph::numNodesWithEdges = numNodes;
      } else {
        base_DistGraph::numNodesWithEdges = base_DistGraph::numOwned;
      }

      base_DistGraph::beginMaster = 0;

      // construct the graph
      base_DistGraph::graph.allocateFrom(_numNodes, _numEdges);
      base_DistGraph::graph.constructNodes();

      auto& base_graph = base_DistGraph::graph;
      galois::do_all(galois::iterate((uint32_t)0, numNodes),
        [&] (auto n) {
          base_graph.fixEndEdge(n, prefixSumOfEdges[n]);
        },
        #if MORE_DIST_STATS
        galois::loopname("EdgeLoading"),
        #endif
        galois::no_stats()
      );

      base_DistGraph::printStatistics();
      loadEdges(base_DistGraph::graph, bGraph);
      bGraph.resetAndFree();
      
      if (transpose) {
        base_DistGraph::graph.transpose(GRNAME);
        base_DistGraph::transposed = true;
      }

      fill_mirrorNodes(base_DistGraph::mirrorNodes);

      galois::CondStatTimer<MORE_DIST_STATS> Tthread_ranges("ThreadRangesTime",
                                                            GRNAME);
      Tthread_ranges.start();
      base_DistGraph::determineThreadRanges();
      Tthread_ranges.stop();

      // find ranges for master + nodes with edges
      base_DistGraph::determineThreadRangesMaster();
      base_DistGraph::determineThreadRangesWithEdges();
      base_DistGraph::initializeSpecificRanges();

      base_DistGraph::constructIncomingEdges();

      Tgraph_construct.stop();

      galois::CondStatTimer<MORE_DIST_STATS> Tgraph_construct_comm(
        "GraphCommSetupTime", GRNAME
      );
      Tgraph_construct_comm.start();
      base_DistGraph::setup_communication();
      Tgraph_construct_comm.stop();
    }

  //! @copydoc DistGraph::G2L
  //! @param gid gid to convert to a local id
  uint32_t G2L(uint64_t gid) const {
    if (gid >= globalOffset && gid < globalOffset + base_DistGraph::numOwned)
      return gid - globalOffset;

    if(isBipartite){
      if (gid >= globalOffset_bipartite && gid < globalOffset_bipartite + numOwned_withoutEdges)
            return gid - globalOffset_bipartite + base_DistGraph::numOwned;
    }

    return GlobalToLocalGhostMap.at(gid);
#if 0
    auto ii = std::lower_bound(ghostMap.begin(), ghostMap.end(), gid);
    assert(*ii == gid);
    return std::distance(ghostMap.begin(), ii) + base_DistGraph::numOwned;
#endif
  }

  //! @copydoc DistGraph::L2G
  //! @param lid lid to convert to a global id
  uint64_t L2G(uint32_t lid) const {
    assert(lid < numNodes);
    if (lid < base_DistGraph::numOwned)
      return lid + globalOffset;
    if(isBipartite){
      if(lid >= base_DistGraph::numOwned && lid < base_DistGraph::numOwned)
        return lid + globalOffset_bipartite;
    }
    return ghostMap[lid - base_DistGraph::numOwned];
  }

  /**
   * Given a loaded graph, construct the edges in the DistGraph graph.
   * Variant that constructs edge data as well.
   * 
   * @tparam GraphTy type of graph to construct
   *
   * @param [in,out] graph Graph to construct edges in
   * @param bGraph Buffered graph that has edges to write into graph in memory
   */
  template<typename GraphTy, 
           typename std::enable_if<!std::is_void<typename GraphTy::edge_data_type>::value>::type* = nullptr>
  void loadEdges(GraphTy& graph, galois::graphs::BufferedGraph<EdgeTy>& bGraph) {
    if (base_DistGraph::id == 0) {
      galois::gPrint("Loading edge-data while creating edges\n");
    }

    galois::Timer timer;
    timer.start();
    bGraph.resetReadCounters();

    galois::do_all(
      galois::iterate(base_DistGraph::gid2host[base_DistGraph::id].first,
                      base_DistGraph::gid2host[base_DistGraph::id].second),
      [&] (auto n) {
        auto ii = bGraph.edgeBegin(n);
        auto ee = bGraph.edgeEnd(n);
        uint32_t lsrc = this->G2L(n);
        uint64_t cur = *graph.edge_begin(lsrc, galois::MethodFlag::UNPROTECTED);
        for (; ii < ee; ++ii) {
          auto gdst = bGraph.edgeDestination(*ii);
          decltype(gdst) ldst = this->G2L(gdst);
          auto gdata = bGraph.edgeData(*ii);
          graph.constructEdge(cur++, ldst, gdata);
        }
        assert(cur == (*graph.edge_end(lsrc)));
      },
      #if MORE_DIST_STATS
      galois::loopname("EdgeLoading"),
      #endif
      galois::no_stats()
    );

    timer.stop();
    galois::gPrint("[", base_DistGraph::id, "] Edge loading time: ", 
                   timer.get_usec()/1000000.0f, " seconds to read ", 
                   bGraph.getBytesRead(), " bytes (", 
                   bGraph.getBytesRead()/(float)timer.get_usec(), 
                   " MBPS)\n");
  }

  /**
   * Given a loaded graph, construct the edges in the DistGraph graph.
   * Variant that does not construct edge data.
   * 
   * @tparam GraphTy type of graph to construct
   *
   * @param [in,out] graph Graph to construct edges in
   * @param bGraph Buffered graph that has edges to write into graph in memory
   */
  template<typename GraphTy, 
           typename std::enable_if<std::is_void<typename GraphTy::edge_data_type>::value>::type* = nullptr>
  void loadEdges(GraphTy& graph, galois::graphs::BufferedGraph<EdgeTy>& bGraph) {
    if (base_DistGraph::id == 0) {
      galois::gPrint("Loading void edge-data while creating edges\n");
    }

    galois::Timer timer;
    timer.start();
    bGraph.resetReadCounters();

    galois::do_all(
      galois::iterate(base_DistGraph::gid2host[base_DistGraph::id].first,
                      base_DistGraph::gid2host[base_DistGraph::id].second),
      [&] (auto n) {
        auto ii = bGraph.edgeBegin(n);
        auto ee = bGraph.edgeEnd(n);
        uint32_t lsrc = this->G2L(n);
        uint64_t cur = *graph.edge_begin(lsrc, galois::MethodFlag::UNPROTECTED);
        for (; ii < ee; ++ii) {
          auto gdst = bGraph.edgeDestination(*ii);
          decltype(gdst) ldst = this->G2L(gdst);
          graph.constructEdge(cur++, ldst);
        }
        assert(cur == (*graph.edge_end(lsrc)));
      },
      #if MORE_DIST_STATS
      galois::loopname("EdgeLoading"),
      #endif
      galois::no_stats()
    );


    timer.stop();
    galois::gPrint("[", base_DistGraph::id, "] Edge loading time: ", 
                   timer.get_usec()/1000000.0f, " seconds to read ", 
                   bGraph.getBytesRead(), " bytes (", 
                   bGraph.getBytesRead()/(float)timer.get_usec(), " MBPS)\n");
  }

  /**
   * Fill in the mirror node array with mapping from local to global.
   * 
   * e.g. mirrorNodes[1][0] will tell me that the first local node I have
   * that has a master in node 1 will have whatever ID is located at 
   * [1][0]
   *
   * @param [in,out] mirrorNodes vector to fill with mirror node mapping
   */
  void fill_mirrorNodes(std::vector<std::vector<size_t>>& mirrorNodes) {
    for (uint32_t h = 0; h < hostNodes.size(); ++h) {
      uint32_t start, end;
      std::tie(start, end) = nodes_by_host(h);
      for(; start != end; ++start){
        mirrorNodes[h].push_back(L2G(start));
      }
    }
  }

  bool is_vertex_cut() const {
    return false;
  }

  void reset_bitset(typename base_DistGraph::SyncType syncType, 
                    void (*bitset_reset_range)(size_t, size_t)) const {
    if (syncType == base_DistGraph::syncBroadcast) { // reset masters
      if (base_DistGraph::numOwned > 0) {
        bitset_reset_range(0, base_DistGraph::numOwned - 1);
      }
    } else { // reset mirrors
      assert(syncType == base_DistGraph::syncReduce);
      if (base_DistGraph::numOwned < numNodes) {
        bitset_reset_range(base_DistGraph::numOwned, numNodes - 1);
      }
    }
  }

  std::vector<std::pair<uint32_t,uint32_t>> getMirrorRanges() const {
    std::vector<std::pair<uint32_t, uint32_t>> mirrorRanges_vec;
    if (base_DistGraph::numOwned < numNodes) {
      mirrorRanges_vec.push_back(std::make_pair(base_DistGraph::numOwned, numNodes));
    }
    return mirrorRanges_vec;
  }


  virtual void boostSerializeLocalGraph(boost::archive::binary_oarchive& ar, 
                                        const unsigned int version = 0) const {
    // unsigned ints
    ar << numNodes;
    ar << globalOffset;

    // maps and vectors
    ar << ghostMap;
    ar << GlobalToLocalGhostMap;

    // pairs
    ar << hostNodes;
  }

  virtual void boostDeSerializeLocalGraph(boost::archive::binary_iarchive& ar, 
                                          const unsigned int version = 0) {
    // unsigned ints
    ar >> numNodes;
    ar >> globalOffset;

    // maps and vectors
    ar >> ghostMap;
    ar >> GlobalToLocalGhostMap;

    // pairs
    ar >> hostNodes;
  }
};

} // end namespace graphs
} // end namespace galois
#endif
