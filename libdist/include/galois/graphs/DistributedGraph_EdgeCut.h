/** partitioned graph wrapper for edgeCut -*- C++ -*-
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
 * @section Contains the edge cut functionality to be used in dGraph.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Gurbinder Gill <gurbinder533@gmail.com>
 */
#ifndef _GALOIS_DIST_HGRAPHEC_H
#define _GALOIS_DIST_HGRAPHEC_H

#include "galois/graphs/DistributedGraph.h"

namespace galois {
namespace graphs {

template<typename NodeTy, typename EdgeTy, bool WithInEdges=false, 
         bool isBipartite=false>
class DistGraph_edgeCut 
    : public DistGraph<NodeTy, EdgeTy, WithInEdges> {
  constexpr static const char* const GRNAME = "dGraph_edgeCut";
  public:
    using base_DistGraph = DistGraph<NodeTy, EdgeTy, WithInEdges>;
    // GID = ghostMap[LID - numOwned]
    std::vector<uint64_t> ghostMap;
    // LID = GlobalToLocalGhostMap[GID]
    std::unordered_map<uint64_t, uint32_t> GlobalToLocalGhostMap;
    //LID Node owned by host i. Stores ghost nodes from each host.
    std::vector<std::pair<uint32_t, uint32_t>> hostNodes;

    std::vector<std::pair<uint64_t, uint64_t>> gid2host_withoutEdges;
    uint32_t numOwned_withoutEdges;
    uint64_t globalOffset_bipartite;

    uint64_t globalOffset;
    uint32_t numNodes;

    // Return the local offsets for the nodes to host.
    std::pair<uint32_t, uint32_t> nodes_by_host(uint32_t host) const {
      return hostNodes[host];
    }

    // Return the gid offsets assigned to the hosts.
    std::pair<uint64_t, uint64_t> nodes_by_host_G(uint32_t host) const {
      return base_DistGraph::gid2host[host];
    }
    std::pair<uint64_t, uint64_t> nodes_by_host_bipartite_G(uint32_t host) const {
          return gid2host_withoutEdges[host];
    }


    // Return the ID to which gid belongs after partition.
    unsigned getHostID(uint64_t gid) const {
      for (auto i = 0U; i < hostNodes.size(); ++i) {
        uint64_t start, end;
        std::tie(start, end) = nodes_by_host_G(i);
        if (gid >= start && gid < end) {
          return i;
        }
        if(isBipartite){
          if (gid >= globalOffset_bipartite && 
              gid < globalOffset_bipartite + numOwned_withoutEdges)
        return i;
        }
      }
      return -1;
    }

    // Return if gid is Owned by local host.
    bool isOwned(uint64_t gid) const {
      return gid >= globalOffset && gid < globalOffset + base_DistGraph::numOwned;
      if (isBipartite) {
        if (gid >= globalOffset_bipartite && gid < globalOffset_bipartite + numOwned_withoutEdges)
          return true;
      }
    }

    // Return is gid is present locally (owned or mirror).
    bool isLocal(uint64_t gid) const {
      if (isOwned(gid)) return true;
      return (GlobalToLocalGhostMap.find(gid) != GlobalToLocalGhostMap.end());
    }

    /**
     * Constructor for DistGraph_edgeCut
     */
    DistGraph_edgeCut(const std::string& filename, 
                   const std::string& partitionFolder, 
                   unsigned host, 
                   unsigned _numHosts, 
                   std::vector<unsigned>& scalefactor, 
                   bool transpose = false, 
                   bool readFromFile = false,
                   std::string localGraphFileName = "local_graph") : 
                    base_DistGraph(host, _numHosts) {
      galois::StatTimer Tgraph_construct("TIME_GRAPH_CONSTRUCT", 
                                                  GRNAME);
      Tgraph_construct.start();
      galois::StatTimer Tgraph_construct_comm("TIME_GRAPH_CONSTRUCT_COMM", 
                                                       GRNAME);
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
      galois::graphs::BufferedGraph<EdgeTy> mpiGraph;

      mpiGraph.loadPartialGraph(filename, nodeBegin, nodeEnd, *edgeBegin, 
                                *edgeEnd, base_DistGraph::numGlobalNodes,
                                base_DistGraph::numGlobalEdges);

      mpiGraph.resetReadCounters();

      // vector to hold a prefix sum for use in thread work distribution
      std::vector<uint64_t> prefixSumOfEdges(base_DistGraph::numOwned);

      // loop through all nodes we own and determine ghosts (note a node
      // we own can also be marked a ghost here if there's an outgoing edge to 
      // it)
      // Also determine prefix sums
      auto edgeOffset = mpiGraph.edgeBegin(nodeBegin);

      galois::do_all(galois::iterate(nodeBegin, nodeEnd),
        [&] (auto n) {
          auto ii = mpiGraph.edgeBegin(n);
          auto ee = mpiGraph.edgeEnd(n);
          for (; ii < ee; ++ii) {
            ghosts.set(mpiGraph.edgeDestination(*ii));
          }
          prefixSumOfEdges[n - nodeBegin] = std::distance(edgeOffset, ee);
        },
        galois::no_stats(),
        galois::loopname("EdgeInspection"));

      timer.stop();
      galois::gPrint("[", base_DistGraph::id, "] Edge inspection time: ", timer.get_usec()/1000000.0f, 
          " seconds to read ", mpiGraph.getBytesRead(), " bytes (",
          mpiGraph.getBytesRead()/(float)timer.get_usec(), " MBPS)\n");

      // only nodes we do not own are actual ghosts (i.e. filter the "ghosts"
      // found above)
      for (uint64_t x = 0; x < g.size(); ++x)
        if (ghosts.test(x) && !isOwned(x))
          ghostMap.push_back(x);

      hostNodes.resize(base_DistGraph::numHosts, std::make_pair(~0, ~0));

      // determine on which hosts each ghost nodes resides
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

      base_DistGraph::graph.allocateFrom(_numNodes, _numEdges);

      base_DistGraph::graph.constructNodes();

      auto& base_graph = base_DistGraph::graph;
      galois::do_all(galois::iterate((uint32_t)0, numNodes),
        [&] (auto n) {
          base_graph.fixEndEdge(n, prefixSumOfEdges[n]);
        },
        galois::no_stats(),
        galois::loopname("EdgeLoading"));

      base_DistGraph::printStatistics();

      loadEdges(base_DistGraph::graph, mpiGraph);

      mpiGraph.resetAndFree();
      
      if (transpose) {
        base_DistGraph::graph.transpose(GRNAME);
        base_DistGraph::transposed = true;
      }

      fill_mirrorNodes(base_DistGraph::mirrorNodes);

      // !transpose because tranpose finds thread ranges for you
      if (!transpose) {
        galois::StatTimer Tthread_ranges("TIME_THREAD_RANGES", 
                                                  GRNAME);

        Tthread_ranges.start();

        base_DistGraph::determineThreadRanges(prefixSumOfEdges);

        // experimental test of new thread ranges
        //base_DistGraph::determine_thread_ranges(0, _numNodes, 
        //                              base_DistGraph::graph.getThreadRangesVector());

        Tthread_ranges.stop();
      }

      // find ranges for master + nodes with edges
      base_DistGraph::determineThreadRangesMaster();
      base_DistGraph::determineThreadRangesWithEdges();
      base_DistGraph::initializeSpecificRanges();

      base_DistGraph::constructIncomingEdges();

      Tgraph_construct.stop();

      Tgraph_construct_comm.start();
      base_DistGraph::setup_communication();
      Tgraph_construct_comm.stop();
    }

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

  template<typename GraphTy, typename std::enable_if<!std::is_void<typename GraphTy::edge_data_type>::value>::type* = nullptr>
  void loadEdges(GraphTy& graph, galois::graphs::BufferedGraph<EdgeTy>& mpiGraph) {
    if (base_DistGraph::id == 0) {
      galois::gPrint("Loading edge-data while creating edges\n");
    }

    galois::Timer timer;
    timer.start();
    mpiGraph.resetReadCounters();

    galois::do_all(
      galois::iterate(base_DistGraph::gid2host[base_DistGraph::id].first,
                      base_DistGraph::gid2host[base_DistGraph::id].second),
      [&] (auto n) {
        auto ii = mpiGraph.edgeBegin(n);
        auto ee = mpiGraph.edgeEnd(n);
        uint32_t lsrc = this->G2L(n);
        uint64_t cur = *graph.edge_begin(lsrc, galois::MethodFlag::UNPROTECTED);
        for (; ii < ee; ++ii) {
          auto gdst = mpiGraph.edgeDestination(*ii);
          decltype(gdst) ldst = this->G2L(gdst);
          auto gdata = mpiGraph.edgeData(*ii);
          graph.constructEdge(cur++, ldst, gdata);
        }
        assert(cur == (*graph.edge_end(lsrc)));
      },
      galois::no_stats(),
      galois::loopname("EdgeLoading"));

    timer.stop();
    galois::gPrint("[", base_DistGraph::id, "] Edge loading time: ", timer.get_usec()/1000000.0f, 
        " seconds to read ", mpiGraph.getBytesRead(), " bytes (",
        mpiGraph.getBytesRead()/(float)timer.get_usec(), " MBPS)\n");
  }

  template<typename GraphTy, typename std::enable_if<std::is_void<typename GraphTy::edge_data_type>::value>::type* = nullptr>
  void loadEdges(GraphTy& graph, galois::graphs::BufferedGraph<EdgeTy>& mpiGraph) {
    if (base_DistGraph::id == 0) {
      galois::gPrint("Loading void edge-data while creating edges\n");
    }

    galois::Timer timer;
    timer.start();
    mpiGraph.resetReadCounters();

    galois::do_all(
      galois::iterate(base_DistGraph::gid2host[base_DistGraph::id].first,
                      base_DistGraph::gid2host[base_DistGraph::id].second),
      [&] (auto n) {
        auto ii = mpiGraph.edgeBegin(n);
        auto ee = mpiGraph.edgeEnd(n);
        uint32_t lsrc = this->G2L(n);
        uint64_t cur = *graph.edge_begin(lsrc, galois::MethodFlag::UNPROTECTED);
        for (; ii < ee; ++ii) {
          auto gdst = mpiGraph.edgeDestination(*ii);
          decltype(gdst) ldst = this->G2L(gdst);
          graph.constructEdge(cur++, ldst);
        }
        assert(cur == (*graph.edge_end(lsrc)));
      },
      galois::no_stats(),
      galois::loopname("EdgeLoading"));


    timer.stop();
    galois::gPrint("[", base_DistGraph::id, "] Edge loading time: ", timer.get_usec()/1000000.0f, 
        " seconds to read ", mpiGraph.getBytesRead(), " bytes (",
        mpiGraph.getBytesRead()/(float)timer.get_usec(), " MBPS)\n");
  }

  void fill_mirrorNodes(std::vector<std::vector<size_t>>& mirrorNodes){
    for(uint32_t h = 0; h < hostNodes.size(); ++h){
      uint32_t start, end;
      std::tie(start, end) = nodes_by_host(h);
      for(; start != end; ++start){
        mirrorNodes[h].push_back(L2G(start));
      }
    }
  }

  std::string getPartitionFileName(const std::string& filename, const std::string & basename, unsigned hostID, unsigned num_hosts){
    return filename;
  }

  bool is_vertex_cut() const{
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

#if 0
  /* Reset a field on a mirror nodes */
  template<typename FnTy>
  void reset_mirrorField() const {
    if (base_DistGraph::numOwned < numNodes) {
        galois::do_all(galois::iterate(base_DistGraph::numOwned, numNodes - 1),
            [&](uint32_t n) {
              uint32_t lid = base_DistGraph::mirrorNodes[base_DistGraph::id][n];
              #ifdef __GALOIS_HET_OPENCL__
              CLNodeDataWrapper d = clGraph.getDataW(lid);
              FnTy::reset(lid, d);
              #else
              FnTy::reset(lid, base_DistGraph::getData(lid));
              #endif
            },
            galois::no_stats(),
            galois::loopname(base_DistGraph::get_run_identifier("RESET:MIRRORS").c_str()));
        }
  }
#endif

  std::vector<std::pair<uint32_t,uint32_t>> getMirrorRanges() const {
    std::vector<std::pair<uint32_t, uint32_t>> mirrorRanges_vec;
    if (base_DistGraph::numOwned < numNodes) {
      mirrorRanges_vec.push_back(std::make_pair(base_DistGraph::numOwned, numNodes));
    }
    return mirrorRanges_vec;
  }


  /*
   * This function serializes the local data structures using boost binary archive.
   */
  virtual void boostSerializeLocalGraph(boost::archive::binary_oarchive& ar, const unsigned int version = 0) const {

    //unsigned ints
    ar << numNodes;
    ar << globalOffset;

    //maps and vectors
    ar << ghostMap;
    ar << GlobalToLocalGhostMap;

    //pairs
    ar << hostNodes;

  }

  /*
   * This function DeSerializes the local data structures using boost binary archive.
   */
  virtual void boostDeSerializeLocalGraph(boost::archive::binary_iarchive& ar, const unsigned int version = 0) {

    //unsigned ints
    ar >> numNodes;
    ar >> globalOffset;

    //maps and vectors
    ar >> ghostMap;
    ar >> GlobalToLocalGhostMap;

    //pairs
    ar >> hostNodes;
  }
};

} // end namespace graphs
} // end namespace galois
#endif
