/** partitioned graph wrapper for edgeCut -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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

#include <vector>
#include <set>
#include <algorithm>
#include <unordered_map>
#include "Galois/Runtime/dGraph.h"
#include "Galois/Runtime/OfflineGraph.h"

//template<typename NodeTy, typename EdgeTy, bool BSPNode = false, bool BSPEdge = false>
//class hGraph;


template<typename NodeTy, typename EdgeTy, bool BSPNode = false, bool BSPEdge = false>
class hGraph_edgeCut : public hGraph<NodeTy, EdgeTy, BSPNode, BSPEdge> {

  public:
    typedef hGraph<NodeTy, EdgeTy, BSPNode, BSPEdge> base_hGraph;
    // GID = ghostMap[LID - numOwned]
    std::vector<uint64_t> ghostMap;
    // LID = GlobalToLocalGhostMap[GID]
    std::unordered_map<uint64_t, uint32_t> GlobalToLocalGhostMap;
    //LID Node owned by host i. Stores ghost nodes from each host.
    std::vector<std::pair<uint32_t, uint32_t> > hostNodes;
    std::vector<std::pair<uint64_t, uint64_t>> gid2host;

    std::vector<std::pair<uint64_t, uint64_t>> gid2host_withoutEdges;
    uint32_t numOwned_withoutEdges;
    bool isBipartite;
    uint64_t last_nodeID_withEdges_bipartite;
    uint64_t globalOffset_bipartite;

    uint64_t globalOffset;
    uint32_t numNodes;
    uint32_t numOwned_withEdges;


    // Return the local offsets for the nodes to host.
    std::pair<uint32_t, uint32_t> nodes_by_host(uint32_t host) const {
      return hostNodes[host];
    }

    // Return the gid offsets assigned to the hosts.
    std::pair<uint64_t, uint64_t> nodes_by_host_G(uint32_t host) const {
      return gid2host[host];
    }
    std::pair<uint64_t, uint64_t> nodes_by_host_bipartite_G(uint32_t host) const {
          return gid2host_withoutEdges[host];
     }


    // Return the ID to which gid belongs after patition.
    unsigned getHostID(uint64_t gid) const {
      for (auto i = 0U; i < hostNodes.size(); ++i) {
        uint64_t start, end;
        std::tie(start, end) = nodes_by_host_G(i);
        if (gid >= start && gid < end) {
          return i;
        }
        if(isBipartite){
          if (gid >= globalOffset_bipartite && gid < globalOffset_bipartite + numOwned_withoutEdges)
		    return i;
        }
      }
      return -1;
    }

    // Return if gid is Owned by local host.
    bool isOwned(uint64_t gid) const {
      return gid >= globalOffset && gid < globalOffset + numOwned_withEdges;
      if(isBipartite){
        if (gid >= globalOffset_bipartite && gid < globalOffset_bipartite + numOwned_withoutEdges)
          return true;
      }
    }

    // Return is gid is present locally (owned or mirror).
    bool isLocal(uint64_t gid) const {
      return isOwned(gid);
      return (GlobalToLocalGhostMap.find(gid) != GlobalToLocalGhostMap.end());
    }


    hGraph_edgeCut(const std::string& filename, 
                   const std::string& partitionFolder, 
                   unsigned host, 
                   unsigned _numHosts, 
                   std::vector<unsigned> scalefactor, 
                   bool transpose = false, 
                   bool bipartite = false) : 
                    base_hGraph(host, _numHosts) /*, uint32_t& _numNodes, uint32_t& _numOwned,uint64_t& _numEdges, uint64_t& _totalNodes, unsigned _id )*/{

      Galois::Statistic statGhostNodes("TotalGhostNodes");
      Galois::StatTimer StatTimer_graph_construct("TIME_GRAPH_CONSTRUCT");
      StatTimer_graph_construct.start();
      Galois::StatTimer StatTimer_graph_construct_comm("TIME_GRAPH_CONSTRUCT_COMM");
      isBipartite = bipartite;
      uint32_t _numNodes;
      uint64_t _numEdges;
      Galois::Graph::OfflineGraph g(filename);

      uint64_t numNodes_to_divide = 0;
      if (isBipartite) {
    	  for (unsigned int n = 0; n < g.size(); ++n){
    		  if(std::distance(g.edge_begin(n), g.edge_end(n))){
                  ++numNodes_to_divide;
                  last_nodeID_withEdges_bipartite = n;
    		  }
    	  }
      }
      else {
    	  numNodes_to_divide = g.size();
      }


      base_hGraph::totalNodes = g.size();
      base_hGraph::totalEdges = g.sizeEdges();
      std::cerr << "[" << base_hGraph::id << "] Total nodes : " << base_hGraph::totalNodes << "\n";
      //compute owners for all nodes
      if (scalefactor.empty() || (base_hGraph::numHosts == 1)) {
        for (unsigned i = 0; i < base_hGraph::numHosts; ++i)
          //gid2host.push_back(Galois::block_range(0U, (unsigned) g.size(), i, base_hGraph::numHosts));
          gid2host.push_back(Galois::block_range(0U, (unsigned) numNodes_to_divide, i, base_hGraph::numHosts));
      } else {
        assert(scalefactor.size() == base_hGraph::numHosts);
        unsigned numBlocks = 0;
        for (unsigned i = 0; i < base_hGraph::numHosts; ++i)
          numBlocks += scalefactor[i];
        std::vector<std::pair<uint64_t, uint64_t>> blocks;
        for (unsigned i = 0; i < numBlocks; ++i)
          //blocks.push_back(Galois::block_range(0U, (unsigned) g.size(), i, numBlocks));
          blocks.push_back(Galois::block_range(0U, (unsigned) numNodes_to_divide, i, numBlocks));
        std::vector<unsigned> prefixSums;
        prefixSums.push_back(0);
        for (unsigned i = 1; i < base_hGraph::numHosts; ++i)
          prefixSums.push_back(prefixSums[i - 1] + scalefactor[i - 1]);
        for (unsigned i = 0; i < base_hGraph::numHosts; ++i) {
          unsigned firstBlock = prefixSums[i];
          unsigned lastBlock = prefixSums[i] + scalefactor[i] - 1;
          gid2host.push_back(std::make_pair(blocks[firstBlock].first, blocks[lastBlock].second));
        }
      }

      numOwned_withEdges = base_hGraph::totalOwnedNodes = base_hGraph::numOwned = (gid2host[base_hGraph::id].second - gid2host[base_hGraph::id].first);
      if(isBipartite){
    	  uint64_t numNodes_without_edges = (g.size() - numNodes_to_divide);
    	  for (unsigned i = 0; i < base_hGraph::numHosts; ++i){
    	    auto p = Galois::block_range(0U, (unsigned)numNodes_without_edges, i, base_hGraph::numHosts);
    	    std::cout << " last node : " << last_nodeID_withEdges_bipartite << ", " << p.first << " , " << p.second << "\n";
    	    gid2host_withoutEdges.push_back(std::make_pair(last_nodeID_withEdges_bipartite + p.first + 1, last_nodeID_withEdges_bipartite + p.second + 1));
    	    globalOffset_bipartite = gid2host_withoutEdges[base_hGraph::id].first;
          }
    	  numOwned_withoutEdges = (gid2host_withoutEdges[base_hGraph::id].second - gid2host_withoutEdges[base_hGraph::id].first);
    	  base_hGraph::totalOwnedNodes = base_hGraph::numOwned = (gid2host[base_hGraph::id].second - gid2host[base_hGraph::id].first) + (gid2host_withoutEdges[base_hGraph::id].second - gid2host_withoutEdges[base_hGraph::id].first);
      }



      globalOffset = gid2host[base_hGraph::id].first;
    std::cerr << "[" << base_hGraph::id << "] Owned nodes: " << base_hGraph::numOwned << "\n";

    _numEdges = g.edge_begin(gid2host[base_hGraph::id].second) - g.edge_begin(gid2host[base_hGraph::id].first); // depends on Offline graph impl
    std::cerr << "[" << base_hGraph::id << "] Total edges : " << _numEdges << "\n";

    std::vector<bool> ghosts(g.size());

    auto ee = g.edge_begin(gid2host[base_hGraph::id].first);
    for (auto n = gid2host[base_hGraph::id].first; n < gid2host[base_hGraph::id].second; ++n) {
      auto ii = ee;
      ee = g.edge_end(n);
      for (; ii < ee; ++ii) {
        ghosts[g.getEdgeDst(ii)] = true;
      }
    }
    std::cerr << "[" << base_hGraph::id << "] Ghost Finding Done " << std::count(ghosts.begin(), ghosts.end(), true) << "\n";

    for (uint64_t x = 0; x < g.size(); ++x)
      if (ghosts[x] && !isOwned(x))
        ghostMap.push_back(x);
    std::cerr << "[" << base_hGraph::id << "] Ghost nodes: " << ghostMap.size() << "\n";

    hostNodes.resize(base_hGraph::numHosts, std::make_pair(~0, ~0));
    GlobalToLocalGhostMap.reserve(ghostMap.size());
    for (unsigned ln = 0; ln < ghostMap.size(); ++ln) {
      unsigned lid = ln + base_hGraph::numOwned;
      auto gid = ghostMap[ln];
      GlobalToLocalGhostMap[gid] = lid;
      bool found = false;
      for (auto h = 0U; h < gid2host.size(); ++h) {
        auto& p = gid2host[h];
        if (gid >= p.first && gid < p.second) {
          hostNodes[h].first = std::min(hostNodes[h].first, lid);
          hostNodes[h].second = lid + 1;
          found = true;
          break;
        }
        else if(isBipartite){
        	auto& p2 = gid2host_withoutEdges[h];
          if(gid >= p2.first && gid < p2.second) {
        	  hostNodes[h].first = std::min(hostNodes[h].first, lid);
         	  hostNodes[h].second = lid + 1;
        	  found = true;
        	  break;
           }
        }

      }
      assert(found);
    }

    numNodes = _numNodes = base_hGraph::numOwned + ghostMap.size();
    assert((uint64_t )base_hGraph::numOwned + (uint64_t )ghostMap.size() == (uint64_t )numNodes);

    base_hGraph::graph.allocateFrom(_numNodes, _numEdges);
    //std::cerr << "Allocate done\n";

    base_hGraph::graph.constructNodes();
    //std::cerr << "Construct nodes done\n";
    loadEdges(base_hGraph::graph, g);
    std::cerr << "Edges loaded \n";

    if (transpose) {
      base_hGraph::graph.transpose();
      base_hGraph::transposed = true;
    }

    fill_slaveNodes(base_hGraph::slaveNodes);
    StatTimer_graph_construct.stop();
    StatTimer_graph_construct_comm.start();
    base_hGraph::setup_communication();
    StatTimer_graph_construct_comm.stop();
  }

  uint32_t G2L(uint64_t gid) const {
    //if (gid >= globalOffset && gid < globalOffset + base_hGraph::numOwned)
    if (gid >= globalOffset && gid < globalOffset + numOwned_withEdges)
      return gid - globalOffset;

    if(isBipartite){
    	if (gid >= globalOffset_bipartite && gid < globalOffset_bipartite + numOwned_withoutEdges)
    	      return gid - globalOffset_bipartite + numOwned_withEdges;
    }

    return GlobalToLocalGhostMap.at(gid);
#if 0
    auto ii = std::lower_bound(ghostMap.begin(), ghostMap.end(), gid);
    assert(*ii == gid);
    return std::distance(ghostMap.begin(), ii) + base_hGraph::numOwned;
#endif
  }

  uint64_t L2G(uint32_t lid) const {
    assert(lid < numNodes);
    //if (lid < base_hGraph::numOwned)
    if (lid < numOwned_withEdges)
      return lid + globalOffset;
    if(isBipartite){
      if(lid >= numOwned_withEdges && lid < base_hGraph::numOwned)
        return lid + globalOffset_bipartite;
    }
    return ghostMap[lid - base_hGraph::numOwned];
  }

  template<typename GraphTy, typename std::enable_if<!std::is_void<typename GraphTy::edge_data_type>::value>::type* = nullptr>
    void loadEdges(GraphTy& graph, Galois::Graph::OfflineGraph& g) {
      fprintf(stderr, "Loading edge-data while creating edges.\n");

      uint64_t cur = 0;
      Galois::Timer timer;
      std::cout <<"["<<base_hGraph::id<<"]PRE :: NumSeeks ";
        g.num_seeks();
        g.reset_seek_counters();
        timer.start();
        auto ee = g.edge_begin(gid2host[base_hGraph::id].first);
        for (auto n = gid2host[base_hGraph::id].first; n < gid2host[base_hGraph::id].second; ++n) {
          auto ii = ee;
          ee = g.edge_end(n);
          for (; ii < ee; ++ii) {
            auto gdst = g.getEdgeDst(ii);
            decltype(gdst) ldst = G2L(gdst);
            auto gdata = g.getEdgeData<typename GraphTy::edge_data_type>(ii);
            graph.constructEdge(cur++, ldst, gdata);
          }
          graph.fixEndEdge(G2L(n), cur);
        }
        // non-owned vertices could also be traversed
        for (uint32_t lid = base_hGraph::numOwned; lid < numNodes; ++lid) {
          graph.fixEndEdge(lid, cur);
        }

        timer.stop();
        std::cout << "EdgeLoading time " << timer.get_usec()/1000000.0f << " seconds\n";
      }

    template<typename GraphTy, typename std::enable_if<std::is_void<typename GraphTy::edge_data_type>::value>::type* = nullptr>
      void loadEdges(GraphTy& graph, Galois::Graph::OfflineGraph& g) {
        std::cout << "n :" << g.size() <<"\n";
        fprintf(stderr, "Loading void edge-data while creating edges.\n");
        uint64_t cur = 0;
        auto ee = g.edge_begin(gid2host[base_hGraph::id].first);
        for (auto n = gid2host[base_hGraph::id].first; n < gid2host[base_hGraph::id].second; ++n) {
          auto ii = ee;
          ee = g.edge_end(n);
          for (; ii < ee; ++ii) {
            auto gdst = g.getEdgeDst(ii);
            decltype(gdst) ldst = G2L(gdst);
            graph.constructEdge(cur++, ldst);
          }
          graph.fixEndEdge(G2L(n), cur);
        }
        // non-owned vertices could also be traversed
        for (uint32_t lid = base_hGraph::numOwned; lid < numNodes; ++lid) {
          graph.fixEndEdge(lid, cur);
        }
      }


    void fill_slaveNodes(std::vector<std::vector<size_t>>& slaveNodes){
      for(uint32_t h = 0; h < hostNodes.size(); ++h){
        uint32_t start, end;
        std::tie(start, end) = nodes_by_host(h);
        for(; start != end; ++start){
          slaveNodes[h].push_back(L2G(start));
        }
      }
    }

    std::string getPartitionFileName(const std::string& filename, const std::string & basename, unsigned hostID, unsigned num_hosts){
      return filename;
    }

    bool is_vertex_cut() const{
      return false;
    }

    uint64_t get_local_total_nodes() const {
      return (base_hGraph::numOwned + base_hGraph::totalSlaveNodes);
    }

};

