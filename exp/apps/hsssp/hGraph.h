/** partitioned graph wrapper -*- C++ -*-
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
 * @section Description
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#include "Galois/gstl.h"
#include "Galois/Graph/LC_CSR_Graph.h"

template<typename NodeTy, typename EdgeTy, bool BSPNode, bool BSPEdge>
class hGraph {

  typedef typename std::conditional<BSPNode, std::pair<NodeTy, NodeTy>,NodeTy>::type realNodeTy;
  typedef typename std::conditional<BSPNode, std::pair<EdgeTy, EdgeTy>,NodeTy>::type realEdgeTy;

  typedef Galois::Graph::LC_CSR_Graph<realNodeTy, realEdgeTy> GraphTy;

  GraphTy graph;
  bool rount;
  unsigned numOwned; // [0, numOwned) = global nodes owned, thus [numOwned, numNodes are replicas
  unsigned id; // my hostid // FIXME: isn't this just Network::ID?
  //ghost cell ID translation
  std::vector<unsigned> L2G; // GID = L2G[LID - numOwned]
  //GID to owner
  std::vector<unsigned> hostNodes; //[ i-1,i ) -> Node owned by host i
  
 public:
  typedef typename GraphTy::GraphNode GraphNode;
  typedef typename GraphTy::iterator iterator;
  typedef typename GraphTy::const_iterator const_iterator;
  typedef typename GraphTy::local_iterator local_iterator;
  typedef typename GraphTy::const_local_iterator const_local_iterator;
  typedef typename GraphTy::edge_iterator edge_iterator;

  hGraph(const std::string& filename, unsigned host, unsigned numHosts) {
    OfflineGraph g(filename);
    auto baseNodes = Galois::block_range(0UL, g.size(), host, numHosts);
    std::set<OfflineGraph::GraphNode> ghosts;
    uint32_t numNodes = baseNodes.second - baseNodes.first;
    uint64_t numEdges = 0;
    for (auto n = baseNodes.first; n < baseNodes.second; ++n) {
      for (auto ii = g.edge_begin(n), ee = g.edge_end(n); ii < ee; ++ii) {
        ++numEdges;
        ghosts.insert(g.getEdgeDst(ii));
      }
    }
    //Logical -> File
    auto trans = [&baseNodes, &ghosts] (uint32_t N) {
      auto num = baseNodes.second - baseNodes.first;
      if (N < num)
        return baseNodes.first + N;
      N -= num;
      auto i = ghosts.begin();
      std::advance(i, N);
      return *i;
    };
    //File -> Logical
    auto inv = [&baseNodes, &ghosts] (uint32_t N) {
      if (N >= baseNodes.first && N < baseNodes.second)
        return N - baseNodes.first;
      auto i = ghosts.find(N);
      return baseNodes.second - baseNodes.first + std::distance(ghosts.begin(), i);
    };
    
    numNodes += ghosts.size();
    graph = decltype(graph)(numNodes, numEdges,
                            [&g, &trans] (uint32_t N) { return std::distance(g.edge_begin(trans(N)), g.edge_end(trans(N))); },
                            [&g, &trans, &inv] (uint32_t N, uint64_t E) { inv(g.getEdgeDst(g.edge_begin(trans(N)) + E)); },
                            [] (uint32_t N, uint64_t E) { return 0; }
                            );
    
  }

  template<typename std::enable_if<BSPNode, int>::type = 0>
  NodeTy& getData(GraphNode N, Galois::MethodFlag mflag = Galois::MethodFlag::ALL) {
    auto& r = graph.getData(N, mflag);
    return round ? r.first : r.second;
  }

  template<typename std::enable_if<!BSPNode, int>::type = 0>
  NodeTy& getData(GraphNode N, Galois::MethodFlag mflag = Galois::MethodFlag::ALL) {
    auto& r = graph.getData(N, mflag);
    return r;
  }

  EdgeTy& getEdgeData(edge_iterator ni, Galois::MethodFlag mflag = Galois::MethodFlag::ALL) {
    auto& r = graph.getEdgeData(ni, mflag);
    if (BSPEdge) {
      return round ? r.first : r.second;
    } else {
      return r;
    }
  }

  GraphNode getEdgeDst(edge_iterator ni) {
    return graph.getEdgeDst(ni);
  }

  size_t size() const { return graph.size(); }
  size_t sizeEdges() const { return graph.sizeEdges(); }

  const_iterator begin() const { return graph.begin(); }
  iterator begin() { return graph.begin(); }
  const_iterator end() const { return graph.begin() + numOwned; }
  iterator end() { return graph.begin() + numOwned; } 

  const_local_iterator local_begin() const;
  local_iterator local_begin();
  const_local_iterator local_end() const;
  local_iterator local_end();

  const_iterator ghost_begin() const { return end(); }
  iterator ghost_begin() { return end(); }
  const_iterator ghost_end() const { return graph.end(); }
  iterator ghost_end() { return graph.end(); }
  
  void sync();
};
