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

template<typename NodeTy, typename EdgeTy, bool BSPNode, bool BSPEdge>
class hGraph {

  typedef realNodeTy std::conditional<BSPNode, std::pair<NodeTy, NodeTy>,NodeTy>::type;
  typedef realEdgeTy std::conditional<BSPNode, std::pair<EdgeTy, EdgeTy>,NodeTy>::type;

  typedef LC_CSR_Graph<realNodeTy, realEdgeTy> GraphTy;

  GraphTy graph;
  bool rount;
  unsigned numOwned; // [0, numOwned) = global nodes owned, thus [numOwned, numNodes are replicas
  unsigned id; // my hostid // FIXME: isn't this just Network::ID?
  //ghost cell ID translation
  std::vector<unsigned> L2G; // GID = L2G[LID - numOwned]
  //GID to owner
  std::vector<unsigned> hostNodes; //[ i-1,i ) -> Node owned by host i
  
 public:

  typedef GraphTy::iterator iterator;
  typedef GraphTy::const_iterator const_iterator;
  typedef GraphTy::local_iterator local_iterator;
  typedef GraphTy::const_local_iterator const_local_iterator;
  typedef GraphTy::edge_iterator edge_iterator;

  hGraph(const std::string& filename, unsigned host, unsinged numHosts) {
    OfflineGraph g(filename);
    auto baseNodes = Galois::block_range(g.begin(), g.end(), host, numHosts);
    std::deque<OfflineGraph::GraphNode> Nodes(baseNodes.first, baseNodes.second);
    
  }

  NodeTy& getData(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    auto& r = graph.getData(N, flag);
    if (BSPNode) {
      return round ? r.first : r.second;
    } else {
      return r;
    }
  }

  EdgeTy& getEdgeData(edge_iterator ni, MethodFlag mflag = MethodFlag::ALL) {
    auto& r = graph.getEdgeData(ni, flag);
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
