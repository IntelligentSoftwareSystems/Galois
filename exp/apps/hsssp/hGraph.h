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

class syncable {
protected:
  static syncable* curSync;
  static void syncRecv(Galois::Runtime::RecvBuffer& buf) {
    curSync->syncRecvImpl(buf);
  }
  virtual void syncRecvImpl(Galois::Runtime::RecvBuffer&) = 0;
};

template<typename NodeTy, typename EdgeTy, bool BSPNode=false, bool BSPEdge=false>
class hGraph : public syncable{

  typedef typename std::conditional<BSPNode, std::pair<NodeTy, NodeTy>,NodeTy>::type realNodeTy;
  typedef typename std::conditional<BSPNode, std::pair<EdgeTy, EdgeTy>,EdgeTy>::type realEdgeTy;

  typedef Galois::Graph::LC_CSR_Graph<realNodeTy, realEdgeTy> GraphTy;

  GraphTy graph;
  bool rount;
  uint32_t numOwned; // [0, numOwned) = global nodes owned, thus [numOwned, numNodes are replicas
  unsigned id; // my hostid // FIXME: isn't this just Network::ID?
  //ghost cell ID translation
  std::vector<uint32_t> L2G; // GID = L2G[LID - numOwned]
  //GID to owner
  std::vector<uint32_t> hostNodes; //[ i-1,i ) -> GID Node owned by host i

  std::pair<uint32_t, uint32_t> nodes_by_host(uint32_t host) {
    if (host == id)
      return std::make_pair(0, numOwned);
    if (host == 0)
      return std::make_pair(numOwned, hostNodes[0]);
    return std::make_pair(hostNodes[host-1], hostNodes[host]);
  }

  template<bool en, typename std::enable_if<en>::type* = nullptr>
  NodeTy& getDataImpl(typename GraphTy::GraphNode N, Galois::MethodFlag mflag = Galois::MethodFlag::ALL) {
    auto& r = graph.getData(N, mflag);
    return round ? r.first : r.second;
  }

  template<bool en, typename std::enable_if<!en>::type* = nullptr>
  NodeTy& getDataImpl(typename GraphTy::GraphNode N, Galois::MethodFlag mflag = Galois::MethodFlag::ALL) {
    auto& r = graph.getData(N, mflag);
    return r;
  }

  template<bool en, typename std::enable_if<en>::type* = nullptr>
  typename GraphTy::edge_data_reference getEdgeDataImpl(typename GraphTy::edge_iterator ni, Galois::MethodFlag mflag = Galois::MethodFlag::ALL) {
    auto r = graph.getEdgeData(ni, mflag);
    return round ? r.first : r.second;
  }

  template<bool en, typename std::enable_if<!en>::type* = nullptr>
  typename GraphTy::edge_data_reference getEdgeDataImpl(typename GraphTy::edge_iterator ni, Galois::MethodFlag mflag = Galois::MethodFlag::ALL) {
    auto r = graph.getEdgeData(ni, mflag);
    return r;
  }

public:
  virtual void syncRecvImpl(Galois::Runtime::RecvBuffer& buf) {
    void (hGraph::*fn)(Galois::Runtime::RecvBuffer&);
    Galois::Runtime::gDeserialize(buf, fn);
    (this->*fn)(buf);
  }

  template<typename FnTy>
  void syncRecvApply(Galois::Runtime::RecvBuffer& buf) {
    uint32_t num;
    Galois::Runtime::gDeserialize(buf, num);
    for(; num ; --num) {
      uint32_t gid;
      typename FnTy::ValTy val;
      Galois::Runtime::gDeserialize(buf, gid, val);
      FnTy::reduce(getData(gid - numOwned), val);
    }
  }
  
 public:
  typedef typename GraphTy::GraphNode GraphNode;
  typedef typename GraphTy::iterator iterator;
  typedef typename GraphTy::const_iterator const_iterator;
  typedef typename GraphTy::local_iterator local_iterator;
  typedef typename GraphTy::const_local_iterator const_local_iterator;
  typedef typename GraphTy::edge_iterator edge_iterator;

  hGraph(const std::string& filename, unsigned host, unsigned numHosts) {
    OfflineGraph g(filename);
    std::cerr << "Offline Graph Done\n";
    auto baseNodes = Galois::block_range(0U, (unsigned)g.size(), host, numHosts);
    numOwned = baseNodes.second - baseNodes.first;
    id = host;
    uint64_t numEdges = g.edge_begin(baseNodes.second) - g.edge_begin(baseNodes.first); // depends on Offline graph impl
    std::cerr << "Edge count Done\n";
    std::vector<bool> ghosts(g.size());
    for (auto n = baseNodes.first; n < baseNodes.second; ++n) {
      for (auto ii = g.edge_begin(n), ee = g.edge_end(n); ii < ee; ++ii) {
        auto dst = g.getEdgeDst(ii);
        ghosts[dst] = true;
      }
    }
    std::cerr << "Ghost Finding Done\n";
    for (uint64_t x = 0; x < g.size(); ++x) {
      if (ghosts[x] && (x < baseNodes.first || x >= baseNodes.second)) {
        L2G.push_back(x);
      }
    }
    std::cerr << "L2G Done\n";

    uint32_t numNodes = numOwned + L2G.size();
    graph.allocateFrom(numNodes, numEdges);
    std::cerr << "Allocate done\n";
    
    graph.constructNodes();
    std::cerr << "Construct nodes done\n";

    uint64_t cur = 0;
    for (auto n = baseNodes.first; n < baseNodes.second; ++n) {
      for (auto ii = g.edge_begin(n), ee = g.edge_end(n); ii < ee; ++ii) {
        auto dst = g.getEdgeDst(ii);
        decltype(dst) rdst;
        if (dst < baseNodes.first || dst >= baseNodes.second) {
          auto i = std::lower_bound(L2G.begin(), L2G.end(), dst);
          rdst = baseNodes.second - baseNodes.first + std::distance(L2G.begin(), i);
        } else {
          rdst = dst - baseNodes.first;
        }
        graph.constructEdge(cur++, rdst);
      }
      graph.fixEndEdge(n, cur);
    }
    std::cerr << "Construct edges done\n";
    
    /*
    auto ii_f = g.begin() + baseNodes.first;
    auto ii_m = graph.begin();
    for (auto x = baseNodes.first; x < baseNodes.second; ++x) {
      auto nf = std::distance(g.edge_begin(*ii_f), g.edge_end(*ii_f));
      auto nm = std::distance(graph.edge_begin(*ii_m), graph.edge_end(*ii_m));
      ++ii_f;
      ++ii_m;
      std::cout << x << " " << nf << " " << nm << "\n";
    }
    */
    //FIXME: hostNodes
  }

  NodeTy& getData(GraphNode N, Galois::MethodFlag mflag = Galois::MethodFlag::ALL) {
    return getDataImpl<BSPNode>(N, mflag);
  }

  typename GraphTy::edge_data_reference getEdgeData(edge_iterator ni, Galois::MethodFlag mflag = Galois::MethodFlag::ALL) {
    return getEdgeDataImpl<BSPEdge>(ni, mflag);
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

  const_iterator ghost_begin() const { return end; }
  iterator ghost_begin() { return end; }
  const_iterator ghost_end() const { return graph.end(); }
  iterator ghost_end() { return graph.end(); }
  
  template<typename FnTy>
  void sync_push() {
    curSync = this;
   void (hGraph::*fn)(Galois::Runtime::RecvBuffer&) = &hGraph::syncRecvApply<FnTy>;
    auto& net = Galois::Runtime::getSystemNetworkInterface();
    for (unsigned x = 0; x < hostNodes.size(); ++x) {
      if (x == id) continue;
      uint32_t start, end;
      std::tie(start, end) = nodes_by_host(x);
      if (start == end) continue;
      Galois::Runtime::SendBuffer b;
      gSerialize(b, fn, (uint32_t)(end-start));
      for (; start != end; ++start) {
        auto gid = L2G[start - numOwned];
        gSerialize(b, gid, FnTy::extract(getData(gid)));
        FnTy::reset(getData(gid));
      }
      net.send(x, syncRecv, b);
    }
    //Will force all messages to be processed before continuing
    Galois::Runtime::getHostBarrier().wait();
  }

};
