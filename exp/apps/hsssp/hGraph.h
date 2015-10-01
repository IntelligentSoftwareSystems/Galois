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

#include "GlobalObj.h"

template<typename NodeTy, typename EdgeTy, bool BSPNode=false, bool BSPEdge=false>
class hGraph : public GlobalObject {

  typedef typename std::conditional<BSPNode, std::pair<NodeTy, NodeTy>,NodeTy>::type realNodeTy;
  typedef typename std::conditional<BSPNode, std::pair<EdgeTy, EdgeTy>,EdgeTy>::type realEdgeTy;

  typedef Galois::Graph::LC_CSR_Graph<realNodeTy, realEdgeTy> GraphTy;

  GraphTy graph;
  bool round;
  uint32_t numOwned; // [0, numOwned) = global nodes owned, thus [numOwned, numNodes are replicas
  uint32_t globalOffset; // [numOwned, end) + globalOffset = GID
  unsigned id; // my hostid // FIXME: isn't this just Network::ID?
  //ghost cell ID translation
  std::vector<uint32_t> ghostMap; // GID = ghostMap[LID - numOwned]
  //GID to owner
  std::vector<std::pair<uint32_t,uint32_t> > hostNodes; //LID Node owned by host i
  //pointer for each host
  std::vector<uintptr_t> hostPtrs;

  //host -> (lid, lid]
  std::pair<uint32_t, uint32_t> nodes_by_host(uint32_t host) const {
    return hostNodes[host];
  }

  uint32_t L2G(uint32_t lid) const {
    assert(lid < graph.size());
    if (lid < numOwned)
      return lid + globalOffset;
    return ghostMap[lid - numOwned];
  }

  uint32_t G2L(uint32_t gid) const {
    if (gid >= globalOffset && gid < globalOffset + numOwned)
      return gid - globalOffset;
    auto ii = std::lower_bound(ghostMap.begin(), ghostMap.end(), gid);
    assert(*ii == gid);
    return std::distance(ghostMap.begin(), ii);
  }

  uint32_t L2H(uint32_t lid) const {
    assert(lid < graph.size());
    if (lid < numOwned)
      return id;
    for (int i = 0; i < hostNodes.size(); ++i)
      if (hostNodes[i].first >= lid && lid < hostNodes[i].second)
        return i;
    abort();
  }

  bool isOwned(uint32_t gid) const {
    return gid >=globalOffset && gid < globalOffset+numOwned;
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
    auto& r = graph.getEdgeData(ni, mflag);
    return round ? r.first : r.second;
  }

  template<bool en, typename std::enable_if<!en>::type* = nullptr>
  typename GraphTy::edge_data_reference getEdgeDataImpl(typename GraphTy::edge_iterator ni, Galois::MethodFlag mflag = Galois::MethodFlag::ALL) {
    auto& r = graph.getEdgeData(ni, mflag);
    return r;
  }

public:
  static void syncRecv(Galois::Runtime::RecvBuffer& buf) {
    uint32_t oid;
    void (hGraph::*fn)(Galois::Runtime::RecvBuffer&);
    Galois::Runtime::gDeserialize(buf, oid, fn);
    hGraph* obj = reinterpret_cast<hGraph*>(ptrForObj(oid));
    (obj->*fn)(buf);
  }

  template<typename FnTy>
  void syncRecvApply(Galois::Runtime::RecvBuffer& buf) {
    uint32_t num;
    Galois::Runtime::gDeserialize(buf, num);
    for(; num ; --num) {
      uint32_t gid;
      typename FnTy::ValTy val;
      Galois::Runtime::gDeserialize(buf, gid, val);
      assert(isOwned(gid));
      FnTy::reduce(getData(gid - globalOffset), val);
    }
  }
  
 public:
  typedef typename GraphTy::GraphNode GraphNode;
  typedef typename GraphTy::iterator iterator;
  typedef typename GraphTy::const_iterator const_iterator;
  typedef typename GraphTy::local_iterator local_iterator;
  typedef typename GraphTy::const_local_iterator const_local_iterator;
  typedef typename GraphTy::edge_iterator edge_iterator;


  //hGraph construction is collective
  hGraph(const std::string& filename, unsigned host, unsigned numHosts)
    :GlobalObject(this), id(host)
  {
    OfflineGraph g(filename);
    std::cerr << "Offline Graph Done\n";

    //compute owners for all nodes
    std::vector<std::pair<uint32_t, uint32_t>> gid2host;
    for (unsigned i = 0; i < numHosts; ++i)
      gid2host.push_back(Galois::block_range(0U, (unsigned)g.size(), i, numHosts));
    numOwned = gid2host[id].second - gid2host[id].first;
    globalOffset = gid2host[id].first;
    std::cerr <<  "Global info done\n";

    uint64_t numEdges = g.edge_begin(gid2host[id].second) - g.edge_begin(gid2host[id].first); // depends on Offline graph impl
    std::cerr << "Edge count Done\n";

    std::vector<bool> ghosts(g.size());
    for (auto n = gid2host[id].first; n < gid2host[id].second; ++n)
      for (auto ii = g.edge_begin(n), ee = g.edge_end(n); ii < ee; ++ii)
        ghosts[g.getEdgeDst(ii)] = true;
    std::cerr << "Ghost Finding Done\n";

    for (uint64_t x = 0; x < g.size(); ++x)
      if (ghosts[x] && !isOwned(x))
        ghostMap.push_back(x);
    std::cerr << "L2G Done\n";

    hostNodes.resize(numHosts, std::make_pair(~0,~0));
    for (unsigned ln = 0; ln < ghostMap.size(); ++ln) {
      unsigned lid = ln + numOwned;
      auto gid = ghostMap[ln];
      for (auto h = 0; h < gid2host.size(); ++h) {
        auto& p = gid2host[h];
        if (gid >= p.first && gid < p.second) {
          hostNodes[h].first = std::min(hostNodes[h].first, lid);
          hostNodes[h].second = lid+1;
          break;
        }
        abort();
      }
    }
    std::cerr << "hostNodes Done\n";

    uint32_t numNodes = numOwned + ghostMap.size();
    graph.allocateFrom(numNodes, numEdges);
    std::cerr << "Allocate done\n";
    
    graph.constructNodes();
    std::cerr << "Construct nodes done\n";

    uint64_t cur = 0;
    for (auto n = gid2host[id].first; n < gid2host[id].second; ++n) {
      for (auto ii = g.edge_begin(n), ee = g.edge_end(n); ii < ee; ++ii) {
        auto gdst = g.getEdgeDst(ii);
        decltype(gdst) ldst = G2L(gdst);
        graph.constructEdge(cur++, ldst);
      }
      graph.fixEndEdge(n, cur);
    }
    std::cerr << "Construct edges done\n";
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

  edge_iterator edge_begin(GraphNode N) {
    return graph.edge_begin(N);
  }

  edge_iterator edge_end(GraphNode N) {
    return graph.edge_end(N);
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
    void (hGraph::*fn)(Galois::Runtime::RecvBuffer&) = &hGraph::syncRecvApply<FnTy>;
    auto& net = Galois::Runtime::getSystemNetworkInterface();
    for (unsigned x = 0; x < hostNodes.size(); ++x) {
      if (x == id) continue;
      uint32_t start, end;
      std::tie(start, end) = nodes_by_host(x);
      if (start == end) continue;
      Galois::Runtime::SendBuffer b;
      gSerialize(b, idForSelf(), fn, (uint32_t)(end-start));
      for (; start != end; ++start) {
        auto gid = L2G(start);
        gSerialize(b, gid, FnTy::extract(getData(start)));
        FnTy::reset(getData(start));
      }
      net.send(x, syncRecv, b);
    }
    //Will force all messages to be processed before continuing
    Galois::Runtime::getHostBarrier().wait();
  }

};
