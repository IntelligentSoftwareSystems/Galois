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
 * Derived from hGraph.
 * @author Rashid Kaleem <rashid.kaleem@gmail.com>
 *
 */

#include <vector>
#include <set>

#include "Galois/gstl.h"
#include "Galois/Graphs/LC_CSR_Graph.h"
#include "Galois/Runtime/Substrate.h"
#include "Galois/Runtime/Network.h"

#include "Galois/Runtime/Serialize.h"


#include "Galois/Dist/GlobalObj.h"
#include "Galois/Dist/OfflineGraph.h"

#ifdef __GALOIS_HET_CUDA__
#include "Galois/Cuda/cuda_mtypes.h"
#endif

#ifndef _GALOIS_DIST_vGraph_H
#define _GALOIS_DIST_vGraph_H

template<typename NodeTy, typename EdgeTy, bool BSPNode=false, bool BSPEdge=false>
class vGraph : public GlobalObject {

  typedef typename std::conditional<BSPNode, std::pair<NodeTy, NodeTy>,NodeTy>::type realNodeTy;
  typedef typename std::conditional<BSPEdge, std::pair<EdgeTy, EdgeTy>,EdgeTy>::type realEdgeTy;

  typedef Galois::Graph::LC_CSR_Graph<realNodeTy, realEdgeTy> GraphTy;

  GraphTy graph;
  bool round;
  uint64_t totalNodes; // Total nodes in the complete graph.
  uint32_t numOwned; // [0, numOwned) = global nodes owned, thus [numOwned, numNodes are replicas
  uint64_t globalOffset; // [numOwned, end) + globalOffset = GID
  unsigned id; // my hostid // FIXME: isn't this just Network::ID?
  //ghost cell ID translation
  std::vector<uint64_t> ghostMap; // GID = ghostMap[LID - numOwned]
  std::vector<std::pair<uint32_t,uint32_t> > hostNodes; //LID Node owned by host i
  //pointer for each host
  std::vector<uintptr_t> hostPtrs;

  //GID to owner
  std::vector<std::pair<uint64_t, uint64_t>> gid2host;

  uint32_t num_recv_expected; // Number of receives expected for local completion.

  //host -> (lid, lid]
  std::pair<uint32_t, uint32_t> nodes_by_host(uint32_t host) const {
    return hostNodes[host];
  }

  std::pair<uint64_t, uint64_t> nodes_by_host_G(uint32_t host) const {
    return gid2host[host];
  }

  uint64_t L2G(uint32_t lid) const {
    assert(lid < graph.size());
    if (lid < numOwned)
      return lid + globalOffset;
    return ghostMap[lid - numOwned];
  }

  uint32_t G2L(uint64_t gid) const {
    if (gid >= globalOffset && gid < globalOffset + numOwned)
      return gid - globalOffset;
    auto ii = std::lower_bound(ghostMap.begin(), ghostMap.end(), gid);
    assert(*ii == gid);
    return std::distance(ghostMap.begin(), ii) + numOwned;
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

  bool isOwned(uint64_t gid) const {
    return gid >=globalOffset && gid < globalOffset+numOwned;
  }

  template<bool en, typename std::enable_if<en>::type* = nullptr>
  NodeTy& getDataImpl(typename GraphTy::GraphNode N, Galois::MethodFlag mflag = Galois::MethodFlag::WRITE) {
    auto& r = graph.getData(N, mflag);
    return round ? r.first : r.second;
  }

  template<bool en, typename std::enable_if<!en>::type* = nullptr>
  NodeTy& getDataImpl(typename GraphTy::GraphNode N, Galois::MethodFlag mflag = Galois::MethodFlag::WRITE) {
    auto& r = graph.getData(N, mflag);
    return r;
  }

  template<bool en, typename std::enable_if<en>::type* = nullptr>
    const NodeTy& getDataImpl(typename GraphTy::GraphNode N, Galois::MethodFlag mflag = Galois::MethodFlag::WRITE) const{
      auto& r = graph.getData(N, mflag);
      return round ? r.first : r.second;
    }

    template<bool en, typename std::enable_if<!en>::type* = nullptr>
    const NodeTy& getDataImpl(typename GraphTy::GraphNode N, Galois::MethodFlag mflag = Galois::MethodFlag::WRITE) const{
      auto& r = graph.getData(N, mflag);
      return r;
    }
  template<bool en, typename std::enable_if<en>::type* = nullptr>
  typename GraphTy::edge_data_reference getEdgeDataImpl(typename GraphTy::edge_iterator ni, Galois::MethodFlag mflag = Galois::MethodFlag::WRITE) {
    auto& r = graph.getEdgeData(ni, mflag);
    return round ? r.first : r.second;
  }

  template<bool en, typename std::enable_if<!en>::type* = nullptr>
  typename GraphTy::edge_data_reference getEdgeDataImpl(typename GraphTy::edge_iterator ni, Galois::MethodFlag mflag = Galois::MethodFlag::WRITE) {
    auto& r = graph.getEdgeData(ni, mflag);
    return r;
  }

public:
  GraphTy & getGraph(){
     return graph;
  }
  static void syncRecv(Galois::Runtime::RecvBuffer& buf) {
    uint32_t oid;
    void (vGraph::*fn)(Galois::Runtime::RecvBuffer&);
    Galois::Runtime::gDeserialize(buf, oid, fn);
    vGraph* obj = reinterpret_cast<vGraph*>(ptrForObj(oid));
    (obj->*fn)(buf);
    //--(obj->num_recv_expected);
    //std::cout << "[ " << Galois::Runtime::getSystemNetworkInterface().ID << "] " << " NUM RECV EXPECTED : " << (obj->num_recv_expected) << "\n";
  }

  template<typename FnTy>
  void syncRecvApply(Galois::Runtime::RecvBuffer& buf) {
    uint32_t num;
    Galois::Runtime::gDeserialize(buf, num);
    for(; num ; --num) {
      uint64_t gid;
      typename FnTy::ValTy val;
      Galois::Runtime::gDeserialize(buf, gid, val);
      /*
      if(!isOwned(gid)){
        std::cout <<"[" << Galois::Runtime::getSystemNetworkInterface().ID <<"]" <<  " GID " << gid  << " num  : " << num << "\n";
        assert(isOwned(gid));
      }
      */
      assert(isOwned(gid));
      FnTy::reduce(getData(gid - globalOffset), val);
    }
  }

  template<typename FnTy>
  void syncPullRecvReply(Galois::Runtime::RecvBuffer& buf) {
    void (vGraph::*fn)(Galois::Runtime::RecvBuffer&) = &vGraph::syncPullRecvApply<FnTy>;
    auto& net = Galois::Runtime::getSystemNetworkInterface();
    uint32_t num;
    unsigned from_id;
    Galois::Runtime::gDeserialize(buf, from_id, num);
    Galois::Runtime::SendBuffer b;
    gSerialize(b, idForSelf(), fn, num);
    for(; num; --num){
      uint64_t gid;
      typename FnTy::ValTy old_val, val;
      Galois::Runtime::gDeserialize(buf, gid, old_val);
      assert(isOwned(gid));
      val = FnTy::extract(getData((gid - globalOffset)));
      if (net.ID == 0) {
        //std::cout << "PullApply step1 : [" << net.ID << "] "<< " to : " << from_id << " : [" << gid - globalOffset << "] : " << val << "\n";
      }
      //For now just send all.
      //if(val != old_val){
      Galois::Runtime::gSerialize(b, gid, val);
      //}
    }
    net.send(from_id, syncRecv, b);
  }

  template<typename FnTy>
  void syncPullRecvApply(Galois::Runtime::RecvBuffer& buf) {
    assert(num_recv_expected > 0);
    uint32_t num;
    Galois::Runtime::gDeserialize(buf, num);
    auto& net = Galois::Runtime::getSystemNetworkInterface();
    //std::cout << "In Apply : [" << net.ID <<"] num_recv_expected : "<< num_recv_expected << "\n";

    for(; num; --num) {
      uint64_t gid;
      typename FnTy::ValTy val;
      Galois::Runtime::gDeserialize(buf, gid, val);
      //assert(isGhost(gid));
      auto LocalId = G2L(gid);
      if (net.ID == 1) {
        //std::cout << "PullApply Step2 : [" << net.ID << "]  : [" << LocalId << "] : " << val << "\n";
      }
      FnTy::setVal(getData(LocalId), val);
    }
    --num_recv_expected;
  }

 public:
  typedef typename GraphTy::GraphNode GraphNode;
  typedef typename GraphTy::iterator iterator;
  typedef typename GraphTy::const_iterator const_iterator;
  typedef typename GraphTy::local_iterator local_iterator;
  typedef typename GraphTy::const_local_iterator const_local_iterator;
  typedef typename GraphTy::edge_iterator edge_iterator;


  //vGraph construction is collective
  vGraph(const std::string& filename, unsigned host, unsigned numHosts)
    :GlobalObject(this), id(host),round(false),num_recv_expected(0)
  {
    OfflineGraph g(filename);
    totalNodes = g.size();
    //compute owners for all nodes
    for (unsigned i = 0; i < numHosts; ++i){
      gid2host.push_back(Galois::block_range(0U, (unsigned)g.size(), i, numHosts));
    }
    numOwned = gid2host[id].second - gid2host[id].first;
    globalOffset = gid2host[id].first;
    //std::cerr <<  "Global info done\n";

    uint64_t numEdges = g.edge_begin(gid2host[id].second) - g.edge_begin(gid2host[id].first); // depends on Offline graph impl
    std::cerr << "Edge count Done " << numEdges << "\n";
    if(false){
       typedef typename GraphTy::iterator node_iterator;
       const int THRESHOLD=7;
       std::vector<node_iterator> toSplit;
       for(auto n = g.begin(); n!=g.end(); ++n){
         size_t num_nbrs = std::distance(g.edge_begin(*n), g.edge_end(*n));
         if(num_nbrs>THRESHOLD){
            toSplit.push_back(n);
         }
       }

       for(auto i : toSplit){
          fprintf(stderr, "ToSplit :: %d\n", *i);
       }
    }
    //PowerGraph code.
    {
       const int numHosts = 4;//Galois::Runtime::NetworkInterface::Num;
       const int numNodes = g.size();
       const int numEdges = g.sizeEdges();
       std::vector<int> allHosts;
       std::vector<int> hostCounters(numHosts);
       std::vector<int> edgeCounters(numHosts);
       std::vector<bool> nodeFound(g.size());
       std::map<int, std::vector<int>> replicas;
       std::vector<std::set<int>> nodeOwners(g.size());
       std::map<edge_iterator,int> edgeOwners;//(g.sizeEdges());

       for(int i=0;i<numHosts; ++i){
          allHosts.push_back(i);
          hostCounters[i]=0;
          edgeCounters[i]=0;
       }
       for(auto n = g.begin(); n!=g.end(); ++n){
          auto srcNode = *n;
          for(auto e = g.edge_begin(*n); e!=g.edge_end(*n); ++e){
             auto dstNode = g.getEdgeDst(e);
             if(!nodeFound[srcNode] && !nodeFound[dstNode]){
                //Neither src/dst has not been assigned yet, pick the lightest edge-wt partition.
                auto idx = std::min_element(allHosts.begin(), allHosts.end(), [&](int l, int r){return edgeCounters[l]<edgeCounters[r];});
                edgeOwners[e]=*idx;
                nodeOwners[srcNode].insert(*idx);
                nodeOwners[dstNode].insert(*idx);
                edgeCounters[*idx]++;
                hostCounters[*idx]+=2;//2 nodes added to host;

             }else if (nodeFound[srcNode] && nodeFound[dstNode]){
                std::vector<int> common(numHosts);
                auto it = std::set_intersection(nodeOwners[srcNode].begin(),
                                                nodeOwners[srcNode].end(),
                                                nodeOwners[dstNode].begin(),
                                                nodeOwners[dstNode].end(),
                                                common.begin());
                common.resize(it-common.begin());
                if(common.size()> 0){
                   //Common host found
                   auto idx = std::min_element(common.begin(), common.end(), [&](int l, int r){return edgeCounters[l]<edgeCounters[r];});
                   nodeFound[srcNode]=nodeFound[dstNode]=true;
                   nodeOwners[dstNode].insert(*it);
                   nodeOwners[srcNode].insert(*it);
                   edgeOwners[e]=*it;
                   edgeCounters[*it]++;
                }else{
                   //Common host not found.
                   std::vector<int> allHost(2*numHosts);
                   auto it = std::set_union(nodeOwners[srcNode].begin(),
                                                nodeOwners[srcNode].end(),
                                                nodeOwners[dstNode].begin(),
                                                nodeOwners[dstNode].end(),
                                                allHost.begin());
                   allHost.resize(it-allHost.begin());
                   auto idx = std::min_element(allHost.begin(), allHost.end(), [&](int l, int r){return edgeCounters[l]<edgeCounters[r];});
//                   nodeFound[dstNode]=*it;
                   nodeOwners[dstNode].insert(*it);
                   nodeOwners[srcNode].insert(*it);
                   edgeOwners[e]=*it;
                   edgeCounters[*it]++;
                }
             }//else if either one is found, but the other isn't
             else if(nodeFound[srcNode]){
                auto it = std::min_element(nodeOwners[srcNode].begin(), nodeOwners[srcNode].end(), [&](int l, int r){return edgeCounters[l] < edgeCounters[r];});
                nodeFound[dstNode]=true;
                nodeOwners[dstNode].insert(*it);
                edgeOwners[e]=*it;
                edgeCounters[*it]++;
             }else if (nodeFound[dstNode]){
                auto it = std::min_element(nodeOwners[dstNode].begin(), nodeOwners[dstNode].end(), [&](int l, int r){return edgeCounters[l] < edgeCounters[r];});
                nodeFound[srcNode]=true;
                nodeOwners[srcNode].insert(*it);
                edgeOwners[e]=*it;
                edgeCounters[*it]++;
             }else{
                //SHOULD NOT BE HERE!
             }
          }
       }
       {
          int replicatedVertices = 0 ;
          int totalEdges = 0;
          for(int i=0; i<numHosts; ++i){
             fprintf(stderr, "Host - %d, #nodes = %d, #edges = %d \n", i, hostCounters[i], edgeCounters[i]);
             totalEdges += edgeCounters[i];
             replicatedVertices+=hostCounters[i];
          }
          int replicasPerNode = 0;
          for(auto i = g.begin(); i!=g.end(); ++i){
             replicasPerNode+= nodeOwners[*i].size();

          }
          fprintf(stderr, "AverageReplicasPerNode = %6.6g\n", replicasPerNode/(float)(numNodes));
          fprintf(stderr, "TotalEdges = %d, ActualEdges = %d\n", totalEdges, numEdges);

       }
    }
    /*Mark all nodes to which current host has an edge*/
    std::vector<bool> ghosts(g.size());
    for (auto n = gid2host[id].first; n < gid2host[id].second; ++n){
      for (auto ii = g.edge_begin(n), ee = g.edge_end(n); ii < ee; ++ii){
        ghosts[g.getEdgeDst(ii)] = true;
      }
    }
    std::cerr << "Ghost Finding Done " << std::count(ghosts.begin(), ghosts.end(), true) << "\n";
    /*For each node to which current host has an edge, but the destination is not owned
      by current host, add it to ghostMap*/
    for (uint64_t x = 0; x < g.size(); ++x){
      if (ghosts[x] && !isOwned(x)){
        ghostMap.push_back(x);
      }
    }

    hostNodes.resize(numHosts, std::make_pair(~0,~0));
    for (unsigned ln = 0; ln < ghostMap.size(); ++ln) {
      unsigned lid = ln + numOwned;
      auto gid = ghostMap[ln];
      bool found = false;
      for (auto h = 0; h < gid2host.size(); ++h) {
        auto& p = gid2host[h];
        if (gid >= p.first && gid < p.second) {
          hostNodes[h].first = std::min(hostNodes[h].first, lid);
          hostNodes[h].second = lid+1;
          found = true;
          break;
        }
      }
      assert(found);
    }
    //std::cerr << "hostNodes Done\n";

    uint32_t numNodes = numOwned + ghostMap.size();
    assert((uint64_t)numOwned + (uint64_t)ghostMap.size() == (uint64_t)numNodes);
    graph.allocateFrom(numNodes, numEdges);
    //std::cerr << "Allocate done\n";

    graph.constructNodes();
    //std::cerr << "Construct nodes done\n";
    loadEdges<std::is_void <EdgeTy>::value>(g);
  }

  template<bool isVoidType, typename std::enable_if<!isVoidType>::type* = nullptr>
  void loadEdges(OfflineGraph & g){
     fprintf(stderr, "Loading edge-data while creating edges.\n");
     uint64_t cur = 0;
     for (auto n = gid2host[id].first; n < gid2host[id].second; ++n) {
           for (auto ii = g.edge_begin(n), ee = g.edge_end(n); ii < ee; ++ii) {
             auto gdst = g.getEdgeDst(ii);
             decltype(gdst) ldst = G2L(gdst);
             auto gdata = g.getEdgeData<EdgeTy>(ii);
             graph.constructEdge(cur++, ldst, gdata);
           }
           graph.fixEndEdge(G2L(n), cur);
         }
  }
  template<bool isVoidType, typename std::enable_if<isVoidType>::type* = nullptr>
  void loadEdges(OfflineGraph & g){
     fprintf(stderr, "Loading void edge-data while creating edges.\n");
     uint64_t cur = 0;
     for (auto n = gid2host[id].first; n < gid2host[id].second; ++n) {
           for (auto ii = g.edge_begin(n), ee = g.edge_end(n); ii < ee; ++ii) {
             auto gdst = g.getEdgeDst(ii);
             decltype(gdst) ldst = G2L(gdst);
             graph.constructEdge(cur++, ldst);
           }
           graph.fixEndEdge(G2L(n), cur);
         }

  }

  NodeTy& getData(GraphNode N, Galois::MethodFlag mflag = Galois::MethodFlag::WRITE) {
    auto& r = getDataImpl<BSPNode>(N, mflag);
//    auto i =Galois::Runtime::NetworkInterface::ID;
    //std::cerr << i << " " << N << " " <<&r << " " << r.dist_current << "\n";
    return r;
  }

  const NodeTy& getData(GraphNode N, Galois::MethodFlag mflag = Galois::MethodFlag::WRITE) const{
    auto& r = getDataImpl<BSPNode>(N, mflag);
//    auto i =Galois::Runtime::NetworkInterface::ID;
    //std::cerr << i << " " << N << " " <<&r << " " << r.dist_current << "\n";
    return r;
  }
  typename GraphTy::edge_data_reference getEdgeData(edge_iterator ni, Galois::MethodFlag mflag = Galois::MethodFlag::WRITE) {
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

  const_iterator ghost_begin() const { return end(); }
  iterator ghost_begin() { return end(); }
  const_iterator ghost_end() const { return graph.end(); }
  iterator ghost_end() { return graph.end(); }

  template<typename FnTy>
  void sync_push() {
    void (vGraph::*fn)(Galois::Runtime::RecvBuffer&) = &vGraph::syncRecvApply<FnTy>;
    Galois::Timer time1, time2;
    time1.start();
    auto& net = Galois::Runtime::getSystemNetworkInterface();
    for (unsigned x = 0; x < hostNodes.size(); ++x) {
      if (x == id) continue;
      uint32_t start, end;
      std::tie(start, end) = nodes_by_host(x);
      //std::cout << net.ID << " " << x << " " << start << " " << end << "\n";
      if (start == end) continue;
      Galois::Runtime::SendBuffer b;
      gSerialize(b, idForSelf(), fn, (uint32_t)(end-start));
      for (; start != end; ++start) {
        auto gid = L2G(start);
        if( gid > totalNodes){
          //std::cout << "[" << net.ID << "] GID : " << gid << " size : " << graph.size() << "\n";

          assert(gid < totalNodes);
        }
        //std::cout << net.ID << " send (" << gid << ") " << start << " " << FnTy::extract(start, getData(start)) << "\n";
        gSerialize(b, gid, FnTy::extract(getData(start)));
        FnTy::reset(getData(start));
      }
      net.send(x, syncRecv, b);
    }
    time1.stop();
    //Will force all messages to be processed before continuing
    time2.start();
    net.flush();
    Galois::Runtime::getHostBarrier().wait();
    time2.stop();

    //std::cout << "[" << net.ID <<"] time1 : " << time1.get() << "(msec) time2 : " << time2.get() << "(msec)\n";
  }

  template<typename FnTy>
  void sync_pull(){
    void (vGraph::*fn)(Galois::Runtime::RecvBuffer&) = &vGraph::syncPullRecvReply<FnTy>;
    auto& net = Galois::Runtime::getSystemNetworkInterface();
    //Galois::Runtime::getHostBarrier().wait();
    num_recv_expected = 0;
    for(unsigned x = 0; x < hostNodes.size(); ++x){
      if(x == id) continue;
      uint32_t start, end;
      std::tie(start, end) = nodes_by_host(x);
      //std::cout << "end - start" << (end - start) << "\n";
      if(start == end) continue;
      Galois::Runtime::SendBuffer b;
      gSerialize(b, idForSelf(), fn, net.ID, (uint32_t)(end-start));
      for (; start != end; ++start) {
        auto gid = L2G(start);
        //std::cout << net.ID << " PULL send (" << gid << ") " << start << " " << FnTy::extract(start, getData(start)) << "\n";
        gSerialize(b, gid, FnTy::extract(getData(start)));
      }
      net.send(x, syncRecv, b);
      ++num_recv_expected;
    }

    //std::cout << "[" << net.ID <<"] num_recv_expected : "<< num_recv_expected << "\n";

    net.flush();
    while(num_recv_expected) {
      net.handleReceives();
    }

    assert(num_recv_expected == 0);
    // Can remove this barrier???.
    Galois::Runtime::getHostBarrier().wait();
  }

  uint64_t getGID(uint32_t nodeID) const {
    return L2G(nodeID);
  }
  uint32_t getLID(uint64_t nodeID) const {
    return G2L(nodeID);
  }
  unsigned getHostID(uint64_t gid){
    for(auto i = 0; i < hostNodes.size(); ++i){
      uint64_t start, end;
      std::tie(start, end) = nodes_by_host_G(i);
      if(gid >= start && gid  < end){
        return i;
      }
    }
    return -1;
  }
  uint32_t getNumOwned()const{
     return numOwned;
  }
  uint64_t getGlobalOffset()const{
     return globalOffset;
  }
#ifdef __GALOIS_HET_CUDA__
  MarshalGraph getMarshalGraph(unsigned host_id) {
     MarshalGraph m;

     m.nnodes = size();
     m.nedges = sizeEdges();
     m.nowned = std::distance(begin(), end());
     assert(m.nowned > 0);
     m.g_offset = getGID(0);
     m.id = host_id;
     m.row_start = (index_type *) calloc(m.nnodes + 1, sizeof(index_type));
     m.edge_dst = (index_type *) calloc(m.nedges, sizeof(index_type));

     // TODO: initialize node_data and edge_data
     m.node_data = NULL;
     m.edge_data = NULL;

     // pinched from Rashid's LC_LinearArray_Graph.h

     size_t edge_counter = 0, node_counter = 0;
     for (auto n = begin(); n != ghost_end() && *n != m.nnodes; n++, node_counter++) {
        m.row_start[node_counter] = edge_counter;
        if (*n < m.nowned) {
           for (auto e = edge_begin(*n); e != edge_end(*n); e++) {
              if (getEdgeDst(e) < m.nnodes)
                 m.edge_dst[edge_counter++] = getEdgeDst(e);
           }
        }
     }

     m.row_start[node_counter] = edge_counter;
     m.nedges = edge_counter;
     return m;
  }
#endif

};
#endif//_GALOIS_DIST_vGraph_H
