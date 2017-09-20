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
 * @author Gurbinder Gill <gurbinder533@gmail.com>
 */

#include "galois/gstl.h"
#include "galois/graphs/LC_CSR_Graph.h"
#include "galois/runtime/Substrate.h"
#include "galois/runtime/Network.h"

//#include "galois/runtime/Barrier.h"
#include "galois/runtime/Serialize.h"
#include "galois/Timer.h"

#include "galois/runtime/GlobalObj.h"
#include "galois/runtime/OfflineGraph.h"

#ifdef __GALOIS_HET_CUDA__
#include "galois/runtime/Cuda/cuda_mtypes.h"
#endif

#ifdef __GALOIS_HET_OPENCL__
#include "galois/OpenCL/CL_Header.h"
#endif

#ifdef __GALOIS_SIMULATE_COMMUNICATION__
#ifdef __GALOIS_SIMULATE_BARE_MPI_COMMUNICATION__
#include "mpi.h"
#endif
#endif

#ifndef _GALOIS_DIST_HGRAPH_H
#define _GALOIS_DIST_HGRAPH_H

template<typename NodeTy, typename EdgeTy, bool BSPNode = false, bool BSPEdge = false>
class hGraph: public GlobalObject {

   typedef typename std::conditional<BSPNode, std::pair<NodeTy, NodeTy>, NodeTy>::type realNodeTy;
   typedef typename std::conditional<BSPEdge, std::pair<EdgeTy, EdgeTy>, EdgeTy>::type realEdgeTy;

   typedef galois::graphs::LC_CSR_Graph<realNodeTy, realEdgeTy> GraphTy;

   GraphTy graph;bool round;
   uint64_t totalNodes; // Total nodes in the complete graph.
   uint32_t numOwned; // [0, numOwned) = global nodes owned, thus [numOwned, numNodes are replicas
   uint64_t globalOffset; // [numOwned, end) + globalOffset = GID
   const unsigned id; // my hostid // FIXME: isn't this just Network::ID?
   //ghost cell ID translation
   std::vector<uint64_t> ghostMap; // GID = ghostMap[LID - numOwned]
   std::vector<std::pair<uint32_t, uint32_t> > hostNodes; //LID Node owned by host i. Stores ghost nodes from each host.
   //pointer for each host
   std::vector<uintptr_t> hostPtrs;

  //memoization optimization
  std::vector<std::vector<size_t>> mirrorNodes; // mirror nodes from different hosts. For reduce
  std::vector<std::vector<size_t>> masterNodes; // master nodes on different hosts. For broadcast
#ifdef __GALOIS_SIMULATE_COMMUNICATION__
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
  unsigned comm_mode; // Communication mode: 0 - original, 1 - simulated net, 2 - simulated bare MPI
#endif
#endif

   //GID to owner
   std::vector<std::pair<uint64_t, uint64_t>> gid2host;

   uint32_t num_recv_expected; // Number of receives expected for local completion.
   uint32_t num_iter_push; //Keep track of number of iterations.
   uint32_t num_iter_pull; //Keep track of number of iterations.
   uint32_t num_run; //Keep track of number of iterations.

  //Stats: for rough estimate of sendBytes.
   galois::Statistic statGhostNodes;

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
      return gid >= globalOffset && gid < globalOffset + numOwned;
   }

   template<bool en, typename std::enable_if<en>::type* = nullptr>
   NodeTy& getDataImpl(typename GraphTy::GraphNode N, galois::MethodFlag mflag = galois::MethodFlag::WRITE) {
      auto& r = graph.getData(N, mflag);
      return round ? r.first : r.second;
   }

   template<bool en, typename std::enable_if<!en>::type* = nullptr>
   NodeTy& getDataImpl(typename GraphTy::GraphNode N, galois::MethodFlag mflag = galois::MethodFlag::WRITE) {
      auto& r = graph.getData(N, mflag);
      return r;
   }

   template<bool en, typename std::enable_if<en>::type* = nullptr>
   const NodeTy& getDataImpl(typename GraphTy::GraphNode N, galois::MethodFlag mflag = galois::MethodFlag::WRITE) const {
      auto& r = graph.getData(N, mflag);
      return round ? r.first : r.second;
   }

   template<bool en, typename std::enable_if<!en>::type* = nullptr>
   const NodeTy& getDataImpl(typename GraphTy::GraphNode N, galois::MethodFlag mflag = galois::MethodFlag::WRITE) const {
      auto& r = graph.getData(N, mflag);
      return r;
   }
   template<bool en, typename std::enable_if<en>::type* = nullptr>
   typename GraphTy::edge_data_reference getEdgeDataImpl(typename GraphTy::edge_iterator ni, galois::MethodFlag mflag = galois::MethodFlag::WRITE) {
      auto& r = graph.getEdgeData(ni, mflag);
      return round ? r.first : r.second;
   }

   template<bool en, typename std::enable_if<!en>::type* = nullptr>
   typename GraphTy::edge_data_reference getEdgeDataImpl(typename GraphTy::edge_iterator ni, galois::MethodFlag mflag = galois::MethodFlag::WRITE) {
      auto& r = graph.getEdgeData(ni, mflag);
      return r;
   }

public:
   GraphTy & getGraph() {
      return graph;
   }
#ifdef __GALOIS_SIMULATE_COMMUNICATION__
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
  void set_comm_mode(unsigned mode) { // Communication mode: 0 - original, 1 - simulated net, 2 - simulated bare MPI
    comm_mode = mode;
  }
#endif
#endif
  static void syncRecv(uint32_t src, galois::runtime::RecvBuffer& buf) {
      uint32_t oid;
      void (hGraph::*fn)(galois::runtime::RecvBuffer&);
      galois::runtime::gDeserialize(buf, oid, fn);
      hGraph* obj = reinterpret_cast<hGraph*>(ptrForObj(oid));
      (obj->*fn)(buf);
      //--(obj->num_recv_expected);
      //std::cout << "[ " << galois::runtime::getSystemNetworkInterface().ID << "] " << " NUM RECV EXPECTED : " << (obj->num_recv_expected) << "\n";
   }

   void exchange_info_landingPad(galois::runtime::RecvBuffer& buf){
     uint32_t hostID;
     uint64_t numItems;
     std::vector<uint64_t> items;
     galois::runtime::gDeserialize(buf, hostID, numItems);

     galois::runtime::gDeserialize(buf, masterNodes[hostID]);
     std::cout << "from : " << hostID << " -> " << numItems << " --> " << masterNodes[hostID].size() << "\n";
   }



   template<typename FnTy>
     void syncRecvApply(uint32_t from_id, galois::runtime::RecvBuffer& buf, std::string loopName) {
       auto& net = galois::runtime::getSystemNetworkInterface();
       std::string set_timer_str("SYNC_SET_" + loopName + "_" + std::to_string(num_run));
       std::string doall_str("LAMBDA::REDUCE_RECV_APPLY_" + loopName + "_" + std::to_string(num_run));
       galois::StatTimer StatTimer_set(set_timer_str.c_str());
       StatTimer_set.start();

       uint32_t num = masterNodes[from_id].size();
       if(num > 0){
         std::vector<typename FnTy::ValTy> val_vec(num);
         galois::runtime::gDeserialize(buf, val_vec);
         if (!FnTy::reduce_batch(from_id, &val_vec[0])) {
           galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num),
               [&](uint32_t n){
               uint32_t lid = masterNodes[from_id][n];
#ifdef __GALOIS_HET_OPENCL__
           CLNodeDataWrapper d = clGraph.getDataW(lid);
           FnTy::reduce(lid, d, val_vec[n]);
#else
           FnTy::reduce(lid, getData(lid), val_vec[n]);
#endif
               }, galois::loopname(doall_str.c_str()));
         }
       }
       StatTimer_set.stop();
   }

   template<typename FnTy>
   void syncBroadcastRecvReply(galois::runtime::RecvBuffer& buf) {
      void (hGraph::*fn)(galois::runtime::RecvBuffer&) = &hGraph::syncBroadcastRecvApply<FnTy>;
      auto& net = galois::runtime::getSystemNetworkInterface();
      uint32_t num;
      unsigned from_id;
      std::string loopName;
      uint32_t num_iter_pull;
      galois::runtime::gDeserialize(buf, loopName, num_iter_pull, from_id, num);
      std::string extract_timer_str("SYNC_EXTRACT_" + loopName + "_" + std::to_string(num_run) + "_" + std::to_string(num_iter_pull));
      galois::StatTimer StatTimer_extract(extract_timer_str.c_str());
      std::string statSendBytes_str("SEND_BYTES_BROADCAST_REPLY_" + loopName + "_" + std::to_string(num_run) + "_" + std::to_string(num_iter_pull));
      galois::Statistic SyncBroadcastReply_send_bytes(statSendBytes_str);
      std::string doall_str("LAMBDA::BROADCAST_RECV_REPLY_" + loopName + "_" + std::to_string(num_run) + "_" + std::to_string(num_iter_pull));
      galois::runtime::SendBuffer b;
      gSerialize(b, idForSelf(), fn, loopName, num_iter_pull, net.ID, num);

      assert(num == masterNodes[from_id].size());
      StatTimer_extract.start();
      if(num > 0){
        std::vector<typename FnTy::ValTy> val_vec(num);

        if (!FnTy::extract_batch(from_id, &val_vec[0])) {
          galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n){
              uint32_t localID = masterNodes[from_id][n];
#ifdef __GALOIS_HET_OPENCL__
              auto val = FnTy::extract((localID), clGraph.getDataR((localID)));
#else
              auto val = FnTy::extract((localID), getData(localID));
#endif
              assert(n < num);
              val_vec[n] = val;

              }, galois::loopname(doall_str.c_str()));
        }

        galois::runtime::gSerialize(b, val_vec);
      }
      StatTimer_extract.stop();

      SyncBroadcastReply_send_bytes += b.size();
      net.sendMsg(from_id, syncRecv, b);
   }

   template<typename FnTy>
   void syncBroadcastRecvApply(uint32_t from_id, galois::runtime::RecvBuffer& buf, std::string loopName) {
      std::string set_timer_str("SYNC_SET_" + loopName + "_" + std::to_string(num_run));
      galois::StatTimer StatTimer_set(set_timer_str.c_str());
      std::string doall_str("LAMBDA::BROADCAST_RECV_APPLY_" + loopName + "_" + std::to_string(num_run));
      auto& net = galois::runtime::getSystemNetworkInterface();

      uint32_t num = mirrorNodes[from_id].size();

      StatTimer_set.start();

      if(num > 0 ){
        std::vector<typename FnTy::ValTy> val_vec(num);

        galois::runtime::gDeserialize(buf, val_vec);

        if (!FnTy::setVal_batch(from_id, &val_vec[0])) {
          galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n){
              uint32_t localID = mirrorNodes[from_id][n];
#ifdef __GALOIS_HET_OPENCL__
              {
              CLNodeDataWrapper d = clGraph.getDataW(localID);
              FnTy::setVal(localID, d, val_vec[n]);
              }
#else
              FnTy::setVal(localID, getData(localID), val_vec[n]);
#endif
              }, galois::loopname(doall_str.c_str()));
         }
      }
      StatTimer_set.stop();
   }

public:
   typedef typename GraphTy::GraphNode GraphNode;
   typedef typename GraphTy::iterator iterator;
   typedef typename GraphTy::const_iterator const_iterator;
   typedef typename GraphTy::local_iterator local_iterator;
   typedef typename GraphTy::const_local_iterator const_local_iterator;
   typedef typename GraphTy::edge_iterator edge_iterator;

   //hGraph construction is collective
   hGraph(const std::string& filename, unsigned host, unsigned numHosts, std::vector<unsigned> scalefactor = std::vector<unsigned>()) :
         GlobalObject(this), id(host), round(false),statGhostNodes("TotalGhostNodes") {
#ifdef __GALOIS_SIMULATE_COMMUNICATION__
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
      comm_mode = 0;
#endif
#endif
      galois::graphs::OfflineGraph g(filename);
      //std::cerr << "Offline Graph Done\n";

      masterNodes.resize(numHosts);
      mirrorNodes.resize(numHosts);
      num_recv_expected = 0;
      num_iter_push = 0;
      num_iter_pull = 0;
      num_run = 0;
      totalNodes = g.size();
      std::cerr << "Total nodes : " << totalNodes << "\n";
      std::cerr << "Total edges : " << g.sizeEdges() << "\n";
      //compute owners for all nodes
      if (scalefactor.empty() || (numHosts == 1)) {
         for (unsigned i = 0; i < numHosts; ++i)
            gid2host.push_back(galois::block_range(0U, (unsigned) g.size(), i, numHosts));
      } else {
         assert(scalefactor.size() == numHosts);
         unsigned numBlocks = 0;
         for (unsigned i = 0; i < numHosts; ++i)
            numBlocks += scalefactor[i];
         std::vector<std::pair<uint64_t, uint64_t>> blocks;
         for (unsigned i = 0; i < numBlocks; ++i)
            blocks.push_back(galois::block_range(0U, (unsigned) g.size(), i, numBlocks));
         std::vector<unsigned> prefixSums;
         prefixSums.push_back(0);
         for (unsigned i = 1; i < numHosts; ++i)
            prefixSums.push_back(prefixSums[i - 1] + scalefactor[i - 1]);
         for (unsigned i = 0; i < numHosts; ++i) {
            unsigned firstBlock = prefixSums[i];
            unsigned lastBlock = prefixSums[i] + scalefactor[i] - 1;
            gid2host.push_back(std::make_pair(blocks[firstBlock].first, blocks[lastBlock].second));
         }
      }

      numOwned = gid2host[id].second - gid2host[id].first;
      globalOffset = gid2host[id].first;
      std::cerr << "[" << id << "] Owned nodes: " << numOwned << "\n";

      uint64_t numEdges = g.edge_begin(gid2host[id].second) - g.edge_begin(gid2host[id].first); // depends on Offline graph impl
      std::cerr << "[" << id << "] Edge count Done " << numEdges << "\n";

      std::vector<bool> ghosts(g.size());
#if 0
      for (auto n = gid2host[id].first; n < gid2host[id].second; ++n){
         for (auto ii = g.edge_begin(n), ee = g.edge_end(n); ii < ee; ++ii){
            ghosts[g.getEdgeDst(ii)] = true;
         }
      }
#endif
      auto ee = g.edge_begin(gid2host[id].first);
      for (auto n = gid2host[id].first; n < gid2host[id].second; ++n) {
         auto ii = ee;
         ee = g.edge_end(n);
         for (; ii < ee; ++ii) {
            ghosts[g.getEdgeDst(ii)] = true;
         }
      }
      std::cerr << "[" << id << "] Ghost Finding Done " << std::count(ghosts.begin(), ghosts.end(), true) << "\n";

      for (uint64_t x = 0; x < g.size(); ++x)
         if (ghosts[x] && !isOwned(x))
            ghostMap.push_back(x);
      std::cerr << "[" << id << "] Ghost nodes: " << ghostMap.size() << "\n";

      hostNodes.resize(numHosts, std::make_pair(~0, ~0));
      for (unsigned ln = 0; ln < ghostMap.size(); ++ln) {
         unsigned lid = ln + numOwned;
         auto gid = ghostMap[ln];
         bool found = false;
         for (auto h = 0; h < gid2host.size(); ++h) {
            auto& p = gid2host[h];
            if (gid >= p.first && gid < p.second) {
               hostNodes[h].first = std::min(hostNodes[h].first, lid);
               hostNodes[h].second = lid + 1;
               found = true;
               break;
            }
         }
         assert(found);
      }

      for(unsigned h = 0; h < hostNodes.size(); ++h){
        std::string temp_str = ("GhostNodes_from_" + std::to_string(h));
        galois::Statistic temp_stat_ghosNode(temp_str);
        uint32_t start, end;
        std::tie(start, end) = nodes_by_host(h);
        temp_stat_ghosNode += (end - start);
        statGhostNodes += (end - start);
      }
      //std::cerr << "hostNodes Done\n";

      uint32_t numNodes = numOwned + ghostMap.size();
      assert((uint64_t )numOwned + (uint64_t )ghostMap.size() == (uint64_t )numNodes);
      graph.allocateFrom(numNodes, numEdges);
      //std::cerr << "Allocate done\n";

      graph.constructNodes();
      //std::cerr << "Construct nodes done\n";
      loadEdges<std::is_void<EdgeTy>::value>(g);
#ifdef __GALOIS_HET_OPENCL__
      clGraph.load_from_hgraph(*this);
#endif
      
      setup_communication();

#ifdef __GALOIS_SIMULATE_COMMUNICATION__
#ifndef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
      simulate_communication();
#endif
#endif
   }

   void setup_communication() {
      galois::StatTimer StatTimer_comm_setup("COMMUNICATION_SETUP_TIME");
      galois::runtime::getHostBarrier().wait();
      StatTimer_comm_setup.start();

      for(uint32_t h = 0; h < hostNodes.size(); ++h){
        uint32_t start, end;
        std::tie(start, end) = nodes_by_host(h);
        for(; start != end; ++start){
          mirrorNodes[h].push_back(L2G(start));
        }
      }

      //Exchange information for memoization optimization.
      exchange_info_init();

      for(uint32_t h = 0; h < masterNodes.size(); ++h){
         galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(masterNodes[h].size()),
             [&](uint32_t n){
             masterNodes[h][n] = G2L(masterNodes[h][n]);
             }, galois::loopname("MASTER_NODES"));
      }

      for(uint32_t h = 0; h < mirrorNodes.size(); ++h){
         galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(mirrorNodes[h].size()),
             [&](uint32_t n){
             mirrorNodes[h][n] = G2L(mirrorNodes[h][n]);
             }, galois::loopname("MIRROR_NODES"));
      }

      for(auto x = 0; x < masterNodes.size(); ++x){
        std::string master_nodes_str = "MASTER_NODES_TO_" + std::to_string(x);
        galois::Statistic StatMasterNodes(master_nodes_str);
        StatMasterNodes += masterNodes[x].size();
      }

      for(auto x = 0; x < mirrorNodes.size(); ++x){
        std::string mirror_nodes_str = "MIRROR_NODES_FROM_" + std::to_string(x);
        galois::Statistic StatMirrorNodes(mirror_nodes_str);
        StatMirrorNodes += mirrorNodes[x].size();
      }

      StatTimer_comm_setup.stop();
   }

#ifdef __GALOIS_SIMULATE_COMMUNICATION__
#ifndef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
   void simulate_communication() {
     for (int i = 0; i < 10; ++i) {
     simulate_broadcast("");
     simulate_reduce("");

#ifdef __GALOIS_SIMULATE_BARE_MPI_COMMUNICATION__
     simulate_bare_mpi_broadcast("");
     simulate_bare_mpi_reduce("");
#endif
     }
   }
#endif
#endif

   template<bool isVoidType, typename std::enable_if<!isVoidType>::type* = nullptr>
   void loadEdges(galois::graphs::OfflineGraph & g) {
      fprintf(stderr, "Loading edge-data while creating edges.\n");

      uint64_t cur = 0;
      galois::Timer timer;
      std::cout <<"["<<id<<"]PRE :: NumSeeks ";
      g.num_seeks();
      g.reset_seek_counters();
      timer.start();
#if 1
      auto ee = g.edge_begin(gid2host[id].first);
      for (auto n = gid2host[id].first; n < gid2host[id].second; ++n) {
         auto ii = ee;
         ee=g.edge_end(n);
         for (; ii < ee; ++ii) {
            auto gdst = g.getEdgeDst(ii);
            decltype(gdst) ldst = G2L(gdst);
            auto gdata = g.getEdgeData<EdgeTy>(ii);
            graph.constructEdge(cur++, ldst, gdata);
         }
         graph.fixEndEdge(G2L(n), cur);
      }
      //RK - This code should be slightly faster than the conventional single-phase
      // code to load the edges since the file pointer is not moved between the
      // destination and the data on each edge.
      // NEEDS TO BE FASTER!
#if 0
      for (auto n = g.begin(); n != g.end(); ++n) {
         if (this->isOwned(*n)) {
            auto ii = g.edge_begin(*n), ee = g.edge_end(*n);
            for (; ii < ee; ++ii) {
               auto gdst = g.getEdgeDst(ii);
               decltype(gdst) ldst = G2L(gdst);
               graph.constructEdge(cur++, ldst);
            }
            graph.fixEndEdge(G2L(*n), cur);
         }
      }
      //Now load the edge data.
      cur=0;
      if(false)for (auto n = g.begin(); n != g.end(); ++n) {
         if (this->isOwned(*n)) {
            auto ii = g.edge_begin(*n), ee = g.edge_end(*n);
            for (; ii < ee; ++ii) {
               auto gdata = g.getEdgeData<EdgeTy>(ii);
               graph.getEdgeData(cur++)=gdata;
            }
         }
      }
#endif
#else
      //Old code - single loop for edge destination and edge Data.
      for (auto n = gid2host[id].first; n < gid2host[id].second; ++n) {
         for (auto ii = g.edge_begin(n), ee = g.edge_end(n); ii < ee; ++ii) {
            auto gdst = g.getEdgeDst(ii);
            decltype(gdst) ldst = G2L(gdst);
            auto gdata = g.getEdgeData<EdgeTy>(ii);
            graph.constructEdge(cur++, ldst, gdata);
         }
         graph.fixEndEdge(G2L(n), cur);
      }
#endif
      timer.stop();
      std::cout <<"["<<id<<"]POST :: NumSeeks ";
      g.num_seeks();
      std::cout << "EdgeLoading time " << timer.get_usec()/1000000.0f << " seconds\n";
   }
   template<bool isVoidType, typename std::enable_if<isVoidType>::type* = nullptr>
   void loadEdges(galois::graphs::OfflineGraph & g) {
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

   NodeTy& getData(GraphNode N, galois::MethodFlag mflag = galois::MethodFlag::WRITE) {
      auto& r = getDataImpl<BSPNode>(N, mflag);
//    auto i =galois::runtime::NetworkInterface::ID;
      //std::cerr << i << " " << N << " " <<&r << " " << r.dist_current << "\n";
      return r;
   }

   const NodeTy& getData(GraphNode N, galois::MethodFlag mflag = galois::MethodFlag::WRITE) const {
      auto& r = getDataImpl<BSPNode>(N, mflag);
//    auto i =galois::runtime::NetworkInterface::ID;
      //std::cerr << i << " " << N << " " <<&r << " " << r.dist_current << "\n";
      return r;
   }
   typename GraphTy::edge_data_reference getEdgeData(edge_iterator ni, galois::MethodFlag mflag = galois::MethodFlag::WRITE) {
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

   size_t size() const {
      return graph.size();
   }
   size_t sizeEdges() const {
      return graph.sizeEdges();
   }

   const_iterator begin() const {
      return graph.begin();
   }
   iterator begin() {
      return graph.begin();
   }
   const_iterator end() const {
      return graph.begin() + numOwned;
   }
   iterator end() {
      return graph.begin() + numOwned;
   }

   const_iterator ghost_begin() const {
      return end();
   }
   iterator ghost_begin() {
      return end();
   }
   const_iterator ghost_end() const {
      return graph.end();
   }
   iterator ghost_end() {
      return graph.end();
   }

  void exchange_info_init(){
    auto& net = galois::runtime::getSystemNetworkInterface();
    //may be reusing tag, so need a barrier
    galois::runtime::getHostBarrier().wait();

    for (unsigned x = 0; x < net.Num; ++x) {
      if((x == id))
        continue;

      galois::runtime::SendBuffer b;
      gSerialize(b, (uint64_t)mirrorNodes[x].size(), mirrorNodes[x]);
      net.sendTagged(x, 1, b);
      std::cout << " number of mirrors from : " << x << " : " << mirrorNodes[x].size() << "\n";
   }

    //receive
    for (unsigned x = 0; x < net.Num; ++x) {
      if((x == id))
        continue;

      decltype(net.recieveTagged(1, nullptr)) p;
      do {
        net.handleReceives();
        p = net.recieveTagged(1, nullptr);
      } while(!p);

      uint64_t numItems;
      galois::runtime::gDeserialize(p->second, numItems);
      galois::runtime::gDeserialize(p->second, masterNodes[p->first]);
      std::cout << "from : " << p->first << " -> " << numItems << " --> " << masterNodes[p->first].size() << "\n";
    }

    //may be reusing tag, so need a barrier
    galois::runtime::getHostBarrier().wait();
   }

#ifdef __GALOIS_SIMULATE_COMMUNICATION__
#ifdef __GALOIS_SIMULATE_BARE_MPI_COMMUNICATION__
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
   template<typename FnTy>
#endif
   void simulate_bare_mpi_broadcast(std::string loopName, bool mem_copy = false) {
      //std::cerr << "WARNING: requires MPI_THREAD_MULTIPLE to be set in MPI_Init_thread() and Net to not receive MPI messages with tag 32767\n";
      std::string statSendBytes_str("SIMULATE_MPI_SEND_BYTES_BROADCAST_" + loopName + "_" + std::to_string(num_run));
      galois::Statistic SyncBroadcast_send_bytes(statSendBytes_str);
      std::string timer_str("SIMULATE_MPI_BROADCAST_" + loopName + "_" + std::to_string(num_run));
      galois::StatTimer StatTimer_syncBroadcast(timer_str.c_str());
      std::string timer_barrier_str("SIMULATE_MPI_BROADCAST_BARRIER_" + loopName + "_" + std::to_string(num_run));
      galois::StatTimer StatTimerBarrier_syncBroadcast(timer_barrier_str.c_str());
      std::string extract_timer_str("SIMULATE_MPI_BROADCAST_EXTRACT_" + loopName +"_" + std::to_string(num_run));
      galois::StatTimer StatTimer_extract(extract_timer_str.c_str());
      std::string set_timer_str("SIMULATE_MPI_BROADCAST_SET_" + loopName +"_" + std::to_string(num_run));
      galois::StatTimer StatTimer_set(set_timer_str.c_str());

#ifndef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      StatTimer_syncBroadcast.start();
      auto& net = galois::runtime::getSystemNetworkInterface();

      std::vector<MPI_Request> requests(2 * net.Num);
      unsigned num_requests = 0;

#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
      static std::vector< std::vector<typename FnTy::ValTy> > sb;
#else
      static std::vector< std::vector<uint64_t> > sb;
#endif
      sb.resize(net.Num);
      std::vector<uint8_t> bs[net.Num];
      for (unsigned x = 0; x < net.Num; ++x) {
         uint32_t num = masterNodes[x].size();
         if((x == id) || (num == 0))
           continue;

         StatTimer_extract.start();

         sb[x].resize(num);

#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
         size_t size = num * sizeof(typename FnTy::ValTy);
         std::vector<typename FnTy::ValTy> &val_vec = sb[x];
#else
         size_t size = num * sizeof(uint64_t);
         std::vector<uint64_t> &val_vec = sb[x];
#endif

#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
         if (!FnTy::extract_batch(x, &val_vec[0])) {
           galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n){
               uint32_t localID = masterNodes[x][n];
#ifdef __GALOIS_HET_OPENCL__
               auto val = FnTy::extract((localID), clGraph.getDataR((localID)));
#else
               auto val = FnTy::extract((localID), getData(localID));
#endif
               val_vec[n] = val;

               }, galois::loopname("BROADCAST_EXTRACT"));
         }
#else
         val_vec[0] = 1;
#endif
         
         if (mem_copy) {
           bs[x].resize(size);
           memcpy(bs[x].data(), sb[x].data(), size);
         }

         StatTimer_extract.stop();

         SyncBroadcast_send_bytes += size;
         //std::cerr << "[" << id << "]" << " mpi send to " << x << " : " << size << "\n";
         if (mem_copy)
           MPI_Isend((uint8_t *)bs[x].data(), size, MPI_BYTE, x, 32767, MPI_COMM_WORLD, &requests[num_requests++]);
         else
           MPI_Isend((uint8_t *)sb[x].data(), size, MPI_BYTE, x, 32767, MPI_COMM_WORLD, &requests[num_requests++]);
      }

#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
      static std::vector< std::vector<typename FnTy::ValTy> > rb;
#else
      static std::vector< std::vector<uint64_t> > rb;
#endif
      rb.resize(net.Num);
      std::vector<uint8_t> b[net.Num];
      for (unsigned x = 0; x < net.Num; ++x) {
         uint32_t num = mirrorNodes[x].size();
         if((x == id) || (num == 0))
           continue;
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
         size_t size = num * sizeof(typename FnTy::ValTy);
#else
         size_t size = num * sizeof(uint64_t);
#endif
         rb[x].resize(num);
         if (mem_copy) b[x].resize(size);

         //std::cerr << "[" << id << "]" << " mpi receive from " << x << " : " << size << "\n";
         if (mem_copy)
           MPI_Irecv((uint8_t *)b[x].data(), size, MPI_BYTE, x, 32767, MPI_COMM_WORLD, &requests[num_requests++]);
         else
           MPI_Irecv((uint8_t *)rb[x].data(), size, MPI_BYTE, x, 32767, MPI_COMM_WORLD, &requests[num_requests++]);
      }

      StatTimerBarrier_syncBroadcast.start();
      MPI_Waitall(num_requests, &requests[0], MPI_STATUSES_IGNORE);
      StatTimerBarrier_syncBroadcast.stop();

      for (unsigned x = 0; x < net.Num; ++x) {
         uint32_t num = mirrorNodes[x].size();
         if((x == id) || (num == 0))
           continue;

         StatTimer_set.start();

         //std::cerr << "[" << id << "]" << " mpi received from " << x << "\n";
         if (mem_copy) memcpy(rb[x].data(), b[x].data(), b[x].size());
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
         std::vector<typename FnTy::ValTy> &val_vec = rb[x];
#else
         std::vector<uint64_t> &val_vec = rb[x];
#endif
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
         if (!FnTy::setVal_batch(x, &val_vec[0])) {
           galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n){
               uint32_t localID = mirrorNodes[x][n];
#ifdef __GALOIS_HET_OPENCL__
               {
               CLNodeDataWrapper d = clGraph.getDataW(localID);
               FnTy::setVal(localID, d, val_vec[n]);
               }
#else
               FnTy::setVal(localID, getData(localID), val_vec[n]);
#endif
               }, galois::loopname("BROADCAST_SET"));
          }
#endif

         StatTimer_set.stop();
      }

      //std::cerr << "[" << id << "]" << "pull mpi done\n";
      StatTimer_syncBroadcast.stop();
   }

#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
   template<typename FnTy>
#endif
   void simulate_bare_mpi_reduce(std::string loopName, bool mem_copy = false) {
      //std::cerr << "WARNING: requires MPI_THREAD_MULTIPLE to be set in MPI_Init_thread() and Net to not receive MPI messages with tag 32767\n";
      std::string statSendBytes_str("SIMULATE_MPI_SEND_BYTES_REDUCE_" + loopName + "_" + std::to_string(num_run));
      galois::Statistic SyncReduce_send_bytes(statSendBytes_str);
      std::string timer_str("SIMULATE_MPI_REDUCE_" + loopName + "_" + std::to_string(num_run));
      galois::StatTimer StatTimer_syncReduce(timer_str.c_str());
      std::string timer_barrier_str("SIMULATE_MPI_REDUCE_BARRIER_" + loopName + "_" + std::to_string(num_run));
      galois::StatTimer StatTimerBarrier_syncReduce(timer_barrier_str.c_str());
      std::string extract_timer_str("SIMULATE_MPI_REDUCE_EXTRACT_" + loopName +"_" + std::to_string(num_run));
      galois::StatTimer StatTimer_extract(extract_timer_str.c_str());
      std::string set_timer_str("SIMULATE_MPI_REDUCE_SET_" + loopName +"_" + std::to_string(num_run));
      galois::StatTimer StatTimer_set(set_timer_str.c_str());

#ifndef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      StatTimer_syncReduce.start();
      auto& net = galois::runtime::getSystemNetworkInterface();

      std::vector<MPI_Request> requests(2 * net.Num);
      unsigned num_requests = 0;

#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
      static std::vector< std::vector<typename FnTy::ValTy> > sb;
#else
      static std::vector< std::vector<uint64_t> > sb;
#endif
      sb.resize(net.Num);
      std::vector<uint8_t> bs[net.Num];
      for (unsigned x = 0; x < net.Num; ++x) {
         uint32_t num = mirrorNodes[x].size();
         if((x == id) || (num == 0))
           continue;

         StatTimer_extract.start();

         sb[x].resize(num);

#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
         size_t size = num * sizeof(typename FnTy::ValTy);
         std::vector<typename FnTy::ValTy> &val_vec = sb[x];
#else
         size_t size = num * sizeof(uint64_t);
         std::vector<uint64_t> &val_vec = sb[x];
#endif

#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
         if (!FnTy::extract_reset_batch(x, &val_vec[0])) {
           galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n){
                uint32_t lid = mirrorNodes[x][n];
#ifdef __GALOIS_HET_OPENCL__
                CLNodeDataWrapper d = clGraph.getDataW(lid);
                auto val = FnTy::extract(lid, getData(lid, d));
                FnTy::reset(lid, d);
#else
                auto val = FnTy::extract(lid, getData(lid));
                FnTy::reset(lid, getData(lid));
#endif
                val_vec[n] = val;
               }, galois::loopname("REDUCE_EXTRACT"));
         }
#else
         val_vec[0] = 1;
#endif
         
         if (mem_copy) {
           bs[x].resize(size);
           memcpy(bs[x].data(), sb[x].data(), size);
         }

         StatTimer_extract.stop();

         SyncReduce_send_bytes += size;
         //std::cerr << "[" << id << "]" << " mpi send to " << x << " : " << size << "\n";
         if (mem_copy)
           MPI_Isend((uint8_t *)bs[x].data(), size, MPI_BYTE, x, 32767, MPI_COMM_WORLD, &requests[num_requests++]);
         else
           MPI_Isend((uint8_t *)sb[x].data(), size, MPI_BYTE, x, 32767, MPI_COMM_WORLD, &requests[num_requests++]);
      }

#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
      static std::vector< std::vector<typename FnTy::ValTy> > rb;
#else
      static std::vector< std::vector<uint64_t> > rb;
#endif
      rb.resize(net.Num);
      std::vector<uint8_t> b[net.Num];
      for (unsigned x = 0; x < net.Num; ++x) {
         uint32_t num = masterNodes[x].size();
         if((x == id) || (num == 0))
           continue;
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
         size_t size = num * sizeof(typename FnTy::ValTy);
#else
         size_t size = num * sizeof(uint64_t);
#endif
         rb[x].resize(num);
         if (mem_copy) b[x].resize(size);

         //std::cerr << "[" << id << "]" << " mpi receive from " << x << " : " << size << "\n";
         if (mem_copy)
           MPI_Irecv((uint8_t *)b[x].data(), size, MPI_BYTE, x, 32767, MPI_COMM_WORLD, &requests[num_requests++]);
         else
           MPI_Irecv((uint8_t *)rb[x].data(), size, MPI_BYTE, x, 32767, MPI_COMM_WORLD, &requests[num_requests++]);
      }

      StatTimerBarrier_syncReduce.start();
      MPI_Waitall(num_requests, &requests[0], MPI_STATUSES_IGNORE);
      StatTimerBarrier_syncReduce.stop();

      for (unsigned x = 0; x < net.Num; ++x) {
         uint32_t num = masterNodes[x].size();
         if((x == id) || (num == 0))
           continue;

         StatTimer_set.start();

         //std::cerr << "[" << id << "]" << " mpi received from " << x << "\n";
         if (mem_copy) memcpy(rb[x].data(), b[x].data(), b[x].size());
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
         std::vector<typename FnTy::ValTy> &val_vec = rb[x];
#else
         std::vector<uint64_t> &val_vec = rb[x];
#endif
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
         if (!FnTy::reduce_batch(x, &val_vec[0])) {
           galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num),
               [&](uint32_t n){
               uint32_t lid = masterNodes[x][n];
#ifdef __GALOIS_HET_OPENCL__
           CLNodeDataWrapper d = clGraph.getDataW(lid);
           FnTy::reduce(lid, d, val_vec[n]);
#else
           FnTy::reduce(lid, getData(lid), val_vec[n]);
#endif
               }, galois::loopname("REDUCE_SET"));
         }
#endif

         StatTimer_set.stop();
      }
      
      //std::cerr << "[" << id << "]" << "push mpi done\n";
      StatTimer_syncReduce.stop();
   }

#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
   template<typename FnTy>
#endif
   void simulate_bare_mpi_broadcast_serialized() {
      //std::cerr << "WARNING: requires MPI_THREAD_MULTIPLE to be set in MPI_Init_thread() and Net to not receive MPI messages with tag 32767\n";
      galois::StatTimer StatTimer_syncBroadcast("SIMULATE_MPI_BROADCAST");
      galois::Statistic SyncBroadcast_send_bytes("SIMULATE_MPI_BROADCAST_SEND_BYTES");

#ifndef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      StatTimer_syncBroadcast.start();
      auto& net = galois::runtime::getSystemNetworkInterface();

      std::vector<MPI_Request> requests(2 * net.Num);
      unsigned num_requests = 0;

      galois::runtime::SendBuffer sb[net.Num];
      for (unsigned x = 0; x < net.Num; ++x) {
         uint32_t num = masterNodes[x].size();
         if((x == id) || (num == 0))
           continue;

#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
         size_t size = num * sizeof(typename FnTy::ValTy);
         std::vector<typename FnTy::ValTy> val_vec(num);
#else
         size_t size = num * sizeof(uint64_t);
         std::vector<uint64_t> val_vec(num);
#endif
         size+=8;

#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
         if (!FnTy::extract_batch(x, &val_vec[0])) {
           galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n){
               uint32_t localID = masterNodes[x][n];
#ifdef __GALOIS_HET_OPENCL__
               auto val = FnTy::extract((localID), clGraph.getDataR((localID)));
#else
               auto val = FnTy::extract((localID), getData(localID));
#endif
               val_vec[n] = val;

               }, galois::loopname("BROADCAST_EXTRACT"));
         }
#else
         val_vec[0] = 1;
#endif

         galois::runtime::gSerialize(sb[x], val_vec);
         assert(size == sb[x].size());
         
         SyncBroadcast_send_bytes += size;
         //std::cerr << "[" << id << "]" << " mpi send to " << x << " : " << size << "\n";
         MPI_Isend(sb[x].linearData(), size, MPI_BYTE, x, 32767, MPI_COMM_WORLD, &requests[num_requests++]);
      }

      galois::runtime::RecvBuffer rb[net.Num];
      for (unsigned x = 0; x < net.Num; ++x) {
         uint32_t num = mirrorNodes[x].size();
         if((x == id) || (num == 0))
           continue;
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
         size_t size = num * sizeof(typename FnTy::ValTy);
#else
         size_t size = num * sizeof(uint64_t);
#endif
         size+=8;
         rb[x].reset(size);

         //std::cerr << "[" << id << "]" << " mpi receive from " << x << " : " << size << "\n";
         MPI_Irecv((uint8_t *)rb[x].linearData(), size, MPI_BYTE, x, 32767, MPI_COMM_WORLD, &requests[num_requests++]);
      }

      MPI_Waitall(num_requests, &requests[0], MPI_STATUSES_IGNORE);

      for (unsigned x = 0; x < net.Num; ++x) {
         uint32_t num = mirrorNodes[x].size();
         if((x == id) || (num == 0))
           continue;
         //std::cerr << "[" << id << "]" << " mpi received from " << x << "\n";
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
         std::vector<typename FnTy::ValTy> val_vec(num);
#else
         std::vector<uint64_t> val_vec(num);
#endif
         galois::runtime::gDeserialize(rb[x], val_vec);
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
         if (!FnTy::setVal_batch(x, &val_vec[0])) {
           galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n){
               uint32_t localID = mirrorNodes[x][n];
#ifdef __GALOIS_HET_OPENCL__
               {
               CLNodeDataWrapper d = clGraph.getDataW(localID);
               FnTy::setVal(localID, d, val_vec[n]);
               }
#else
               FnTy::setVal(localID, getData(localID), val_vec[n]);
#endif
               }, galois::loopname("BROADCAST_SET"));
          }
#endif
      }

      //std::cerr << "[" << id << "]" << "pull mpi done\n";
      StatTimer_syncBroadcast.stop();
   }

#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
   template<typename FnTy>
#endif
   void simulate_bare_mpi_reduce_serialized() {
      //std::cerr << "WARNING: requires MPI_THREAD_MULTIPLE to be set in MPI_Init_thread() and Net to not receive MPI messages with tag 32767\n";
      galois::StatTimer StatTimer_syncReduce("SIMULATE_MPI_REDUCE");
      galois::Statistic SyncReduce_send_bytes("SIMULATE_MPI_REDUCE_SEND_BYTES");

#ifndef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      StatTimer_syncReduce.start();
      auto& net = galois::runtime::getSystemNetworkInterface();

      std::vector<MPI_Request> requests(2 * net.Num);
      unsigned num_requests = 0;

      galois::runtime::SendBuffer sb[net.Num];
      for (unsigned x = 0; x < net.Num; ++x) {
         uint32_t num = mirrorNodes[x].size();
         if((x == id) || (num == 0))
           continue;

#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
         size_t size = num * sizeof(typename FnTy::ValTy);
         std::vector<typename FnTy::ValTy> val_vec(num);
#else
         size_t size = num * sizeof(uint64_t);
         std::vector<uint64_t> val_vec(num);
#endif
         size+=8;

#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
         if (!FnTy::extract_reset_batch(x, &val_vec[0])) {
           galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n){
                uint32_t lid = mirrorNodes[x][n];
#ifdef __GALOIS_HET_OPENCL__
                CLNodeDataWrapper d = clGraph.getDataW(lid);
                auto val = FnTy::extract(lid, getData(lid, d));
                FnTy::reset(lid, d);
#else
                auto val = FnTy::extract(lid, getData(lid));
                FnTy::reset(lid, getData(lid));
#endif
                val_vec[n] = val;
               }, galois::loopname("REDUCE_EXTRACT"));
         }
#else
         val_vec[0] = 1;
#endif

         galois::runtime::gSerialize(sb[x], val_vec);
         assert(size == sb[x].size());

         SyncReduce_send_bytes += size;
         //std::cerr << "[" << id << "]" << " mpi send to " << x << " : " << size << "\n";
         MPI_Isend(sb[x].linearData(), size, MPI_BYTE, x, 32767, MPI_COMM_WORLD, &requests[num_requests++]);
      }

      galois::runtime::RecvBuffer rb[net.Num];
      for (unsigned x = 0; x < net.Num; ++x) {
         uint32_t num = masterNodes[x].size();
         if((x == id) || (num == 0))
           continue;
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
         size_t size = num * sizeof(typename FnTy::ValTy);
#else
         size_t size = num * sizeof(uint64_t);
#endif
         size+=8;
         rb[x].reset(size);

         //std::cerr << "[" << id << "]" << " mpi receive from " << x << " : " << size << "\n";
         MPI_Irecv((uint8_t *)rb[x].linearData(), size, MPI_BYTE, x, 32767, MPI_COMM_WORLD, &requests[num_requests++]);
      }

      MPI_Waitall(num_requests, &requests[0], MPI_STATUSES_IGNORE);

      for (unsigned x = 0; x < net.Num; ++x) {
         uint32_t num = masterNodes[x].size();
         if((x == id) || (num == 0))
           continue;
         //std::cerr << "[" << id << "]" << " mpi received from " << x << "\n";
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
         std::vector<typename FnTy::ValTy> val_vec(num);
#else
         std::vector<uint64_t> val_vec(num);
#endif
         galois::runtime::gDeserialize(rb[x], val_vec);
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
         if (!FnTy::reduce_batch(x, &val_vec[0])) {
           galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num),
               [&](uint32_t n){
               uint32_t lid = masterNodes[x][n];
#ifdef __GALOIS_HET_OPENCL__
           CLNodeDataWrapper d = clGraph.getDataW(lid);
           FnTy::reduce(lid, d, val_vec[n]);
#else
           FnTy::reduce(lid, getData(lid), val_vec[n]);
#endif
               }, galois::loopname("REDUCE_SET"));
         }
#endif
      }
      
      //std::cerr << "[" << id << "]" << "push mpi done\n";
      StatTimer_syncReduce.stop();
   }
#endif
#endif

#ifdef __GALOIS_SIMULATE_COMMUNICATION__
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
   template<typename FnTy>
#endif
   void syncRecvApplyPull(galois::runtime::RecvBuffer& buf) {
     unsigned from_id;
     uint32_t num;
     std::string loopName;
     uint32_t num_iter_push;
     galois::runtime::gDeserialize(buf, from_id, num);
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
     std::vector<typename FnTy::ValTy> val_vec(num);
#else
     std::vector<uint64_t> val_vec(num);
#endif
     galois::runtime::gDeserialize(buf, val_vec);
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
     if (!FnTy::setVal_batch(from_id, &val_vec[0])) {
       galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n){
           uint32_t localID = mirrorNodes[from_id][n];
#ifdef __GALOIS_HET_OPENCL__
           {
           CLNodeDataWrapper d = clGraph.getDataW(localID);
           FnTy::setVal(localID, d, val_vec[n]);
           }
#else
           FnTy::setVal(localID, getData(localID), val_vec[n]);
#endif
           }, galois::loopname("BROADCAST_SET"));
      }
#endif
   }

#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
   template<typename FnTy>
#endif
   void syncRecvApplyPush(galois::runtime::RecvBuffer& buf) {
     unsigned from_id;
     uint32_t num;
     std::string loopName;
     uint32_t num_iter_push;
     galois::runtime::gDeserialize(buf, from_id, num);
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
     std::vector<typename FnTy::ValTy> val_vec(num);
#else
     std::vector<uint64_t> val_vec(num);
#endif
     galois::runtime::gDeserialize(buf, val_vec);
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
     if (!FnTy::reduce_batch(from_id, &val_vec[0])) {
       galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num),
           [&](uint32_t n){
           uint32_t lid = masterNodes[from_id][n];
#ifdef __GALOIS_HET_OPENCL__
       CLNodeDataWrapper d = clGraph.getDataW(lid);
       FnTy::reduce(lid, d, val_vec[n]);
#else
       FnTy::reduce(lid, getData(lid), val_vec[n]);
#endif
           }, galois::loopname("REDUCE_SET"));
     }
#endif
   }

#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
   template<typename FnTy>
#endif
   void simulate_broadcast(std::string loopName) {
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
      void (hGraph::*fn)(galois::runtime::RecvBuffer&) = &hGraph::syncRecvApplyPull<FnTy>;
#else
      void (hGraph::*fn)(galois::runtime::RecvBuffer&) = &hGraph::syncRecvApplyPull;
#endif
      galois::StatTimer StatTimer_syncBroadcast("SIMULATE_NET_BROADCAST");
      galois::Statistic SyncBroadcast_send_bytes("SIMULATE_NET_BROADCAST_SEND_BYTES");

#ifndef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
      galois::runtime::getHostBarrier().wait();
#endif
      StatTimer_syncBroadcast.start();
      auto& net = galois::runtime::getSystemNetworkInterface();

      for (unsigned x = 0; x < net.Num; ++x) {
         uint32_t num = masterNodes[x].size();
         if((x == id) || (num == 0))
           continue;

         galois::runtime::SendBuffer b;
         gSerialize(b, idForSelf(), fn, net.ID, num);

#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
         std::vector<typename FnTy::ValTy> val_vec(num);
#else
         std::vector<uint64_t> val_vec(num);
#endif

#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
         if (!FnTy::extract_batch(x, &val_vec[0])) {
           galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n){
               uint32_t localID = masterNodes[x][n];
#ifdef __GALOIS_HET_OPENCL__
               auto val = FnTy::extract((localID), clGraph.getDataR((localID)));
#else
               auto val = FnTy::extract((localID), getData(localID));
#endif
               val_vec[n] = val;

               }, galois::loopname("BROADCAST_EXTRACT"));
         }
#else
         val_vec[0] = 1;
#endif

         gSerialize(b, val_vec);

         SyncBroadcast_send_bytes += b.size();
         net.sendMsg(x, syncRecv, b);
      }
      //Will force all messages to be processed before continuing
      net.flush();

      galois::runtime::getHostBarrier().wait();
      StatTimer_syncBroadcast.stop();
   }

#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
   template<typename FnTy>
#endif
   void simulate_reduce(std::string loopName) {
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
      void (hGraph::*fn)(galois::runtime::RecvBuffer&) = &hGraph::syncRecvApplyPush<FnTy>;
#else
      void (hGraph::*fn)(galois::runtime::RecvBuffer&) = &hGraph::syncRecvApplyPush;
#endif
      galois::StatTimer StatTimer_syncReduce("SIMULATE_NET_REDUCE");
      galois::Statistic SyncReduce_send_bytes("SIMULATE_NET_REDUCE_SEND_BYTES");

#ifndef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
      galois::runtime::getHostBarrier().wait();
#endif
      StatTimer_syncReduce.start();
      auto& net = galois::runtime::getSystemNetworkInterface();

      for (unsigned x = 0; x < net.Num; ++x) {
         uint32_t num = mirrorNodes[x].size();
         if((x == id) || (num == 0))
           continue;

         galois::runtime::SendBuffer b;
         gSerialize(b, idForSelf(), fn, net.ID, num);

#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
         std::vector<typename FnTy::ValTy> val_vec(num);
#else
         std::vector<uint64_t> val_vec(num);
#endif

#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
         if (!FnTy::extract_reset_batch(x, &val_vec[0])) {
           galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n){
                uint32_t lid = mirrorNodes[x][n];
#ifdef __GALOIS_HET_OPENCL__
                CLNodeDataWrapper d = clGraph.getDataW(lid);
                auto val = FnTy::extract(lid, getData(lid, d));
                FnTy::reset(lid, d);
#else
                auto val = FnTy::extract(lid, getData(lid));
                FnTy::reset(lid, getData(lid));
#endif
                val_vec[n] = val;
               }, galois::loopname("REDUCE_EXTRACT"));
         }
#else
         val_vec[0] = 1;
#endif

         gSerialize(b, val_vec);

         SyncReduce_send_bytes += b.size();
         net.sendMsg(x, syncRecv, b);
      }
      //Will force all messages to be processed before continuing
      net.flush();

      galois::runtime::getHostBarrier().wait();

      StatTimer_syncReduce.stop();
   }
#endif


   template<typename FnTy>
   void reduce(std::string loopName) {
#ifdef __GALOIS_SIMULATE_COMMUNICATION__
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
      if (comm_mode == 1) {
        simulate_reduce<FnTy>(loopName);
        return;
      } else if (comm_mode == 2) {
        simulate_bare_mpi_reduce<FnTy>(loopName);
        return;
      }
#endif
#endif
      ++num_iter_push;
      std::string extract_timer_str("REDUCE_EXTRACT_" + loopName +"_" + std::to_string(num_run));
      std::string timer_str("REDUCE_" + loopName + "_" + std::to_string(num_run));
      std::string timer_barrier_str("REDUCE_BARRIER_" + loopName + "_" + std::to_string(num_run));
      std::string statSendBytes_str("SEND_BYTES_REDUCE_" + loopName + "_" + std::to_string(num_run));
      std::string doall_str("LAMBDA::REDUCE_" + loopName + "_" + std::to_string(num_run));
      galois::Statistic SyncReduce_send_bytes(statSendBytes_str);
      galois::StatTimer StatTimer_syncReduce(timer_str.c_str());
      galois::StatTimer StatTimerBarrier_syncReduce(timer_barrier_str.c_str());
      galois::StatTimer StatTimer_extract(extract_timer_str.c_str());

      StatTimer_syncReduce.start();
      auto& net = galois::runtime::getSystemNetworkInterface();

      for (unsigned x = 0; x < net.Num; ++x) {
         uint32_t num = mirrorNodes[x].size();
         if((x == id))
           continue;

         galois::runtime::SendBuffer b;

         StatTimer_extract.start();
         if(num > 0 ){
           std::vector<typename FnTy::ValTy> val_vec(num);

           if (!FnTy::extract_reset_batch(x, &val_vec[0])) {
             galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n){
                  uint32_t lid = mirrorNodes[x][n];
#ifdef __GALOIS_HET_OPENCL__
                  CLNodeDataWrapper d = clGraph.getDataW(lid);
                  auto val = FnTy::extract(lid, getData(lid, d));
                  FnTy::reset(lid, d);
#else
                  auto val = FnTy::extract(lid, getData(lid));
                  FnTy::reset(lid, getData(lid));
#endif
                  val_vec[n] = val;
                 }, galois::loopname(doall_str.c_str()));
           }

           gSerialize(b, val_vec);
         } else {
           gSerialize(b, loopName);
         }
         StatTimer_extract.stop();

         SyncReduce_send_bytes += b.size();
         net.sendTagged(x, galois::runtime::evilPhase, b);
      }
      //Will force all messages to be processed before continuing
      net.flush();

      //receive
      for (unsigned x = 0; x < net.Num; ++x) {
        if ((x == id))
          continue;
        decltype(net.recieveTagged(galois::runtime::evilPhase,nullptr)) p;
        do {
          net.handleReceives();
          p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
        } while (!p);
        syncRecvApply<FnTy>(p->first, p->second, loopName);
      }
      ++galois::runtime::evilPhase;

      StatTimer_syncReduce.stop();

   }

   template<typename FnTy>
   void broadcast(std::string loopName) {
#ifdef __GALOIS_SIMULATE_COMMUNICATION__
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
      if (comm_mode == 1) {
        simulate_broadcast<FnTy>(loopName);
        return;
      } else if (comm_mode == 2) {
        simulate_bare_mpi_broadcast<FnTy>(loopName);
        return;
      }
#endif
#endif
      ++num_iter_pull;
      std::string doall_str("LAMBDA::BROADCAST_" + loopName + "_" + std::to_string(num_run));
      std::string timer_str("BROADCAST_" + loopName +"_" + std::to_string(num_run));
      std::string timer_barrier_str("BROADCAST_BARRIER_" + loopName +"_" + std::to_string(num_run));
      std::string statSendBytes_str("SEND_BYTES_BROADCAST_" + loopName +"_" + std::to_string(num_run));
      galois::Statistic SyncBroadcast_send_bytes(statSendBytes_str);
      galois::StatTimer StatTimer_syncBroadcast(timer_str.c_str());
      galois::StatTimer StatTimer_extract("BROADCAST_EXTRACT", loopName);
      auto& net = galois::runtime::getSystemNetworkInterface();

      StatTimer_syncBroadcast.start();

      for (unsigned x = 0; x < net.Num; ++x) {
        uint32_t num = masterNodes[x].size();
        if((x == id))
          continue;

        galois::runtime::SendBuffer b;

        StatTimer_extract.start();
        if(num > 0 ){
          std::vector<typename FnTy::ValTy> val_vec(num);
          if (!FnTy::extract_batch(x, &val_vec[0])) {

            galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n){
                uint32_t localID = masterNodes[x][n];
#ifdef __GALOIS_HET_OPENCL__
                auto val = FnTy::extract((localID), clGraph.getDataR(localID));
#else
                auto val = FnTy::extract((localID), getData(localID));
#endif
                val_vec[n] = val;
                }, galois::loopname(doall_str.c_str()));
          }
          gSerialize(b, val_vec);
        } else {
          gSerialize(b, loopName);
        }
        StatTimer_extract.stop();

        SyncBroadcast_send_bytes += b.size();
        net.sendTagged(x, galois::runtime::evilPhase, b);

      }


      net.flush();

      //receive
      for (unsigned x = 0; x < net.Num; ++x) {
        if ((x == id))
          continue;
        decltype(net.recieveTagged(galois::runtime::evilPhase,nullptr)) p;
        do {
          net.handleReceives();
          p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
        } while (!p);
        syncBroadcastRecvApply<FnTy>(p->first, p->second, loopName);
      }

      ++galois::runtime::evilPhase;

      StatTimer_syncBroadcast.stop();
   }

   uint64_t getGID(uint32_t nodeID) const {
      return L2G(nodeID);
   }
   uint32_t getLID(uint64_t nodeID) const {
      return G2L(nodeID);
   }
   unsigned getHostID(uint64_t gid) {
      for (auto i = 0; i < hostNodes.size(); ++i) {
         uint64_t start, end;
         std::tie(start, end) = nodes_by_host_G(i);
         if (gid >= start && gid < end) {
            return i;
         }
      }
      return -1;
   }
   uint32_t getNumOwned() const {
      return numOwned;
   }
   uint64_t getGlobalOffset() const {
      return globalOffset;
   }
#ifdef __GALOIS_HET_CUDA__
   template<bool isVoidType, typename std::enable_if<isVoidType>::type* = nullptr>
   void setMarshalEdge(MarshalGraph &m, size_t index, edge_iterator &e) {
      // do nothing
   }
   template<bool isVoidType, typename std::enable_if<!isVoidType>::type* = nullptr>
   void setMarshalEdge(MarshalGraph &m, size_t index, edge_iterator &e) {
      m.edge_data[index] = getEdgeData(e);
   }
   MarshalGraph getMarshalGraph(unsigned host_id) {
      assert(host_id == id);
      MarshalGraph m;

      m.nnodes = size();
      m.nedges = sizeEdges();
      m.nowned = std::distance(begin(), end());
      assert(m.nowned > 0);
      m.id = host_id;
      m.row_start = (index_type *) calloc(m.nnodes + 1, sizeof(index_type));
      m.edge_dst = (index_type *) calloc(m.nedges, sizeof(index_type));

      // initialize node_data with localID-to-globalID mapping
      m.node_data = (index_type *) calloc(m.nnodes, sizeof(node_data_type));
      for (index_type i = 0; i < m.nnodes; ++i) {
        m.node_data[i] = getGID(i);
      }

      if (std::is_void<EdgeTy>::value) {
         m.edge_data = NULL;
      } else {
         if (!std::is_same<EdgeTy, edge_data_type>::value) {
            fprintf(stderr, "WARNING: Edge data type mismatch between CPU and GPU\n");
         }
         m.edge_data = (edge_data_type *) calloc(m.nedges, sizeof(edge_data_type));
      }

      // pinched from Rashid's LC_LinearArray_Graph.h
      size_t edge_counter = 0, node_counter = 0;
      for (auto n = begin(); n != ghost_end() && *n != m.nnodes; n++, node_counter++) {
         m.row_start[node_counter] = edge_counter;
         if (*n < m.nowned) {
            for (auto e = edge_begin(*n); e != edge_end(*n); e++) {
               if (getEdgeDst(e) < m.nnodes) {
                  setMarshalEdge<std::is_void<EdgeTy>::value>(m, edge_counter, e);
                  m.edge_dst[edge_counter++] = getEdgeDst(e);
               }
            }
         }
      }

      m.row_start[node_counter] = edge_counter;
      m.nedges = edge_counter;

      // copy memoization meta-data
      m.num_master_nodes = (unsigned int *) calloc(hostNodes.size(), sizeof(unsigned int));;
      m.master_nodes = (unsigned int **) calloc(hostNodes.size(), sizeof(unsigned int *));;
      for(uint32_t h = 0; h < hostNodes.size(); ++h){
        m.num_master_nodes[h] = masterNodes[h].size();
        if (masterNodes[h].size() > 0) {
          m.master_nodes[h] = (unsigned int *) calloc(masterNodes[h].size(), sizeof(unsigned int));;
          std::copy(masterNodes[h].begin(), masterNodes[h].end(), m.master_nodes[h]);
        } else {
          m.master_nodes[h] = NULL;
        }
      }
      m.num_mirror_nodes = (unsigned int *) calloc(hostNodes.size(), sizeof(unsigned int));;
      m.mirror_nodes = (unsigned int **) calloc(hostNodes.size(), sizeof(unsigned int *));;
      for(uint32_t h = 0; h < hostNodes.size(); ++h){
        m.num_mirror_nodes[h] = mirrorNodes[h].size();
        if (mirrorNodes[h].size() > 0) {
          m.mirror_nodes[h] = (unsigned int *) calloc(mirrorNodes[h].size(), sizeof(unsigned int));;
          std::copy(mirrorNodes[h].begin(), mirrorNodes[h].end(), m.mirror_nodes[h]);
        } else {
          m.mirror_nodes[h] = NULL;
        }
      }

      return m;
   }
#endif

#ifdef __GALOIS_HET_OPENCL__
public:
   typedef galois::OpenCL::Graphs::CL_LC_Graph<NodeTy, EdgeTy> CLGraphType;
   typedef typename CLGraphType::NodeDataWrapper CLNodeDataWrapper;
   typedef typename CLGraphType::NodeIterator CLNodeIterator;
   CLGraphType clGraph;
#endif

#ifdef __GALOIS_HET_OPENCL__
   const cl_mem & device_ptr() {
      return clGraph.device_ptr();
   }
   CLNodeDataWrapper getDataW(GraphNode N, galois::MethodFlag mflag = galois::MethodFlag::WRITE) {
      return clGraph.getDataW(N);
   }
   const CLNodeDataWrapper getDataR(GraphNode N,galois::MethodFlag mflag = galois::MethodFlag::READ) {
      return clGraph.getDataR(N);
   }

#endif


  /**For resetting num_iter_pull and push.**/
   void reset_num_iter(uint32_t runNum){
      num_iter_pull = 0;
      num_iter_push = 0;
      num_run = runNum;
   }
   uint32_t get_run_num() {
     return num_run;
   }
   /** Report stats to be printed.**/
   void reportStats(){
    statGhostNodes.report();
   }
};
#endif//_GALOIS_DIST_HGRAPH_H
