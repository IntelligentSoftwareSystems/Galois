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

#include "Galois/gstl.h"
#include "Galois/Graphs/LC_CSR_Graph.h"
#include "Galois/Runtime/Substrate.h"
#include "Galois/Runtime/Network.h"

//#include "Galois/Runtime/Barrier.h"
#include "Galois/Runtime/Serialize.h"

#include "Galois/Dist/GlobalObj.h"
#include "Galois/Dist/OfflineGraph.h"

#ifdef __GALOIS_HET_CUDA__
#include "Galois/Cuda/cuda_mtypes.h"
#endif

#ifdef __GALOIS_HET_OPENCL__
#include "Galois/OpenCL/CL_Header.h"
#endif

#ifndef _GALOIS_DIST_HGRAPH_H
#define _GALOIS_DIST_HGRAPH_H

template<typename NodeTy, typename EdgeTy, bool BSPNode = false, bool BSPEdge = false>
class hGraph: public GlobalObject {

   typedef typename std::conditional<BSPNode, std::pair<NodeTy, NodeTy>, NodeTy>::type realNodeTy;
   typedef typename std::conditional<BSPEdge, std::pair<EdgeTy, EdgeTy>, EdgeTy>::type realEdgeTy;

   typedef Galois::Graph::LC_CSR_Graph<realNodeTy, realEdgeTy> GraphTy;

   GraphTy graph;bool round;
   uint64_t totalNodes; // Total nodes in the complete graph.
   uint32_t numOwned; // [0, numOwned) = global nodes owned, thus [numOwned, numNodes are replicas
   uint64_t globalOffset; // [numOwned, end) + globalOffset = GID
   const unsigned id; // my hostid // FIXME: isn't this just Network::ID?
   //ghost cell ID translation
   std::vector<uint64_t> ghostMap; // GID = ghostMap[LID - numOwned]
   std::vector<std::pair<uint32_t, uint32_t> > hostNodes; //LID Node owned by host i
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
      return gid >= globalOffset && gid < globalOffset + numOwned;
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
   const NodeTy& getDataImpl(typename GraphTy::GraphNode N, Galois::MethodFlag mflag = Galois::MethodFlag::WRITE) const {
      auto& r = graph.getData(N, mflag);
      return round ? r.first : r.second;
   }

   template<bool en, typename std::enable_if<!en>::type* = nullptr>
   const NodeTy& getDataImpl(typename GraphTy::GraphNode N, Galois::MethodFlag mflag = Galois::MethodFlag::WRITE) const {
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
   GraphTy & getGraph() {
      return graph;
   }
   static void syncRecv(Galois::Runtime::RecvBuffer& buf) {
      uint32_t oid;
      void (hGraph::*fn)(Galois::Runtime::RecvBuffer&);
      Galois::Runtime::gDeserialize(buf, oid, fn);
      hGraph* obj = reinterpret_cast<hGraph*>(ptrForObj(oid));
      (obj->*fn)(buf);
      //--(obj->num_recv_expected);
      //std::cout << "[ " << Galois::Runtime::getSystemNetworkInterface().ID << "] " << " NUM RECV EXPECTED : " << (obj->num_recv_expected) << "\n";
   }

   template<typename FnTy>
   void syncRecvApply(Galois::Runtime::RecvBuffer& buf) {
      uint32_t num;
      Galois::Runtime::gDeserialize(buf, num);
      for (; num; --num) {
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
#ifdef __GALOIS_HET_OPENCL__
         CLNodeDataWrapper d = clGraph.getDataW(gid - globalOffset);
         FnTy::reduce((gid - globalOffset), d, val);
#else
         FnTy::reduce((gid - globalOffset), getData(gid - globalOffset), val);
#endif
      }
   }

   template<typename FnTy>
   void syncPullRecvReply(Galois::Runtime::RecvBuffer& buf) {
      void (hGraph::*fn)(Galois::Runtime::RecvBuffer&) = &hGraph::syncPullRecvApply<FnTy>;
      auto& net = Galois::Runtime::getSystemNetworkInterface();
      uint32_t num;
      unsigned from_id;
      Galois::Runtime::gDeserialize(buf, from_id, num);
      Galois::Runtime::SendBuffer b;
      gSerialize(b, idForSelf(), fn, num);
      for (; num; --num) {
         uint64_t gid;
         typename FnTy::ValTy old_val, val;
         Galois::Runtime::gDeserialize(buf, gid, old_val);
         assert(isOwned(gid));
#ifdef __GALOIS_HET_OPENCL__
         val = FnTy::extract((gid - globalOffset), clGraph.getDataR((gid - globalOffset)));
#else
         val = FnTy::extract((gid - globalOffset), getData((gid - globalOffset)));
#endif
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

      for (; num; --num) {
         uint64_t gid;
         typename FnTy::ValTy val;
         Galois::Runtime::gDeserialize(buf, gid, val);
         //assert(isGhost(gid));
         auto LocalId = G2L(gid);
         if (net.ID == 1) {
            //std::cout << "PullApply Step2 : [" << net.ID << "]  : [" << LocalId << "] : " << val << "\n";
         }
#ifdef __GALOIS_HET_OPENCL__
         {
            CLNodeDataWrapper d = clGraph.getDataW(LocalId);
            FnTy::setVal(LocalId, d, val);
         }
#else
         FnTy::setVal(LocalId, getData(LocalId), val);
#endif
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

   //hGraph construction is collective
   hGraph(const std::string& filename, unsigned host, unsigned numHosts, std::vector<unsigned> scalefactor = std::vector<unsigned>()) :
         GlobalObject(this), id(host), round(false) {
      OfflineGraph g(filename);
      //std::cerr << "Offline Graph Done\n";

      num_recv_expected = 0;
      totalNodes = g.size();
      std::cerr << "Total nodes : " << totalNodes << "\n";
      std::cerr << "Total edges : " << g.sizeEdges() << "\n";
      //compute owners for all nodes
      if (scalefactor.empty() || (numHosts == 1)) {
         for (unsigned i = 0; i < numHosts; ++i)
            gid2host.push_back(Galois::block_range(0U, (unsigned) g.size(), i, numHosts));
      } else {
         assert(scalefactor.size() == numHosts);
         unsigned numBlocks = 0;
         for (unsigned i = 0; i < numHosts; ++i)
            numBlocks += scalefactor[i];
         std::vector<std::pair<uint64_t, uint64_t>> blocks;
         for (unsigned i = 0; i < numBlocks; ++i)
            blocks.push_back(Galois::block_range(0U, (unsigned) g.size(), i, numBlocks));
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
   }

   template<bool isVoidType, typename std::enable_if<!isVoidType>::type* = nullptr>
   void loadEdges(OfflineGraph & g) {
      fprintf(stderr, "Loading edge-data while creating edges.\n");

      uint64_t cur = 0;
      Galois::Timer timer;
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
   void loadEdges(OfflineGraph & g) {
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

   const NodeTy& getData(GraphNode N, Galois::MethodFlag mflag = Galois::MethodFlag::WRITE) const {
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

   template<typename FnTy>
   void sync_push() {
      void (hGraph::*fn)(Galois::Runtime::RecvBuffer&) = &hGraph::syncRecvApply<FnTy>;
      Galois::Timer time1, time2;
      time1.start();
      auto& net = Galois::Runtime::getSystemNetworkInterface();
      for (unsigned x = 0; x < hostNodes.size(); ++x) {
         if (x == id)
            continue;
         uint32_t start, end;
         std::tie(start, end) = nodes_by_host(x);
         //std::cout << net.ID << " " << x << " " << start << " " << end << "\n";
         if (start == end)
            continue;
         Galois::Runtime::SendBuffer b;
         gSerialize(b, idForSelf(), fn, (uint32_t) (end - start));
         for (; start != end; ++start) {
            auto gid = L2G(start);
            if (gid > totalNodes) {
               //std::cout << "[" << net.ID << "] GID : " << gid << " size : " << graph.size() << "\n";

               assert(gid < totalNodes);
            }
            //std::cout << net.ID << " send (" << gid << ") " << start << " " << FnTy::extract(start, getData(start)) << "\n";
#ifdef __GALOIS_HET_OPENCL__
            CLNodeDataWrapper d = clGraph.getDataW(start);
            gSerialize(b, gid, FnTy::extract(start, d));
            FnTy::reset(start, d);
#else
            gSerialize(b, gid, FnTy::extract(start, getData(start)));
            FnTy::reset(start, getData(start));
#endif
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
   void sync_pull() {
      void (hGraph::*fn)(Galois::Runtime::RecvBuffer&) = &hGraph::syncPullRecvReply<FnTy>;
      auto& net = Galois::Runtime::getSystemNetworkInterface();
      //Galois::Runtime::getHostBarrier().wait();
      num_recv_expected = 0;
      for (unsigned x = 0; x < hostNodes.size(); ++x) {
         if (x == id)
            continue;
         uint32_t start, end;
         std::tie(start, end) = nodes_by_host(x);
         //std::cout << "end - start" << (end - start) << "\n";
         if (start == end)
            continue;
         Galois::Runtime::SendBuffer b;
         gSerialize(b, idForSelf(), fn, net.ID, (uint32_t) (end - start));
         for (; start != end; ++start) {
            auto gid = L2G(start);
            //std::cout << net.ID << " PULL send (" << gid << ") " << start << " " << FnTy::extract(start, getData(start)) << "\n";
#ifdef __GALOIS_HET_OPENCL__
            gSerialize(b, gid, FnTy::extract(start, clGraph.getDataR(start)));
#else
            gSerialize(b, gid, FnTy::extract(start, getData(start)));
#endif
         }
         net.send(x, syncRecv, b);
         ++num_recv_expected;
      }

      //std::cout << "[" << net.ID <<"] num_recv_expected : "<< num_recv_expected << "\n";

      net.flush();
      while (num_recv_expected) {
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
      MarshalGraph m;

      m.nnodes = size();
      m.nedges = sizeEdges();
      m.nowned = std::distance(begin(), end());
      assert(m.nowned > 0);
      m.g_offset = getGID(0);
      m.id = host_id;
      m.row_start = (index_type *) calloc(m.nnodes + 1, sizeof(index_type));
      m.edge_dst = (index_type *) calloc(m.nedges, sizeof(index_type));

      // TODO: initialize node_data
      m.node_data = NULL;

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
      return m;
   }
#endif

#ifdef __GALOIS_HET_OPENCL__
public:
   typedef Galois::OpenCL::Graphs::CL_LC_Graph<NodeTy, EdgeTy> CLGraphType;
   typedef typename CLGraphType::NodeDataWrapper CLNodeDataWrapper;
   typedef typename CLGraphType::NodeIterator CLNodeIterator;
   CLGraphType clGraph;
#endif

#ifdef __GALOIS_HET_OPENCL__
   const cl_mem & device_ptr() {
      return clGraph.device_ptr();
   }
   CLNodeDataWrapper getDataW(GraphNode N, Galois::MethodFlag mflag = Galois::MethodFlag::WRITE) {
      return clGraph.getDataW(N);
   }
   const CLNodeDataWrapper getDataR(GraphNode N,Galois::MethodFlag mflag = Galois::MethodFlag::READ) {
      return clGraph.getDataR(N);
   }

#endif

};
#endif//_GALOIS_DIST_HGRAPH_H
