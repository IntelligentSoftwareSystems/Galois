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

#include "Galois/Runtime/Serialize.h"
#include "Galois/Statistic.h"

#include "Galois/Runtime/GlobalObj.h"
#include "Galois/Runtime/OfflineGraph.h"
#include <vector>
#include <set>
#include <algorithm>
#include <unordered_map>
#include <iostream>
#include <bitset>

#include "Galois/Runtime/Dynamic_bitset.h"

//#include "Galois/Runtime/dGraph_vertexCut.h"
//#include "Galois/Runtime/dGraph_edgeCut.h"

#ifdef __GALOIS_HET_CUDA__
#include "Galois/Runtime/Cuda/cuda_mtypes.h"
#endif

#ifdef __GALOIS_HET_OPENCL__
#include "Galois/OpenCL/CL_Header.h"
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

  public:
   typedef typename std::conditional<BSPNode, std::pair<NodeTy, NodeTy>, NodeTy>::type realNodeTy;
   typedef typename std::conditional<BSPEdge && !std::is_void<EdgeTy>::value, std::pair<EdgeTy, EdgeTy>, EdgeTy>::type realEdgeTy;

   typedef Galois::Graph::LC_CSR_Graph<realNodeTy, realEdgeTy> GraphTy;

   GraphTy graph;
   bool round;
   uint64_t totalNodes; // Total nodes in the complete graph.
   uint64_t totalSlaveNodes; // Total slave nodes from others.
   uint32_t numOwned; // [0, numOwned) = global nodes owned, thus [numOwned, numNodes are replicas
   uint64_t globalOffset; // [numOwned, end) + globalOffset = GID
   const unsigned id; // my hostid // FIXME: isn't this just Network::ID?
   const uint32_t numHosts;
   //ghost cell ID translation

  //memoization optimization
  std::vector<std::vector<size_t>> slaveNodes; // slave nodes from different hosts. For sync_push
  std::vector<std::vector<size_t>> masterNodes; // master nodes on different hosts. For sync_pull

  //using vector of atomic 64 bit ints for tracking dirty fields
  Galois::DynamicBitSet bit_set_compute;


  /****** VIRTUAL FUNCTIONS *********/
  virtual uint32_t G2L(uint64_t) const = 0 ;
  virtual uint64_t L2G(uint32_t) const = 0;
  virtual bool is_vertex_cut() const = 0;
  virtual std::pair<uint32_t, uint32_t> nodes_by_host(uint32_t) const = 0;
  virtual std::pair<uint64_t, uint64_t> nodes_by_host_G(uint32_t) const = 0;
  virtual unsigned getHostID(uint64_t) const = 0;
  virtual bool isOwned(uint64_t) const = 0;
  virtual uint64_t get_local_total_nodes() const = 0;


#ifdef __GALOIS_SIMULATE_COMMUNICATION__
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
  unsigned comm_mode; // Communication mode: 0 - original, 1 - simulated net, 2 - simulated bare MPI
#endif
#endif


   uint32_t num_recv_expected; // Number of receives expected for local completion.
   uint32_t num_iter_push; //Keep track of number of iterations.
   uint32_t num_iter_pull; //Keep track of number of iterations.
   uint32_t num_run; //Keep track of number of runs.
   uint32_t num_iteration; //Keep track of number of iterations.

  //Stats: for rough estimate of sendBytes.
   Galois::Statistic statGhostNodes;

   /****** checkpointing **********/
   Galois::Runtime::RecvBuffer checkpoint_recvBuffer;
  // Select from edgeCut or vertexCut
  //typedef typename std::conditional<PartitionTy, DS_vertexCut ,DS_edgeCut>::type DS_type;
  //DS_type DS;


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
#ifdef __GALOIS_SIMULATE_COMMUNICATION__
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
  void set_comm_mode(unsigned mode) { // Communication mode: 0 - original, 1 - simulated net, 2 - simulated bare MPI
    comm_mode = mode;
  }
#endif
#endif
  static void syncRecv(uint32_t src, Galois::Runtime::RecvBuffer& buf) {
      uint32_t oid;
      void (hGraph::*fn)(Galois::Runtime::RecvBuffer&);
      Galois::Runtime::gDeserialize(buf, oid, fn);
      hGraph* obj = reinterpret_cast<hGraph*>(ptrForObj(oid));
      (obj->*fn)(buf);
   }

   void exchange_info_landingPad(Galois::Runtime::RecvBuffer& buf){
     uint32_t hostID;
     uint64_t numItems;
     std::vector<uint64_t> items;
     Galois::Runtime::gDeserialize(buf, hostID, numItems);

     Galois::Runtime::gDeserialize(buf, masterNodes[hostID]);
     std::cout << "from : " << hostID << " -> " << numItems << " --> " << masterNodes[hostID].size() << "\n";
   }



   template<typename FnTy>
     void syncRecvApply_ck(uint32_t from_id, Galois::Runtime::RecvBuffer& buf, std::string loopName) {
       auto& net = Galois::Runtime::getSystemNetworkInterface();
       std::string set_timer_str("SYNC_SET_" + loopName + "_" + get_run_identifier());
       std::string doall_str("LAMBDA::SYNC_PUSH_RECV_APPLY_" + loopName + "_" + get_run_identifier());
       Galois::StatTimer StatTimer_set(set_timer_str.c_str());
       StatTimer_set.start();

       uint32_t num = masterNodes[from_id].size();
       std::vector<typename FnTy::ValTy> val_vec(num);
       Galois::Runtime::gDeserialize(buf, val_vec);
       if(num > 0){
         if (!FnTy::reduce_batch(from_id, &val_vec[0])) {
           Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num),
               [&](uint32_t n){
               uint32_t lid = masterNodes[from_id][n];
#ifdef __GALOIS_HET_OPENCL__
           CLNodeDataWrapper d = clGraph.getDataW(lid);
           FnTy::reduce(lid, d, val_vec[n]);
#else
           FnTy::reduce(lid, getData(lid), val_vec[n]);
#endif
               }, Galois::loopname(doall_str.c_str()), Galois::numrun(get_run_identifier()));
         }
       }
       if(net.ID == (from_id + 1)%net.Num){
        checkpoint_recvBuffer = std::move(buf);
       }
       StatTimer_set.stop();
   }

   template<typename FnTy>
     void syncRecvApply(uint32_t from_id, Galois::Runtime::RecvBuffer& buf, std::string loopName) {
       auto& net = Galois::Runtime::getSystemNetworkInterface();
       std::string set_timer_str("SYNC_SET_" + loopName + "_" + get_run_identifier());
       std::string doall_str("LAMBDA::SYNC_PUSH_RECV_APPLY_" + loopName + "_" + get_run_identifier());
       Galois::StatTimer StatTimer_set(set_timer_str.c_str());
       StatTimer_set.start();

       uint32_t num = masterNodes[from_id].size();
       if(num > 0){

         Galois::DynamicBitSet bit_set_comm;
         bit_set_comm.resize_init(num);

         Galois::Runtime::gDeserialize(buf, bit_set_comm);
         auto bits_set_num = bit_set_comm.bit_count(num);

         if(bits_set_num > 0){
         std::vector<typename FnTy::ValTy> val_vec(bits_set_num);
         Galois::Runtime::gDeserialize(buf, val_vec);

         if (!FnTy::reduce_batch(from_id, &val_vec[0])) {
           Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num),
               [&](uint32_t n){

               if(bit_set_comm.is_set(n)){

                  uint32_t lid = masterNodes[from_id][n];
                  uint32_t vec_index = bit_set_comm.bit_count(n);

#ifdef __GALOIS_HET_OPENCL__
                 CLNodeDataWrapper d = clGraph.getDataW(lid);
                 FnTy::reduce(lid, d, val_vec[vec_index]);
#else

                 FnTy::reduce(lid, getData(lid), val_vec[vec_index]);
#endif
               }
               }, Galois::loopname(doall_str.c_str()), Galois::numrun(get_run_identifier()));
         }
        }
       }
       StatTimer_set.stop();
   }

   template<typename FnTy>
   void syncPullRecvReply(Galois::Runtime::RecvBuffer& buf) {
      void (hGraph::*fn)(Galois::Runtime::RecvBuffer&) = &hGraph::syncPullRecvApply<FnTy>;
      auto& net = Galois::Runtime::getSystemNetworkInterface();
      uint32_t num;
      unsigned from_id;
      std::string loopName;
      uint32_t num_iter_pull;
      Galois::Runtime::gDeserialize(buf, loopName, num_iter_pull, from_id, num);
      std::string extract_timer_str("SYNC_EXTRACT_" + loopName + "_" + get_run_identifier());
      Galois::StatTimer StatTimer_extract(extract_timer_str.c_str());
      std::string statSendBytes_str("SEND_BYTES_SYNC_PULL_REPLY_" + loopName + "_" + get_run_identifier());
      Galois::Statistic SyncPullReply_send_bytes(statSendBytes_str);
      std::string doall_str("LAMBDA::SYNC_PULL_RECV_REPLY_" + loopName + "_" + get_run_identifier());
      Galois::Runtime::SendBuffer b;
      gSerialize(b, idForSelf(), fn, loopName, num_iter_pull, net.ID, num);

      assert(num == masterNodes[from_id].size());
      StatTimer_extract.start();
      if(num > 0){
        std::vector<typename FnTy::ValTy> val_vec(num);

        if (!FnTy::extract_batch(from_id, &val_vec[0])) {
          Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n){
              uint32_t localID = masterNodes[from_id][n];
#ifdef __GALOIS_HET_OPENCL__
              auto val = FnTy::extract((localID), clGraph.getDataR((localID)));
#else
              auto val = FnTy::extract((localID), getData(localID));
#endif
              assert(n < num);
              val_vec[n] = val;

              }, Galois::loopname(doall_str.c_str()), Galois::numrun(get_run_identifier()));
        }

        Galois::Runtime::gSerialize(b, val_vec);
      }
      StatTimer_extract.stop();

      SyncPullReply_send_bytes += b.size();
      net.sendMsg(from_id, syncRecv, b);
   }

   template<typename FnTy>
   void syncPullRecvApply(uint32_t from_id, Galois::Runtime::RecvBuffer& buf, std::string loopName) {
      std::string set_timer_str("SYNC_SET_" + loopName + "_" + get_run_identifier());
      Galois::StatTimer StatTimer_set(set_timer_str.c_str());
      std::string doall_str("LAMBDA::SYNC_PULL_RECV_APPLY_" + loopName + "_" + get_run_identifier());
      auto& net = Galois::Runtime::getSystemNetworkInterface();

      uint32_t num = slaveNodes[from_id].size();

      StatTimer_set.start();

      if(num > 0){
         Galois::DynamicBitSet bit_set_comm;
         bit_set_comm.resize_init(num);

         Galois::Runtime::gDeserialize(buf, bit_set_comm);
         auto bits_set_num = bit_set_comm.bit_count(num);

         if(bits_set_num > 0){
         std::vector<typename FnTy::ValTy> val_vec(bits_set_num);
         Galois::Runtime::gDeserialize(buf, val_vec);

        if (!FnTy::setVal_batch(from_id, &val_vec[0])) {
          Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n){

              if(bit_set_comm.is_set(n)){
                uint32_t localID = slaveNodes[from_id][n];
                uint32_t vec_index = bit_set_comm.bit_count(n);

#ifdef __GALOIS_HET_OPENCL__
                  {
                  CLNodeDataWrapper d = clGraph.getDataW(localID);
                  FnTy::setVal(localID, d, val_vec[vec_index]);
                  }
#else
                  FnTy::setVal(localID, getData(localID), val_vec[vec_index]);
#endif

              }
              }, Galois::loopname(doall_str.c_str()), Galois::numrun(get_run_identifier()));
         }
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


   //hGraph(const std::string& filename, const std::string& partitionFolder, unsigned host, unsigned numHosts, std::vector<unsigned> scalefactor = std::vector<unsigned>()) :
   hGraph(unsigned host, unsigned numHosts) :
         GlobalObject(this), id(host), numHosts(numHosts), round(false),statGhostNodes("TotalGhostNodes") {
#ifdef __GALOIS_SIMULATE_COMMUNICATION__
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
      comm_mode = 0;
#endif
#endif
      masterNodes.resize(numHosts);
      slaveNodes.resize(numHosts);
      //masterNodes_bitvec.resize(numHosts);
      num_recv_expected = 0;
      num_iter_push = 0;
      num_iter_pull = 0;
      num_run = 0;
      num_iteration = 0;

      //uint32_t numNodes;
      //uint64_t numEdges;
#if 0
      std::string part_fileName = getPartitionFileName(filename,partitionFolder,id,numHosts);
      //OfflineGraph g(part_fileName);
      hGraph(filename, partitionFolder, host, numHosts, scalefactor, numNodes, numOwned, numEdges, totalNodes, id);
      graph.allocateFrom(numNodes, numEdges);
      //std::cerr << "Allocate done\n";

      graph.constructNodes();
      //std::cerr << "Construct nodes done\n";
      loadEdges(graph, g);
      std::cerr << "Edges loaded \n";
      //testPart<PartitionTy>(g);

#ifdef __GALOIS_HET_OPENCL__
      clGraph.load_from_hgraph(*this);
#endif

      setup_communication();


#ifdef __GALOIS_SIMULATE_COMMUNICATION__
#ifndef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
      simulate_communication();
#endif
#endif
#endif
   }
   void setup_communication() {
      Galois::StatTimer StatTimer_comm_setup("COMMUNICATION_SETUP_TIME");
      Galois::Runtime::getHostBarrier().wait();
      StatTimer_comm_setup.start();


      //Exchange information for memoization optimization.
      exchange_info_init();

      for(uint32_t h = 0; h < masterNodes.size(); ++h){
         Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(masterNodes[h].size()),
             [&](uint32_t n){
             masterNodes[h][n] = G2L(masterNodes[h][n]);
             }, Galois::loopname("MASTER_NODES"), Galois::numrun(get_run_identifier()));
      }

      for(uint32_t h = 0; h < slaveNodes.size(); ++h){
         Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(slaveNodes[h].size()),
             [&](uint32_t n){
             slaveNodes[h][n] = G2L(slaveNodes[h][n]);
             }, Galois::loopname("SLAVE_NODES"), Galois::numrun(get_run_identifier()));
      }

      for(auto x = 0; x < masterNodes.size(); ++x){
        //masterNodes_bitvec[x].resize(masterNodes[x].size());
        std::string master_nodes_str = "MASTER_NODES_TO_" + std::to_string(x);
        Galois::Statistic StatMasterNodes(master_nodes_str);
        StatMasterNodes += masterNodes[x].size();
      }

      totalSlaveNodes = 0;
      for(auto x = 0; x < slaveNodes.size(); ++x){
        std::string slave_nodes_str = "SLAVE_NODES_FROM_" + std::to_string(x);
        Galois::Statistic StatSlaveNodes(slave_nodes_str);
        StatSlaveNodes += slaveNodes[x].size();
        totalSlaveNodes += slaveNodes[x].size();
      }

      std::cout << "["<< id << "]" << "Total local nodes : " << get_local_total_nodes() << " NumOwned : "<< numOwned <<"\n"; 
      bit_set_compute.resize_init(get_local_total_nodes());

      send_info_to_host();
      StatTimer_comm_setup.stop();
   }

#ifdef __GALOIS_SIMULATE_COMMUNICATION__
   void simulate_communication() {
     for (int i = 0; i < 10; ++i) {
     simulate_sync_pull("");
     simulate_sync_push("");

#ifdef __GALOIS_SIMULATE_BARE_MPI_COMMUNICATION__
     simulate_bare_mpi_sync_pull("");
     simulate_bare_mpi_sync_push("");
#endif
     }
   }
#endif

#if 0
   template<bool PartTy, typename std::enable_if<!std::integral_constant<bool, PartTy>::value>::type* = nullptr>
   void testPart(OfflineGraph& g){
       std::cout << "False type... edge partition\n";
   }

   template<bool PartTy, typename std::enable_if<std::integral_constant<bool, PartTy>::value>::type* = nullptr>
   void testPart(OfflineGraph& g){
       std::cout << "true type... vertex partition\n";
   }
#endif

#if 0
   template<bool isVoidType, typename std::enable_if<!isVoidType>::type* = nullptr>
   void loadEdges(OfflineGraph & g) {
      fprintf(stderr, "Loading edge-data while creating edges.\n");

      uint64_t cur = 0;
      Galois::Timer timer;
      std::cout <<"["<<id<<"]PRE :: NumSeeks ";
      g.num_seeks();
      g.reset_seek_counters();
      timer.start();
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

#endif

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

  void exchange_info_init(){
    auto& net = Galois::Runtime::getSystemNetworkInterface();
    //may be reusing tag, so need a barrier
    Galois::Runtime::getHostBarrier().wait();

    for (unsigned x = 0; x < net.Num; ++x) {
      if((x == id))
        continue;

      Galois::Runtime::SendBuffer b;
      gSerialize(b, (uint64_t)slaveNodes[x].size(), slaveNodes[x]);
      net.sendTagged(x, 1, b);
      std::cout << " number of slaves from : " << x << " : " << slaveNodes[x].size() << "\n";
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
      Galois::Runtime::gDeserialize(p->second, numItems);
      Galois::Runtime::gDeserialize(p->second, masterNodes[p->first]);
      std::cout << "from : " << p->first << " -> " << numItems << " --> " << masterNodes[p->first].size() << "\n";
    }

    //may be reusing tag, so need a barrier
    Galois::Runtime::getHostBarrier().wait();
   }

  void send_info_to_host(){
    auto& net = Galois::Runtime::getSystemNetworkInterface();
    //may be reusing tag, so need a barrier
    Galois::Runtime::getHostBarrier().wait();

    //send info to host 0, msg tagged with 1
    for(unsigned x = 0; x < net.Num; ++x){
      if(x == id)
        continue;
      Galois::Runtime::SendBuffer b;
      gSerialize(b, totalSlaveNodes);
      net.sendTagged(x,1,b);
    }

    //receive and print
      uint64_t global_total_slave_nodes = totalSlaveNodes;
      for(unsigned x = 0; x < net.Num; ++x){
        if(x == id)
          continue;
        decltype(net.recieveTagged(1, nullptr)) p;
        do{
          net.handleReceives();
          p = net.recieveTagged(1, nullptr);
        }while(!p);

        uint64_t total_slave_nodes_from_others;
        Galois::Runtime::gDeserialize(p->second, total_slave_nodes_from_others);
        global_total_slave_nodes += total_slave_nodes_from_others;
      }

      float replication_factor = (float)(global_total_slave_nodes + totalNodes)/(float)totalNodes;
      Galois::Runtime::reportStat("(NULL)", "REPLICATION_FACTOR_" + get_run_identifier(), std::to_string(replication_factor), 0);
      Galois::Runtime::reportStat("(NULL)", "TOTAL_NODES_" + get_run_identifier(), totalNodes, 0);
      Galois::Runtime::reportStat("(NULL)", "TOTAL_GLOBAL_GHOSTNODES_" + get_run_identifier(), global_total_slave_nodes, 0);

    //may be reusing tag, so need a barrier
    Galois::Runtime::getHostBarrier().wait();

  }

#ifdef __GALOIS_SIMULATE_COMMUNICATION__
#ifdef __GALOIS_SIMULATE_BARE_MPI_COMMUNICATION__
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
   template<typename FnTy>
#endif
   void simulate_bare_mpi_sync_pull(std::string loopName, bool mem_copy = false) {
      //std::cerr << "WARNING: requires MPI_THREAD_MULTIPLE to be set in MPI_Init_thread() and Net to not receive MPI messages with tag 32767\n";
      std::string statSendBytes_str("SIMULATE_MPI_SEND_BYTES_SYNC_PULL_" + loopName + "_" + std::to_string(num_run));
      Galois::Statistic SyncPull_send_bytes(statSendBytes_str);
      std::string timer_str("SIMULATE_MPI_SYNC_PULL_" + loopName + "_" + std::to_string(num_run));
      Galois::StatTimer StatTimer_syncPull(timer_str.c_str());
      std::string timer_barrier_str("SIMULATE_MPI_SYNC_PULL_BARRIER_" + loopName + "_" + std::to_string(num_run));
      Galois::StatTimer StatTimerBarrier_syncPull(timer_barrier_str.c_str());
      std::string extract_timer_str("SIMULATE_MPI_SYNC_PULL_EXTRACT_" + loopName +"_" + std::to_string(num_run));
      Galois::StatTimer StatTimer_extract(extract_timer_str.c_str());
      std::string set_timer_str("SIMULATE_MPI_SYNC_PULL_SET_" + loopName +"_" + std::to_string(num_run));
      Galois::StatTimer StatTimer_set(set_timer_str.c_str());

#ifndef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      StatTimer_syncPull.start();
      auto& net = Galois::Runtime::getSystemNetworkInterface();

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
           Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n){
               uint32_t localID = masterNodes[x][n];
#ifdef __GALOIS_HET_OPENCL__
               auto val = FnTy::extract((localID), clGraph.getDataR((localID)));
#else
               auto val = FnTy::extract((localID), getData(localID));
#endif
               val_vec[n] = val;

               }, Galois::loopname("SYNC_PULL_EXTRACT"), Galois::numrun(get_run_identifier()));
         }
#else
         val_vec[0] = 1;
#endif
         
         if (mem_copy) {
           bs[x].resize(size);
           memcpy(bs[x].data(), sb[x].data(), size);
         }

         StatTimer_extract.stop();

         SyncPull_send_bytes += size;
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
         uint32_t num = slaveNodes[x].size();
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

      StatTimerBarrier_syncPull.start();
      MPI_Waitall(num_requests, &requests[0], MPI_STATUSES_IGNORE);
      StatTimerBarrier_syncPull.stop();

      for (unsigned x = 0; x < net.Num; ++x) {
         uint32_t num = slaveNodes[x].size();
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
           Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n){
               uint32_t localID = slaveNodes[x][n];
#ifdef __GALOIS_HET_OPENCL__
               {
               CLNodeDataWrapper d = clGraph.getDataW(localID);
               FnTy::setVal(localID, d, val_vec[n]);
               }
#else
               FnTy::setVal(localID, getData(localID), val_vec[n]);
#endif
               }, Galois::loopname("SYNC_PULL_SET"), Galois::numrun(get_run_identifier()));
          }
#endif

         StatTimer_set.stop();
      }

      //std::cerr << "[" << id << "]" << "pull mpi done\n";
      StatTimer_syncPull.stop();
   }

#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
   template<typename FnTy>
#endif
   void simulate_bare_mpi_sync_push(std::string loopName, bool mem_copy = false) {
      //std::cerr << "WARNING: requires MPI_THREAD_MULTIPLE to be set in MPI_Init_thread() and Net to not receive MPI messages with tag 32767\n";
      std::string statSendBytes_str("SIMULATE_MPI_SEND_BYTES_SYNC_PUSH_" + loopName + "_" + std::to_string(num_run));
      Galois::Statistic SyncPush_send_bytes(statSendBytes_str);
      std::string timer_str("SIMULATE_MPI_SYNC_PUSH_" + loopName + "_" + std::to_string(num_run));
      Galois::StatTimer StatTimer_syncPush(timer_str.c_str());
      std::string timer_barrier_str("SIMULATE_MPI_SYNC_PUSH_BARRIER_" + loopName + "_" + std::to_string(num_run));
      Galois::StatTimer StatTimerBarrier_syncPush(timer_barrier_str.c_str());
      std::string extract_timer_str("SIMULATE_MPI_SYNC_PUSH_EXTRACT_" + loopName +"_" + std::to_string(num_run));
      Galois::StatTimer StatTimer_extract(extract_timer_str.c_str());
      std::string set_timer_str("SIMULATE_MPI_SYNC_PUSH_SET_" + loopName +"_" + std::to_string(num_run));
      Galois::StatTimer StatTimer_set(set_timer_str.c_str());

#ifndef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      StatTimer_syncPush.start();
      auto& net = Galois::Runtime::getSystemNetworkInterface();

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
         uint32_t num = slaveNodes[x].size();
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
           Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n){
                uint32_t lid = slaveNodes[x][n];
#ifdef __GALOIS_HET_OPENCL__
                CLNodeDataWrapper d = clGraph.getDataW(lid);
                auto val = FnTy::extract(lid, getData(lid, d));
                FnTy::reset(lid, d);
#else
                auto val = FnTy::extract(lid, getData(lid));
                FnTy::reset(lid, getData(lid));
#endif
                val_vec[n] = val;
               }, Galois::loopname("SYNC_PUSH_EXTRACT"), Galois::numrun(get_run_identifier()));
         }
#else
         val_vec[0] = 1;
#endif
         
         if (mem_copy) {
           bs[x].resize(size);
           memcpy(bs[x].data(), sb[x].data(), size);
         }

         StatTimer_extract.stop();

         SyncPush_send_bytes += size;
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

      StatTimerBarrier_syncPush.start();
      MPI_Waitall(num_requests, &requests[0], MPI_STATUSES_IGNORE);
      StatTimerBarrier_syncPush.stop();

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
           Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num),
               [&](uint32_t n){
               uint32_t lid = masterNodes[x][n];
#ifdef __GALOIS_HET_OPENCL__
           CLNodeDataWrapper d = clGraph.getDataW(lid);
           FnTy::reduce(lid, d, val_vec[n]);
#else
           FnTy::reduce(lid, getData(lid), val_vec[n]);
#endif
               }, Galois::loopname("SYNC_PUSH_SET"), Galois::numrun(get_run_identifier()));
         }
#endif

         StatTimer_set.stop();
      }
      
      //std::cerr << "[" << id << "]" << "push mpi done\n";
      StatTimer_syncPush.stop();
   }

#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
   template<typename FnTy>
#endif
   void simulate_bare_mpi_sync_pull_serialized() {
      //std::cerr << "WARNING: requires MPI_THREAD_MULTIPLE to be set in MPI_Init_thread() and Net to not receive MPI messages with tag 32767\n";
      Galois::StatTimer StatTimer_syncPull("SIMULATE_MPI_SYNC_PULL");
      Galois::Statistic SyncPull_send_bytes("SIMULATE_MPI_SYNC_PULL_SEND_BYTES");

#ifndef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      StatTimer_syncPull.start();
      auto& net = Galois::Runtime::getSystemNetworkInterface();

      std::vector<MPI_Request> requests(2 * net.Num);
      unsigned num_requests = 0;

      Galois::Runtime::SendBuffer sb[net.Num];
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
           Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n){
               uint32_t localID = masterNodes[x][n];
#ifdef __GALOIS_HET_OPENCL__
               auto val = FnTy::extract((localID), clGraph.getDataR((localID)));
#else
               auto val = FnTy::extract((localID), getData(localID));
#endif
               val_vec[n] = val;

               }, Galois::loopname("SYNC_PULL_EXTRACT"), Galois::numrun(get_run_identifier()));
         }
#else
         val_vec[0] = 1;
#endif

         Galois::Runtime::gSerialize(sb[x], val_vec);
         assert(size == sb[x].size());
         
         SyncPull_send_bytes += size;
         //std::cerr << "[" << id << "]" << " mpi send to " << x << " : " << size << "\n";
         MPI_Isend(sb[x].linearData(), size, MPI_BYTE, x, 32767, MPI_COMM_WORLD, &requests[num_requests++]);
      }

      Galois::Runtime::RecvBuffer rb[net.Num];
      for (unsigned x = 0; x < net.Num; ++x) {
         uint32_t num = slaveNodes[x].size();
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
         uint32_t num = slaveNodes[x].size();
         if((x == id) || (num == 0))
           continue;
         //std::cerr << "[" << id << "]" << " mpi received from " << x << "\n";
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
         std::vector<typename FnTy::ValTy> val_vec(num);
#else
         std::vector<uint64_t> val_vec(num);
#endif
         Galois::Runtime::gDeserialize(rb[x], val_vec);
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
         if (!FnTy::setVal_batch(x, &val_vec[0])) {
           Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n){
               uint32_t localID = slaveNodes[x][n];
#ifdef __GALOIS_HET_OPENCL__
               {
               CLNodeDataWrapper d = clGraph.getDataW(localID);
               FnTy::setVal(localID, d, val_vec[n]);
               }
#else
               FnTy::setVal(localID, getData(localID), val_vec[n]);
#endif
               }, Galois::loopname("SYNC_PULL_SET"), Galois::numrun(get_run_identifier()));
          }
#endif
      }

      //std::cerr << "[" << id << "]" << "pull mpi done\n";
      StatTimer_syncPull.stop();
   }

#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
   template<typename FnTy>
#endif
   void simulate_bare_mpi_sync_push_serialized() {
      //std::cerr << "WARNING: requires MPI_THREAD_MULTIPLE to be set in MPI_Init_thread() and Net to not receive MPI messages with tag 32767\n";
      Galois::StatTimer StatTimer_syncPush("SIMULATE_MPI_SYNC_PUSH");
      Galois::Statistic SyncPush_send_bytes("SIMULATE_MPI_SYNC_PUSH_SEND_BYTES");

#ifndef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      StatTimer_syncPush.start();
      auto& net = Galois::Runtime::getSystemNetworkInterface();

      std::vector<MPI_Request> requests(2 * net.Num);
      unsigned num_requests = 0;

      Galois::Runtime::SendBuffer sb[net.Num];
      for (unsigned x = 0; x < net.Num; ++x) {
         uint32_t num = slaveNodes[x].size();
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
           Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n){
                uint32_t lid = slaveNodes[x][n];
#ifdef __GALOIS_HET_OPENCL__
                CLNodeDataWrapper d = clGraph.getDataW(lid);
                auto val = FnTy::extract(lid, getData(lid, d));
                FnTy::reset(lid, d);
#else
                auto val = FnTy::extract(lid, getData(lid));
                FnTy::reset(lid, getData(lid));
#endif
                val_vec[n] = val;
               }, Galois::loopname("SYNC_PUSH_EXTRACT"), Galois::numrun(get_run_identifier()));
         }
#else
         val_vec[0] = 1;
#endif

         Galois::Runtime::gSerialize(sb[x], val_vec);
         assert(size == sb[x].size());

         SyncPush_send_bytes += size;
         //std::cerr << "[" << id << "]" << " mpi send to " << x << " : " << size << "\n";
         MPI_Isend(sb[x].linearData(), size, MPI_BYTE, x, 32767, MPI_COMM_WORLD, &requests[num_requests++]);
      }

      Galois::Runtime::RecvBuffer rb[net.Num];
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
         Galois::Runtime::gDeserialize(rb[x], val_vec);
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
         if (!FnTy::reduce_batch(x, &val_vec[0])) {
           Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num),
               [&](uint32_t n){
               uint32_t lid = masterNodes[x][n];
#ifdef __GALOIS_HET_OPENCL__
           CLNodeDataWrapper d = clGraph.getDataW(lid);
           FnTy::reduce(lid, d, val_vec[n]);
#else
           FnTy::reduce(lid, getData(lid), val_vec[n]);
#endif
               }, Galois::loopname("SYNC_PUSH_SET"), Galois::numrun(get_run_identifier()));
         }
#endif
      }
      
      //std::cerr << "[" << id << "]" << "push mpi done\n";
      StatTimer_syncPush.stop();
   }
#endif
#endif

#ifdef __GALOIS_SIMULATE_COMMUNICATION__
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
   template<typename FnTy>
#endif
   void syncRecvApplyPull(Galois::Runtime::RecvBuffer& buf) {
     unsigned from_id;
     uint32_t num;
     std::string loopName;
     uint32_t num_iter_push;
     Galois::Runtime::gDeserialize(buf, from_id, num);
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
     std::vector<typename FnTy::ValTy> val_vec(num);
#else
     std::vector<uint64_t> val_vec(num);
#endif
     Galois::Runtime::gDeserialize(buf, val_vec);
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
     if (!FnTy::setVal_batch(from_id, &val_vec[0])) {
       Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n){
           uint32_t localID = slaveNodes[from_id][n];
#ifdef __GALOIS_HET_OPENCL__
           {
           CLNodeDataWrapper d = clGraph.getDataW(localID);
           FnTy::setVal(localID, d, val_vec[n]);
           }
#else
           FnTy::setVal(localID, getData(localID), val_vec[n]);
#endif
           }, Galois::loopname("SYNC_PULL_SET"), Galois::numrun(get_run_identifier()));
      }
#endif
   }

#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
   template<typename FnTy>
#endif
   void syncRecvApplyPush(Galois::Runtime::RecvBuffer& buf) {
     unsigned from_id;
     uint32_t num;
     std::string loopName;
     uint32_t num_iter_push;
     Galois::Runtime::gDeserialize(buf, from_id, num);
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
     std::vector<typename FnTy::ValTy> val_vec(num);
#else
     std::vector<uint64_t> val_vec(num);
#endif
     Galois::Runtime::gDeserialize(buf, val_vec);
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
     if (!FnTy::reduce_batch(from_id, &val_vec[0])) {
       Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num),
           [&](uint32_t n){
           uint32_t lid = masterNodes[from_id][n];
#ifdef __GALOIS_HET_OPENCL__
       CLNodeDataWrapper d = clGraph.getDataW(lid);
       FnTy::reduce(lid, d, val_vec[n]);
#else
       FnTy::reduce(lid, getData(lid), val_vec[n]);
#endif
           }, Galois::loopname("SYNC_PUSH_SET"), Galois::numrun(get_run_identifier()));
     }
#endif
   }

#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
   template<typename FnTy>
#endif
   void simulate_sync_pull(std::string loopName) {
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
      void (hGraph::*fn)(Galois::Runtime::RecvBuffer&) = &hGraph::syncRecvApplyPull<FnTy>;
#else
      void (hGraph::*fn)(Galois::Runtime::RecvBuffer&) = &hGraph::syncRecvApplyPull;
#endif
      Galois::StatTimer StatTimer_syncPull("SIMULATE_NET_SYNC_PULL");
      Galois::Statistic SyncPull_send_bytes("SIMULATE_NET_SYNC_PULL_SEND_BYTES");

#ifndef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
      Galois::Runtime::getHostBarrier().wait();
#endif
      StatTimer_syncPull.start();
      auto& net = Galois::Runtime::getSystemNetworkInterface();

      for (unsigned x = 0; x < net.Num; ++x) {
         uint32_t num = masterNodes[x].size();
         if((x == id) || (num == 0))
           continue;

         Galois::Runtime::SendBuffer b;
         gSerialize(b, idForSelf(), fn, net.ID, num);

#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
         std::vector<typename FnTy::ValTy> val_vec(num);
#else
         std::vector<uint64_t> val_vec(num);
#endif

#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
         if (!FnTy::extract_batch(x, &val_vec[0])) {
           Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n){
               uint32_t localID = masterNodes[x][n];
#ifdef __GALOIS_HET_OPENCL__
               auto val = FnTy::extract((localID), clGraph.getDataR((localID)));
#else
               auto val = FnTy::extract((localID), getData(localID));
#endif
               val_vec[n] = val;

               }, Galois::loopname("SYNC_PULL_EXTRACT"), Galois::numrun(get_run_identifier()));
         }
#else
         val_vec[0] = 1;
#endif

         gSerialize(b, val_vec);

         SyncPull_send_bytes += b.size();
         net.sendMsg(x, syncRecv, b);
      }
      //Will force all messages to be processed before continuing
      net.flush();

      Galois::Runtime::getHostBarrier().wait();
      StatTimer_syncPull.stop();
   }

#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
   template<typename FnTy>
#endif
   void simulate_sync_push(std::string loopName) {
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
      void (hGraph::*fn)(Galois::Runtime::RecvBuffer&) = &hGraph::syncRecvApplyPush<FnTy>;
#else
      void (hGraph::*fn)(Galois::Runtime::RecvBuffer&) = &hGraph::syncRecvApplyPush;
#endif
      Galois::StatTimer StatTimer_syncPush("SIMULATE_NET_SYNC_PUSH");
      Galois::Statistic SyncPush_send_bytes("SIMULATE_NET_SYNC_PUSH_SEND_BYTES");

#ifndef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
      Galois::Runtime::getHostBarrier().wait();
#endif
      StatTimer_syncPush.start();
      auto& net = Galois::Runtime::getSystemNetworkInterface();

      for (unsigned x = 0; x < net.Num; ++x) {
         uint32_t num = slaveNodes[x].size();
         if((x == id) || (num == 0))
           continue;

         Galois::Runtime::SendBuffer b;
         gSerialize(b, idForSelf(), fn, net.ID, num);

#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
         std::vector<typename FnTy::ValTy> val_vec(num);
#else
         std::vector<uint64_t> val_vec(num);
#endif

#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
         if (!FnTy::extract_reset_batch(x, &val_vec[0])) {
           Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n){
                uint32_t lid = slaveNodes[x][n];
#ifdef __GALOIS_HET_OPENCL__
                CLNodeDataWrapper d = clGraph.getDataW(lid);
                auto val = FnTy::extract(lid, getData(lid, d));
                FnTy::reset(lid, d);
#else
                auto val = FnTy::extract(lid, getData(lid));
                FnTy::reset(lid, getData(lid));
#endif
                val_vec[n] = val;
               }, Galois::loopname("SYNC_PUSH_EXTRACT"), Galois::numrun(get_run_identifier()));
         }
#else
         val_vec[0] = 1;
#endif

         gSerialize(b, val_vec);

         SyncPush_send_bytes += b.size();
         net.sendMsg(x, syncRecv, b);
      }
      //Will force all messages to be processed before continuing
      net.flush();

      Galois::Runtime::getHostBarrier().wait();

      StatTimer_syncPush.stop();
   }
#endif


   template<typename FnTy>
   void sync_push_isend(std::string loopName) {
#ifdef __GALOIS_SIMULATE_COMMUNICATION__
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
      if (comm_mode == 1) {
        simulate_sync_push<FnTy>(loopName);
        return;
      } else if (comm_mode == 2) {
        simulate_bare_mpi_sync_push<FnTy>(loopName);
        return;
      }
#endif
#endif
      ++num_iter_push;
      std::string extract_timer_str("SYNC_PUSH_EXTRACT_" + loopName +"_" + get_run_identifier());
      std::string timer_str("SYNC_PUSH_" + loopName + "_" + get_run_identifier());
      std::string timer_barrier_str("SYNC_PUSH_BARRIER_" + loopName + "_" + get_run_identifier());
      std::string statSendBytes_str("SEND_BYTES_SYNC_PUSH_" + loopName + "_" + get_run_identifier());
      std::string doall_str("LAMBDA::SYNC_PUSH_" + loopName + "_" + get_run_identifier());

      Galois::Statistic SyncPush_send_bytes(statSendBytes_str);
      Galois::StatTimer StatTimer_syncPush(timer_str.c_str());
      Galois::StatTimer StatTimerBarrier_syncPush(timer_barrier_str.c_str());
      Galois::StatTimer StatTimer_extract(extract_timer_str.c_str());
      auto& net = Galois::Runtime::getSystemNetworkInterface();


      /***************
       * Extra timers for communication pipeline
       **************/

      Galois::StatTimer StatTimer_SendTime(std::string("SYNC_PUSH_SEND_" +  loopName + get_run_identifier()).c_str());
      Galois::StatTimer StatTimer_RecvTime(std::string("SYNC_PUSH_RECV_" +  loopName + "_" + get_run_identifier()).c_str());

 

      StatTimer_SendTime.start();

      for (unsigned x = 0; x < net.Num; ++x) {
        uint32_t num = slaveNodes[x].size();
        if((x == id))
          continue;

        Galois::Runtime::SendBuffer b;

        StatTimer_extract.start();
        if(num > 0 ){
          Galois::DynamicBitSet bit_set_comm;
          bit_set_comm.resize_init(num);

          std::string doall_str2("LAMBDA::SYNC_PUSH_BIT_SET" + loopName + "_" + get_run_identifier());

          if(bit_set_compute.bit_count() > 0){
            Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n ){
                uint32_t lid = slaveNodes[x][n];
                if(bit_set_compute.is_set(lid)){
                bit_set_comm.bit_set(n);
                }

                }, Galois::loopname(doall_str2.c_str()), Galois::numrun(get_run_identifier()));
          }

          //bits set to be sent to x
          auto bits_set_num = bit_set_comm.bit_count(num);
          //std::cout << "BIT_COMM : " << bits_set_num << " , BITS_COMP : " << bit_set_compute.bit_count()  << " num : " << num << "\n";
          Galois::Runtime::reportStat("(NULL)", "SYNC_PUSH_BITS_SAVED", (unsigned)(bits_set_num - num), 0);
          std::vector<typename FnTy::ValTy> val_vec(bits_set_num);

          if(bits_set_num > 0){

            if (!FnTy::extract_reset_batch(x, &val_vec[0])) {
              Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n){

                  if(bit_set_comm.is_set(n)){
                    uint32_t lid = slaveNodes[x][n];
                    uint32_t vec_index = bit_set_comm.bit_count(n);

#ifdef __GALOIS_HET_OPENCL__
                  CLNodeDataWrapper d = clGraph.getDataW(lid);
                  auto val = FnTy::extract(lid, getData(lid, d));
                  FnTy::reset(lid, d);
#else
                  auto val = FnTy::extract(lid, getData(lid));
                  FnTy::reset(lid, getData(lid));
#endif
                  val_vec[vec_index] = val;
                  }
                 }, Galois::loopname(doall_str.c_str()), Galois::numrun(get_run_identifier()));
           }
          }

           gSerialize(b, bit_set_comm, val_vec);
           bit_set_compute.reset();
         } else {
           gSerialize(b, loopName);
         }
         StatTimer_extract.stop();

         SyncPush_send_bytes += b.size();
         net.sendTagged(x, Galois::Runtime::evilPhase, b);
      }

      //Will force all messages to be processed before continuing
      net.flush();
      StatTimer_SendTime.stop();
   }


   template<typename FnTy>
   void sync_push_recv(std::string loopName) {
#ifdef __GALOIS_SIMULATE_COMMUNICATION__
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
      if (comm_mode == 1) {
        simulate_sync_push<FnTy>(loopName);
        return;
      } else if (comm_mode == 2) {
        simulate_bare_mpi_sync_push<FnTy>(loopName);
        return;
      }
#endif
#endif
      ++num_iter_push;
      std::string extract_timer_str("SYNC_PUSH_EXTRACT_" + loopName +"_" + get_run_identifier());
      std::string timer_str("SYNC_PUSH_" + loopName + "_" + get_run_identifier());
      std::string timer_barrier_str("SYNC_PUSH_BARRIER_" + loopName + "_" + get_run_identifier());
      std::string statSendBytes_str("SEND_BYTES_SYNC_PUSH_" + loopName + "_" + get_run_identifier());
      std::string doall_str("LAMBDA::SYNC_PUSH_" + loopName + "_" + get_run_identifier());

      Galois::Statistic SyncPush_send_bytes(statSendBytes_str);
      Galois::StatTimer StatTimer_syncPush(timer_str.c_str());
      Galois::StatTimer StatTimerBarrier_syncPush(timer_barrier_str.c_str());
      Galois::StatTimer StatTimer_extract(extract_timer_str.c_str());
      auto& net = Galois::Runtime::getSystemNetworkInterface();


      /***************
       * Extra timers for communication pipeline
       **************/

      Galois::StatTimer StatTimer_RecvTime(std::string("SYNC_PUSH_RECV_" +  loopName + "_" + get_run_identifier()).c_str());

   //receive
    StatTimer_RecvTime.start();
      for (unsigned x = 0; x < net.Num; ++x) {
        if ((x == id))
          continue;
        decltype(net.recieveTagged(Galois::Runtime::evilPhase,nullptr)) p;
        do {
          net.handleReceives();
          p = net.recieveTagged(Galois::Runtime::evilPhase, nullptr);
        } while (!p);
        syncRecvApply<FnTy>(p->first, p->second, loopName);
      }
      ++Galois::Runtime::evilPhase;

    StatTimer_RecvTime.stop();

   }

   template<typename FnTy>
   void sync_push(std::string loopName) {
#ifdef __GALOIS_SIMULATE_COMMUNICATION__
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
      if (comm_mode == 1) {
        simulate_sync_push<FnTy>(loopName);
        return;
      } else if (comm_mode == 2) {
        simulate_bare_mpi_sync_push<FnTy>(loopName);
        return;
      }
#endif
#endif
      ++num_iter_push;
      std::string extract_timer_str("SYNC_PUSH_EXTRACT_" + loopName +"_" + get_run_identifier());
      std::string timer_str("SYNC_PUSH_" + loopName + "_" + get_run_identifier());
      std::string timer_barrier_str("SYNC_PUSH_BARRIER_" + loopName + "_" + get_run_identifier());;
      std::string statSendBytes_str("SEND_BYTES_SYNC_PUSH_" + loopName + "_" + get_run_identifier());
      std::string doall_str("LAMBDA::SYNC_PUSH_" + loopName + "_" + get_run_identifier());

      Galois::Statistic SyncPush_send_bytes(statSendBytes_str);
      Galois::StatTimer StatTimer_syncPush(timer_str.c_str());
      Galois::StatTimer StatTimerBarrier_syncPush(timer_barrier_str.c_str());
      Galois::StatTimer StatTimer_extract(extract_timer_str.c_str());
      auto& net = Galois::Runtime::getSystemNetworkInterface();


      /***************
       * Extra timers for communication pipeline
       **************/

      Galois::StatTimer StatTimer_SendTime(std::string("SYNC_PUSH_SEND_" +  loopName + "_" + get_run_identifier()).c_str());
      Galois::StatTimer StatTimer_RecvTime(std::string("SYNC_PUSH_RECV_" +  loopName + "_" + get_run_identifier()).c_str());

 
      StatTimer_syncPush.start();

      StatTimer_SendTime.start();

      for (unsigned h = 1; h < net.Num; ++h) {
         unsigned x = (id + h) % net.Num;
         uint32_t num = slaveNodes[x].size();

         Galois::Runtime::SendBuffer b;

         StatTimer_extract.start();
         if(num > 0){
           Galois::DynamicBitSet bit_set_comm;
           bit_set_comm.resize_init(num);

            std::string doall_str2("LAMBDA::SYNC_PUSH_BIT_SET" + loopName + "_" + get_run_identifier());

            if(bit_set_compute.bit_count() > 0){
               Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n ){
                    uint32_t lid = slaveNodes[x][n];
                    if(bit_set_compute.is_set(lid)){
                      bit_set_comm.bit_set(n);
                    }

                   }, Galois::loopname(doall_str2.c_str()), Galois::numrun(get_run_identifier()));
            }

           //bits set to be sent to x
           auto bits_set_num = bit_set_comm.bit_count(num);
           //std::cout << "BIT_COMM : " << bits_set_num << " , BITS_COMP : " << bit_set_compute.bit_count()  << " num : " << num << "\n";
           Galois::Runtime::reportStat("(NULL)", "SYNC_PUSH_BITS_SAVED", (unsigned)(bits_set_num - num), 0);
           std::vector<typename FnTy::ValTy> val_vec(bits_set_num);

           if(bits_set_num > 0){
             if (!FnTy::extract_reset_batch(x, &val_vec[0])) {
               Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n){

                    if(bit_set_comm.is_set(n)){
                      uint32_t lid = slaveNodes[x][n];
                      uint32_t vec_index = bit_set_comm.bit_count(n);
#ifdef __GALOIS_HET_OPENCL__
                      CLNodeDataWrapper d = clGraph.getDataW(lid);
                      auto val = FnTy::extract(lid, getData(lid, d));
                      FnTy::reset(lid, d);
#else
                      auto val = FnTy::extract(lid, getData(lid));
                      FnTy::reset(lid, getData(lid));
#endif
                      val_vec[vec_index] = val;
                    }
                   }, Galois::loopname(doall_str.c_str()), Galois::numrun(get_run_identifier()));
             }
           }

           gSerialize(b, bit_set_comm, val_vec);
           bit_set_compute.reset();
         } else {
           gSerialize(b, loopName);
         }
         StatTimer_extract.stop();

         SyncPush_send_bytes += b.size();
         net.sendTagged(x, Galois::Runtime::evilPhase, b);
      }
      StatTimer_SendTime.stop();

      //Will force all messages to be processed before continuing
      net.flush();

      //receive
    StatTimer_RecvTime.start();
      for (unsigned x = 0; x < net.Num; ++x) {
        if ((x == id))
          continue;
        decltype(net.recieveTagged(Galois::Runtime::evilPhase,nullptr)) p;
        do {
          net.handleReceives();
          p = net.recieveTagged(Galois::Runtime::evilPhase, nullptr);
        } while (!p);
        syncRecvApply<FnTy>(p->first, p->second, loopName);
      }
    StatTimer_RecvTime.stop();
      ++Galois::Runtime::evilPhase;

      StatTimer_syncPush.stop();

   }

   template<typename FnTy>
   void sync_push_ck(std::string loopName) {
#ifdef __GALOIS_SIMULATE_COMMUNICATION__
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
      if (comm_mode == 1) {
        simulate_sync_push<FnTy>(loopName);
        return;
      } else if (comm_mode == 2) {
        simulate_bare_mpi_sync_push<FnTy>(loopName);
        return;
      }
#endif
#endif
      ++num_iter_push;
      std::string extract_timer_str("SYNC_PUSH_EXTRACT_" + loopName +"_" + get_run_identifier());
      std::string timer_str("SYNC_PUSH_" + loopName + "_" + get_run_identifier());
      std::string timer_barrier_str("SYNC_PUSH_BARRIER_" + loopName + "_" + get_run_identifier());
      std::string statSendBytes_str("SEND_BYTES_SYNC_PUSH_" + loopName + "_" + get_run_identifier());
      std::string doall_str("LAMBDA::SYNC_PUSH_" + loopName + "_" + get_run_identifier());
      Galois::Statistic SyncPush_send_bytes(statSendBytes_str);
      Galois::StatTimer StatTimer_syncPush(timer_str.c_str());
      Galois::StatTimer StatTimerBarrier_syncPush(timer_barrier_str.c_str());
      Galois::StatTimer StatTimer_extract(extract_timer_str.c_str());

      std::string statChkPtBytes_str("CHECKPOINT_BYTES_SYNC_PUSH_" + loopName +"_" + get_run_identifier());
      Galois::Statistic checkpoint_bytes(statChkPtBytes_str);

      std::string checkpoint_timer_str("TIME_CHECKPOINT_SYNC_PUSH_MEM_" + get_run_identifier());
      Galois::StatTimer StatTimer_checkpoint(checkpoint_timer_str.c_str());


      StatTimer_syncPush.start();
      auto& net = Galois::Runtime::getSystemNetworkInterface();

      for (unsigned h = 1; h < net.Num; ++h) {
         unsigned x = (id + h) % net.Num;
         uint32_t num = slaveNodes[x].size();

         Galois::Runtime::SendBuffer b;

         StatTimer_extract.start();
           std::vector<typename FnTy::ValTy> val_vec(num);

         if(num > 0 ){
           if (!FnTy::extract_reset_batch(x, &val_vec[0])) {
             Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n){
                  uint32_t lid = slaveNodes[x][n];
#ifdef __GALOIS_HET_OPENCL__
                  CLNodeDataWrapper d = clGraph.getDataW(lid);
                  auto val = FnTy::extract(lid, getData(lid, d));
                  FnTy::reset(lid, d);
#else
                  auto val = FnTy::extract(lid, getData(lid));
                  FnTy::reset(lid, getData(lid));
#endif
                  val_vec[n] = val;
                 }, Galois::loopname(doall_str.c_str()), Galois::numrun(get_run_identifier()));
           }

        }

           gSerialize(b, val_vec);
      /*   }
           else {
           gSerialize(b, loopName);
         }
         */


      SyncPush_send_bytes += b.size();
      auto send_bytes = b.size();

      StatTimer_checkpoint.start();
         if(x == (net.ID + 1)%net.Num){
           //checkpoint owned nodes.
           std::vector<typename FnTy::ValTy> checkpoint_val_vec(numOwned);
           Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(numOwned), [&](uint32_t n) {

               auto val = FnTy::extract(n, getData(n));
               checkpoint_val_vec[n] = val;
               }, Galois::loopname(doall_str.c_str()), Galois::numrun(get_run_identifier()));
         gSerialize(b, checkpoint_val_vec);
         checkpoint_bytes += (b.size() - send_bytes);

         }
      StatTimer_checkpoint.stop();

         StatTimer_extract.stop();

         net.sendTagged(x, Galois::Runtime::evilPhase, b);
      }
      //Will force all messages to be processed before continuing
      net.flush();

      //receive
      for (unsigned x = 0; x < net.Num; ++x) {
        if ((x == id))
          continue;
        decltype(net.recieveTagged(Galois::Runtime::evilPhase,nullptr)) p;
        do {
          net.handleReceives();
          p = net.recieveTagged(Galois::Runtime::evilPhase, nullptr);
        } while (!p);
        syncRecvApply_ck<FnTy>(p->first, p->second, loopName);
      }
      ++Galois::Runtime::evilPhase;

      StatTimer_syncPush.stop();

   }

  /****************************
   * Communication Pipelining
   ***************************/
    template<typename FnTy>
   void sync_pull_isend(std::string loopName) {
#ifdef __GALOIS_SIMULATE_COMMUNICATION__
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
      if (comm_mode == 1) {
        simulate_sync_pull<FnTy>(loopName);
        return;
      } else if (comm_mode == 2) {
        simulate_bare_mpi_sync_pull<FnTy>(loopName);
        return;
      }
#endif
#endif
      ++num_iter_pull;
      std::string doall_str("LAMBDA::SYNC_PULL_iSEND_" + loopName + "_" + get_run_identifier());
      std::string timer_str("SYNC_PULL_iSEND_" + loopName +"_" + get_run_identifier());
      std::string statSendBytes_str("SEND_BYTES_SYNC_PULL_iSEND_" + loopName +"_" + get_run_identifier());
      Galois::Statistic SyncPull_send_bytes(statSendBytes_str);
      Galois::StatTimer StatTimer_syncPull(timer_str.c_str());
      auto& net = Galois::Runtime::getSystemNetworkInterface();


      /***************
       * Extra timers for communication pipeline
       **************/

      Galois::StatTimer StatTimer_SendTime(std::string("SYNC_PULL_iSEND_" +  loopName + "_" + get_run_identifier()).c_str());


    StatTimer_SendTime.start();
      for (unsigned x = 0; x < net.Num; ++x) {
        uint32_t num = masterNodes[x].size();
        if((x == id))
          continue;

        Galois::Runtime::SendBuffer b;

        if(num > 0 ){
          Galois::DynamicBitSet bit_set_comm;
          bit_set_comm.resize_init(num);

          std::string doall_str2("LAMBDA::SYNC_PULL_BIT_SET" + loopName + "_" + get_run_identifier());

          Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n ){
              uint32_t lid = masterNodes[x][n];
              if(bit_set_compute.is_set(lid))
              bit_set_comm.bit_set(n);

              }, Galois::loopname(doall_str2.c_str()), Galois::numrun(get_run_identifier()));


          //bits set to be sent to x
          auto bits_set_num = bit_set_comm.bit_count(num);

          //std::cout << "BIT_COMM : " << bits_set_num << " , BITS_COMP : " << bit_set_compute.bit_count()  << " num : " << num << "\n";
          Galois::Runtime::reportStat("(NULL)", "SYNC_PULL_BITS_SAVED", (unsigned)(bits_set_num - num), 0);

          std::vector<typename FnTy::ValTy> val_vec(bits_set_num);

          if(bits_set_num > 0){

            if (!FnTy::extract_batch(x, &val_vec[0])) {

              Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n){

                  if(bit_set_comm.is_set(n)){
                  uint32_t localID = masterNodes[x][n];
                  uint32_t vec_index = bit_set_comm.bit_count(n);

#ifdef __GALOIS_HET_OPENCL__
                  auto val = FnTy::extract((localID), clGraph.getDataR(localID));
#else
                  auto val = FnTy::extract((localID), getData(localID));
#endif
                  val_vec[vec_index] = val;
                  }
                  }, Galois::loopname(doall_str.c_str()), Galois::numrun(get_run_identifier()));
            }
          }
          gSerialize(b, bit_set_comm, val_vec);
          bit_set_compute.reset();
        } else {
          gSerialize(b, loopName);
        }

        SyncPull_send_bytes += b.size();
        net.sendTagged(x, Galois::Runtime::evilPhase, b);

      }

      net.flush();

    StatTimer_SendTime.stop();

   }

   template<typename FnTy>
   void sync_pull_recv(std::string loopName) {
#ifdef __GALOIS_SIMULATE_COMMUNICATION__
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
      if (comm_mode == 1) {
        simulate_sync_pull<FnTy>(loopName);
        return;
      } else if (comm_mode == 2) {
        simulate_bare_mpi_sync_pull<FnTy>(loopName);
        return;
      }
#endif
#endif
      ++num_iter_pull;
      auto& net = Galois::Runtime::getSystemNetworkInterface();


      /***************
       * Extra timers for communication pipeline
       **************/

      Galois::StatTimer StatTimer_RecvTime(std::string("SYNC_PULL_RECV_" +  loopName + "_" + get_run_identifier()).c_str());

       //receive
    StatTimer_RecvTime.start();
      for (unsigned x = 0; x < net.Num; ++x) {
        if ((x == id))
          continue;
        decltype(net.recieveTagged(Galois::Runtime::evilPhase,nullptr)) p;
        do {
          net.handleReceives();
          p = net.recieveTagged(Galois::Runtime::evilPhase, nullptr);
        } while (!p);
        syncPullRecvApply<FnTy>(p->first, p->second, loopName);
      }

      ++Galois::Runtime::evilPhase;
    StatTimer_RecvTime.stop();

   }

   template<typename FnTy>
   void sync_pull(std::string loopName) {
#ifdef __GALOIS_SIMULATE_COMMUNICATION__
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
      if (comm_mode == 1) {
        simulate_sync_pull<FnTy>(loopName);
        return;
      } else if (comm_mode == 2) {
        simulate_bare_mpi_sync_pull<FnTy>(loopName);
        return;
      }
#endif
#endif
      ++num_iter_pull;
      std::string doall_str("LAMBDA::SYNC_PULL_" + loopName + "_" + get_run_identifier());
      std::string timer_str("SYNC_PULL_" + loopName +"_" + get_run_identifier());
      std::string statSendBytes_str("SEND_BYTES_SYNC_PULL_" + loopName +"_" + get_run_identifier());
      Galois::Statistic SyncPull_send_bytes(statSendBytes_str);
      Galois::StatTimer StatTimer_syncPull(timer_str.c_str());
      auto& net = Galois::Runtime::getSystemNetworkInterface();


      /***************
       * Extra timers for communication pipeline
       **************/

      Galois::StatTimer StatTimer_SendTime(std::string("SYNC_PULL_SEND_" +  loopName + "_" + get_run_identifier()).c_str());
      Galois::StatTimer StatTimer_RecvTime(std::string("SYNC_PULL_RECV_" +  loopName + "_" + get_run_identifier()).c_str());

      StatTimer_syncPull.start();


    StatTimer_SendTime.start();
      for (unsigned h = 1; h < net.Num; ++h) {
        unsigned x = (id + h) % net.Num;
        uint32_t num = masterNodes[x].size();

        Galois::Runtime::SendBuffer b;

        if(num > 0 ){
           Galois::DynamicBitSet bit_set_comm;
           bit_set_comm.resize_init(num);

            std::string doall_str2("LAMBDA::SYNC_PULL_BIT_SET" + loopName + "_" + get_run_identifier());

           Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n ){
                uint32_t lid = masterNodes[x][n];
                if(bit_set_compute.is_set(lid))
                  bit_set_comm.bit_set(n);

               }, Galois::loopname(doall_str2.c_str()), Galois::numrun(get_run_identifier()));


           //bits set to be sent to x
           auto bits_set_num = bit_set_comm.bit_count(num);

           //std::cout << "BIT_COMM : " << bits_set_num << " , BITS_COMP : " << bit_set_compute.bit_count()  << " num : " << num << "\n";
           Galois::Runtime::reportStat("(NULL)", "SYNC_PULL_BITS_SAVED", (unsigned)(bits_set_num - num), 0);

           std::vector<typename FnTy::ValTy> val_vec(bits_set_num);

           if(bits_set_num > 0){


             if (!FnTy::extract_batch(x, &val_vec[0])) {

               Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n){

                   if(bit_set_comm.is_set(n)){
                     uint32_t localID = masterNodes[x][n];
                     uint32_t vec_index = bit_set_comm.bit_count(n);

#ifdef __GALOIS_HET_OPENCL__
                     auto val = FnTy::extract((localID), clGraph.getDataR(localID));
#else
                     auto val = FnTy::extract((localID), getData(localID));
#endif
                     val_vec[vec_index] = val;
                   }
                   }, Galois::loopname(doall_str.c_str()), Galois::numrun(get_run_identifier()));
             }
           }
           gSerialize(b, bit_set_comm, val_vec);
           bit_set_compute.reset();
        } else {
          gSerialize(b, loopName);
        }

        SyncPull_send_bytes += b.size();
        net.sendTagged(x, Galois::Runtime::evilPhase, b);
      }

    StatTimer_SendTime.stop();

      net.flush();

      //receive
    StatTimer_RecvTime.start();
      for (unsigned x = 0; x < net.Num; ++x) {
        if ((x == id))
          continue;
        decltype(net.recieveTagged(Galois::Runtime::evilPhase,nullptr)) p;
        do {
          net.handleReceives();
          p = net.recieveTagged(Galois::Runtime::evilPhase, nullptr);
        } while (!p);
        syncPullRecvApply<FnTy>(p->first, p->second, loopName);
      }
    StatTimer_RecvTime.stop();

      ++Galois::Runtime::evilPhase;
      StatTimer_syncPull.stop();
   }


 /****************************************
  * Fault Tolerance
  * 1. CheckPointing
  ***************************************/
  template <typename FnTy>
  void checkpoint(std::string loopName) {
    auto& net = Galois::Runtime::getSystemNetworkInterface();
    std::string doall_str("LAMBDA::CHECKPOINT_" + loopName + "_" + get_run_identifier());
    std::string checkpoint_timer_str("TIME_CHECKPOINT_TOTAL_" + get_run_identifier());
    Galois::StatTimer StatTimer_checkpoint(checkpoint_timer_str.c_str());
    StatTimer_checkpoint.start();


    std::string statChkPtBytes_str("CHECKPOINT_BYTES_" + loopName +"_" + get_run_identifier());
    Galois::Statistic checkpoint_bytes(statChkPtBytes_str);
    //checkpoint owned nodes.
    std::vector<typename FnTy::ValTy> val_vec(numOwned);
    Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(numOwned), [&](uint32_t n) {

          auto val = FnTy::extract(n, getData(n));
          val_vec[n] = val;
        }, Galois::loopname(doall_str.c_str()), Galois::numrun(get_run_identifier()));

#if 0
    //Write val_vec to disk.
      for(auto k = 0; k < 10; ++k){
        std::cout << "BEFORE : val_vec[" << k <<"] :" << val_vec[k] << "\n";
      }
#endif

    checkpoint_bytes += val_vec.size() * sizeof(typename FnTy::ValTy);

    std::string chkPt_fileName = "/scratch/02982/ggill0/Checkpoint_" + loopName + "_" + FnTy::field_name() + "_" + std::to_string(net.ID);
    //std::string chkPt_fileName = "Checkpoint_" + loopName + "_" + FnTy::field_name() + "_" + std::to_string(net.ID);
    std::ofstream chkPt_file(chkPt_fileName, std::ios::out | std::ofstream::binary);
    chkPt_file.seekp(0);
    chkPt_file.write(reinterpret_cast<char*>(&val_vec[0]), val_vec.size()*sizeof(uint32_t));
    chkPt_file.close();
    StatTimer_checkpoint.stop();
  }

  template<typename FnTy>
    void checkpoint_apply(std::string loopName){
      auto& net = Galois::Runtime::getSystemNetworkInterface();
      std::string doall_str("LAMBDA::CHECKPOINT_APPLY_" + loopName + "_" + get_run_identifier());
      //checkpoint owned nodes.
      std::vector<typename FnTy::ValTy> val_vec(numOwned);
      //read val_vec from disk.
      std::string chkPt_fileName = "/scratch/02982/ggill0/Checkpoint_" + loopName + "_" + FnTy::field_name() + "_" + std::to_string(net.ID);
      //std::string chkPt_fileName = "Checkpoint_" + loopName + "_" + FnTy::field_name() + "_" + std::to_string(net.ID);
      std::ifstream chkPt_file(chkPt_fileName, std::ios::in | std::ofstream::binary);
      if(!chkPt_file.is_open()){
        std::cout << "Unable to open checkpoint file " << chkPt_fileName << " ! Exiting!\n";
        exit(1);
      }
      chkPt_file.read(reinterpret_cast<char*>(&val_vec[0]), numOwned*sizeof(uint32_t));

      for(auto k = 0; k < 10; ++k){
        std::cout << "AFTER : val_vec[" << k << "] : " << val_vec[k] << "\n";
      }

      Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(numOwned), [&](uint32_t n) {

          FnTy::setVal(n, getData(n), val_vec[n]);
          }, Galois::loopname(doall_str.c_str()), Galois::numrun(get_run_identifier()));
    }

 /*************************************************
  * Fault Tolerance
  * 1. CheckPointing in the memory of another node
  ************************************************/
  template<typename FnTy>
  void saveCheckPoint(Galois::Runtime::RecvBuffer& b){
    checkpoint_recvBuffer = std::move(b);
  }

  template<typename FnTy>
  void checkpoint_mem(std::string loopName) {
    auto& net = Galois::Runtime::getSystemNetworkInterface();
    std::string doall_str("LAMBDA::CHECKPOINT_MEM_" + loopName + "_" + get_run_identifier());

    std::string statChkPtBytes_str("CHECKPOINT_BYTES_" + loopName +"_" + get_run_identifier());
    Galois::Statistic checkpoint_bytes(statChkPtBytes_str);

    std::string checkpoint_timer_str("TIME_CHECKPOINT_TOTAL_MEM_" + get_run_identifier());
    Galois::StatTimer StatTimer_checkpoint(checkpoint_timer_str.c_str());

    std::string checkpoint_timer_send_str("TIME_CHECKPOINT_TOTAL_MEM_SEND_" + get_run_identifier());
    Galois::StatTimer StatTimer_checkpoint_send(checkpoint_timer_send_str.c_str());

    std::string checkpoint_timer_recv_str("TIME_CHECKPOINT_TOTAL_MEM_recv_" + get_run_identifier());
    Galois::StatTimer StatTimer_checkpoint_recv(checkpoint_timer_recv_str.c_str());

    StatTimer_checkpoint.start();

    StatTimer_checkpoint_send.start();
    //checkpoint owned nodes.
    std::vector<typename FnTy::ValTy> val_vec(numOwned);
    Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(numOwned), [&](uint32_t n) {

          auto val = FnTy::extract(n, getData(n));
          val_vec[n] = val;
        }, Galois::loopname(doall_str.c_str()), Galois::numrun(get_run_identifier()));

    Galois::Runtime::SendBuffer b;
    gSerialize(b, val_vec);

#if 0
    if(net.ID == 0 )
      for(auto k = 0; k < 10; ++k){
        std::cout << "before : val_vec[" << k << "] : " << val_vec[k] << "\n";
      }
#endif

    checkpoint_bytes += b.size();
    //send to your neighbor on your left.
    net.sendTagged((net.ID + 1)%net.Num, Galois::Runtime::evilPhase, b);

    StatTimer_checkpoint_send.stop();

    net.flush();

    StatTimer_checkpoint_recv.start();
    //receiving the checkpointed data.
    decltype(net.recieveTagged(Galois::Runtime::evilPhase,nullptr)) p;
    do {
      net.handleReceives();
      p = net.recieveTagged(Galois::Runtime::evilPhase, nullptr);
    } while (!p);
    checkpoint_recvBuffer = std::move(p->second);

    std::cerr << net.ID << " recvBuffer SIZE ::::: " << checkpoint_recvBuffer.size() << "\n";

    ++Galois::Runtime::evilPhase;
    StatTimer_checkpoint_recv.stop();

    StatTimer_checkpoint.stop();
  }

    template<typename FnTy>
    void checkpoint_mem_apply(Galois::Runtime::RecvBuffer& b){
      auto& net = Galois::Runtime::getSystemNetworkInterface();
      std::string doall_str("LAMBDA::CHECKPOINT_MEM_APPLY_" + get_run_identifier());

      std::string checkpoint_timer_str("TIME_CHECKPOINT_MEM_APPLY" + get_run_identifier());
      Galois::StatTimer StatTimer_checkpoint(checkpoint_timer_str.c_str());
      StatTimer_checkpoint.start();

      uint32_t from_id;
      Galois::Runtime::RecvBuffer recv_checkpoint_buf;
      gDeserialize(b, from_id);
      recv_checkpoint_buf = std::move(b);
      std::cerr << net.ID << " : " << recv_checkpoint_buf.size() << "\n";
      //gDeserialize(b, recv_checkpoint_buf);

      std::vector<typename FnTy::ValTy> val_vec(numOwned);
      gDeserialize(recv_checkpoint_buf, val_vec);

    if(net.ID == 0 )
      for(auto k = 0; k < 10; ++k){
        std::cout << "After : val_vec[" << k << "] : " << val_vec[k] << "\n";
      }
      Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(numOwned), [&](uint32_t n) {

          FnTy::setVal(n, getData(n), val_vec[n]);
          }, Galois::loopname(doall_str.c_str()), Galois::numrun(get_run_identifier()));
    }


    template<typename FnTy>
    void recovery_help_landingPad(Galois::Runtime::RecvBuffer& buff){
      void (hGraph::*fn)(Galois::Runtime::RecvBuffer&) = &hGraph::checkpoint_mem_apply<FnTy>;
      auto& net = Galois::Runtime::getSystemNetworkInterface();
      uint32_t from_id;
      std::string help_str;
      gDeserialize(buff, from_id, help_str);

      Galois::Runtime::SendBuffer b;
      gSerialize(b, idForSelf(), fn, net.ID, checkpoint_recvBuffer);
      net.sendMsg(from_id, syncRecv, b);

      //send back the checkpointed nodes for from_id.

    }


    template<typename FnTy>
    void recovery_send_help(std::string loopName){
      void (hGraph::*fn)(Galois::Runtime::RecvBuffer&) = &hGraph::recovery_help_landingPad<FnTy>;
      auto& net = Galois::Runtime::getSystemNetworkInterface();
      Galois::Runtime::SendBuffer b;
      std::string help_str = "recoveryHelp";

      gSerialize(b, idForSelf(), fn, net.ID, help_str);

      //send help message to the host that is keeping checkpoint for you.
      net.sendMsg((net.ID + 1)%net.Num, syncRecv, b);
    }


  /*****************************************************/

 /****************************************
  * Fault Tolerance
  * 1. Zorro
  ***************************************/
#if 0
  void recovery_help_landingPad(Galois::Runtime::RecvBuffer& b){
    uint32_t from_id;
    std::string help_str;
    gDeserialize(b, from_id, help_str);

    //send back the slaveNode for from_id.

  }

  template<typename FnTy>
  void recovery_send_help(std::string loopName){
    void (hGraph::*fn)(Galois::Runtime::RecvBuffer&) = &hGraph::recovery_help<FnTy>;
    auto& net = Galois::Runtime::getSystemNetworkInterface();
    Galois::Runtime::SendBuffer b;
    std::string help_str = "recoveryHelp";

    gSerialize(b, idForSelf(), help_str);

    for(auto i = 0; i < net.Num; ++i){
      net.sendMsg(i, syncRecv, b);
    }
  }
#endif


  /*************************************/


   uint64_t getGID(uint32_t nodeID) const {
      return L2G(nodeID);
   }
   uint32_t getLID(uint64_t nodeID) const {
      return G2L(nodeID);
   }
#if 0
   unsigned getHostID(uint64_t gid) {
     getHostID(gid);
   }
#endif
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
      m.num_master_nodes = (unsigned int *) calloc(masterNodes.size(), sizeof(unsigned int));;
      m.master_nodes = (unsigned int **) calloc(masterNodes.size(), sizeof(unsigned int *));;
      for(uint32_t h = 0; h < masterNodes.size(); ++h){
        m.num_master_nodes[h] = masterNodes[h].size();
        if (masterNodes[h].size() > 0) {
          m.master_nodes[h] = (unsigned int *) calloc(masterNodes[h].size(), sizeof(unsigned int));;
          std::copy(masterNodes[h].begin(), masterNodes[h].end(), m.master_nodes[h]);
        } else {
          m.master_nodes[h] = NULL;
        }
      }
      m.num_slave_nodes = (unsigned int *) calloc(slaveNodes.size(), sizeof(unsigned int));;
      m.slave_nodes = (unsigned int **) calloc(slaveNodes.size(), sizeof(unsigned int *));;
      for(uint32_t h = 0; h < slaveNodes.size(); ++h){
        m.num_slave_nodes[h] = slaveNodes[h].size();
        if (slaveNodes[h].size() > 0) {
          m.slave_nodes[h] = (unsigned int *) calloc(slaveNodes[h].size(), sizeof(unsigned int));;
          std::copy(slaveNodes[h].begin(), slaveNodes[h].end(), m.slave_nodes[h]);
        } else {
          m.slave_nodes[h] = NULL;
        }
      }

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


  /**For resetting num_iter_pull and push.**/
   void reset_num_iter(uint32_t runNum){
      num_iter_pull = 0;
      num_iter_push = 0;
      num_run = runNum;
   }
   uint32_t get_run_num() {
     return num_run;
   }
   void set_num_iter(uint32_t iteration){
    num_iteration = iteration;
   }

   std::string get_run_identifier(){
    return std::string(std::to_string(num_run) + "_" + std::to_string(num_iteration));
   }
   /** Report stats to be printed.**/
   void reportStats(){
    statGhostNodes.report();
   }

  void bit_set(uint32_t LID){
    bit_set_compute.bit_set(LID);
  }

  void get_bit_set_count(){
     Galois::Runtime::reportStat("(NULL)", "FRAC_BITS_CHANGED_" + std::to_string(num_run) + "_HOST_", bit_set_compute.bit_count() , 0);
  }

};
#endif//_GALOIS_DIST_HGRAPH_H
