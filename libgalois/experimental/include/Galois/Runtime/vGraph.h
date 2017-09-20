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
 * Derived from hGraph. Graph abstraction for vertex cut.
 * @author Rashid Kaleem <rashid.kaleem@gmail.com>
 * @author Gurbinder Gill <gurbinder533@gmail.com>
 *
 */

#include <vector>
#include <set>
#include <algorithm>
#include <unordered_map>

#include "Galois/gstl.h"
#include "Galois/Graphs/LC_CSR_Graph.h"
#include "Galois/Runtime/Substrate.h"
#include "Galois/Runtime/Network.h"

#include "Galois/Runtime/Serialize.h"

#include "Galois/Runtime/Tracer.h"
#include "Galois/Threads.h"

#include "Galois/Runtime/GlobalObj.h"
#include "Galois/Runtime/OfflineGraph.h"

#ifdef __GALOIS_HET_CUDA__
#include "Galois/Runtime/Cuda/cuda_mtypes.h"
#endif

#ifdef __GALOIS_SIMULATE_COMMUNICATION__
#ifdef __GALOIS_SIMULATE_BARE_MPI_COMMUNICATION__
#include "mpi.h"
#endif
#endif

#ifndef _GALOIS_DIST_vGraph_H
#define _GALOIS_DIST_vGraph_H

/** Utilities for reading partitioned graphs. **/
struct NodeInfo {
   NodeInfo() :
         local_id(0), global_id(0), owner_id(0) {
   }
   NodeInfo(size_t l, size_t g, size_t o) :
         local_id(l), global_id(g), owner_id(o) {
   }
   size_t local_id;
   size_t global_id;
   size_t owner_id;
};

std::string getPartitionFileName(const std::string & basename, unsigned hostID, unsigned num_hosts){
   std::string result = basename;
   result+= ".PART.";
   result+=std::to_string(hostID);
   result+= ".OF.";
   result+=std::to_string(num_hosts);
   return result;
}
std::string getMetaFileName(const std::string & basename, unsigned hostID, unsigned num_hosts){
   std::string result = basename;
   result+= ".META.";
   result+=std::to_string(hostID);
   result+= ".OF.";
   result+=std::to_string(num_hosts);
   return result;
}

bool readMetaFile(const std::string& metaFileName, std::vector<NodeInfo>& localToGlobalMap_meta){
  std::ifstream meta_file(metaFileName, std::ifstream::binary);
  if (!meta_file.is_open()) {
    std::cout << "Unable to open file " << metaFileName << "! Exiting!\n";
    return false;
  }
  size_t num_entries;
  meta_file.read(reinterpret_cast<char*>(&num_entries), sizeof(num_entries));
  std::cout << "Partition :: " << " Number of nodes :: " << num_entries << "\n";
  for (size_t i = 0; i < num_entries; ++i) {
    std::pair<size_t, size_t> entry;
    size_t owner;
    meta_file.read(reinterpret_cast<char*>(&entry.first), sizeof(entry.first));
    meta_file.read(reinterpret_cast<char*>(&entry.second), sizeof(entry.second));
    meta_file.read(reinterpret_cast<char*>(&owner), sizeof(owner));
    localToGlobalMap_meta.push_back(NodeInfo(entry.second, entry.first, owner));
  }
  return true;
}


/**********Global vectors for book keeping*********************/
//std::vector<std::vector<uint64_t>> masterNodes(4); // master nodes on different hosts. For broadcast
//std::map<uint64_t, uint32_t> GIDtoOwnerMap;

/**************************************************************/

template<typename NodeTy, typename EdgeTy, bool BSPNode = false, bool BSPEdge = false>
class vGraph : public GlobalObject {

   typedef typename std::conditional<BSPNode, std::pair<NodeTy, NodeTy>, NodeTy>::type realNodeTy;
   typedef typename std::conditional<BSPEdge, std::pair<EdgeTy, EdgeTy>, EdgeTy>::type realEdgeTy;

   typedef galois::Graph::LC_CSR_Graph<realNodeTy, realEdgeTy> GraphTy;

  GraphTy graph;
  bool round;
   uint64_t totalNodes; // Total nodes in the complete graph.
   uint32_t numOwned; // [0, numOwned) = global nodes owned, thus [numOwned, numNodes are replicas
   uint64_t globalOffset; // [numOwned, end) + globalOffset = GID
  unsigned id; // my hostid // FIXME: isn't this just Network::ID?
   //ghost cell ID translation
   std::vector<uint64_t> ghostMap; // GID = ghostMap[LID - numOwned]
   std::vector<std::pair<uint32_t, uint32_t> > hostNodes; //LID Node owned by host i
   //pointer for each host
   std::vector<uintptr_t> hostPtrs;

  /*** Vertex Cut ***/
  std::vector<NodeInfo> localToGlobalMap_meta;
  std::vector<std::vector<size_t>> mirrorNodes; // mirror nodes from different hosts. For reduce
  std::vector<std::vector<size_t>> masterNodes; // master nodes on different hosts. For broadcast
#ifdef __GALOIS_SIMULATE_COMMUNICATION__
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
  unsigned comm_mode; // Communication mode: 0 - original, 1 - simulated net, 2 - simulated bare MPI
#endif
#endif
  std::unordered_map<size_t, size_t> LocalToGlobalMap;
  std::unordered_map<size_t, size_t> GlobalToLocalMap;

  std::unordered_map<size_t, size_t> GIDtoOwnerMap;

  std::vector<size_t> OwnerVec; //To store the ownerIDs of sorted according to the Global IDs.
  std::vector<size_t> GlobalVec; //Global Id's sorted vector.
  std::vector<size_t> LocalVec; //Local Id's sorted vector.

   //GID to owner
   std::vector<std::pair<uint64_t, uint64_t>> gid2host;

   uint32_t num_iter_push; //Keep track of number of iterations.
   uint32_t num_iter_pull; //Keep track of number of iterations.
   uint32_t num_run; //Keep track of number of iterations.

#if 0
   //host -> (lid, lid]
   std::pair<uint32_t, uint32_t> nodes_by_host(uint32_t host) const {
      return hostNodes[host];
   }

   std::pair<uint64_t, uint64_t> nodes_by_host_G(uint32_t host) const {
      return gid2host[host];
   }
#endif

  size_t L2G(size_t lid) {
    //return LocalToGlobalMap[lid];
    return GlobalVec[lid];
  }

  size_t G2L(size_t gid) {

    //we can assume that GID exits and is unique. Index is localID since it is sorted.
    auto iter = std::lower_bound(GlobalVec.begin(), GlobalVec.end(), gid);
    assert(*iter == gid);
    if(*iter == gid)
      return (iter - GlobalVec.begin());
    else
      abort();
    //return GlobalToLocalMap[gid];
  }

#if 0
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
#endif

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
 
   template<typename FnTy>
   void syncRecvApply(uint32_t from_id, galois::Runtime::RecvBuffer& buf, std::string loopName) {
     auto& net = galois::Runtime::getSystemNetworkInterface();
     std::string doall_str("LAMBDA::REDUCE_RECV_APPLY_" + loopName + "_" + std::to_string(num_run));
     galois::Runtime::reportLoopInstance(doall_str);
     galois::StatTimer StatTimer_set("SYNC_SET", loopName, galois::start_now);

     uint32_t num = masterNodes[from_id].size();
     if(num > 0){
       std::vector<typename FnTy::ValTy> val_vec(num);
       galois::Runtime::gDeserialize(buf, val_vec);
       if (!FnTy::reduce_batch(from_id, &val_vec[0])) {
       galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num),
                      [&](uint32_t n){
                        uint32_t lid = masterNodes[from_id][n];
                        FnTy::reduce(lid, getData(lid), val_vec[n]);
                      }, galois::loopname(doall_str.c_str()));
       }
     }
   }

   template<typename FnTy>
   void syncBroadcastRecvReply(uint32_t from_id, galois::Runtime::RecvBuffer& buf) {
     auto& net = galois::Runtime::getSystemNetworkInterface();
     uint32_t num;
     std::string loopName;
     galois::Runtime::gDeserialize(buf, loopName, num);
     galois::StatTimer StatTimer_extract("SYNC_EXTRACT", loopName, galois::start_now);
     galois::Statistic SyncBroadcastReply_send_bytes("SEND_BYTES_BROADCAST_REPLY", loopName);
     std::string doall_str("LAMBDA::BROADCAST_RECV_REPLY_" + loopName + "_" + std::to_string(num_run));
     galois::Runtime::reportLoopInstance(doall_str);
     galois::Runtime::SendBuffer b;
     assert(num == masterNodes[from_id].size());
     gSerialize(b, loopName, num);

#if 0
     /********** Serial loop: works ****************/
     for(auto n : masterNodes[from_id]){
       typename FnTy::ValTy val;
       auto localID = G2L(n);
       val = FnTy::extract((localID), getData(localID));
       
       galois::Runtime::gSerialize(b, n, val);
     }
     
     SyncBroadcastReply_send_bytes += b.size();
     net.sendMsg(from_id, syncRecv, b);
#endif
     
     std::vector<typename FnTy::ValTy> val_vec(num);
     if(num >0) {
       //std::cout << "["<< net.ID << "] num : " << num << "\n";
       
       if (!FnTy::extract_batch(from_id, &val_vec[0])) {
         galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n){
             uint32_t localID = masterNodes[from_id][n];
             //std::cout << "["<< net.ID << "] n : " << n << "\n";
             auto val = FnTy::extract((localID), getData(localID));
             assert(n < num);
             val_vec[n] = val;
             
           }, galois::loopname(doall_str.c_str()));
       }
       
     }
     galois::Runtime::gSerialize(b, val_vec);
     StatTimer_extract.stop();
     //std::cout << "[" << net.ID << "] Serialized : sending to other host\n";
     SyncBroadcastReply_send_bytes += b.size();
     net.sendTagged(from_id, galois::Runtime::evilPhase + 1, b);
     //     net.sendMsg(from_id, syncRecv, b);
   }
  
  template<typename FnTy>
  void syncBroadcastRecvApply(uint32_t from_id, galois::Runtime::RecvBuffer& buf, std::string loopName) {
    std::string doall_str("LAMBDA::BROADCAST_RECV_APPLY_" + loopName + "_" + std::to_string(num_run));
    galois::Runtime::reportLoopInstance(doall_str);
    galois::StatTimer StatTimer_set("SYNC_SET", loopName, galois::start_now);

    uint32_t num = mirrorNodes[from_id].size();
    auto& net = galois::Runtime::getSystemNetworkInterface();


    //std::cerr << "["<<id<<"] broadcast APPLY INSIDE sent from : " << from_id << " tag : " << galois::Runtime::evilPhase << "\n";
    if(num > 0 ){
      std::vector<typename FnTy::ValTy> val_vec(num);

      galois::Runtime::gDeserialize(buf, val_vec);

      if (!FnTy::setVal_batch(from_id, &val_vec[0])) {
        galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n){
            uint32_t localID = mirrorNodes[from_id][n];
            FnTy::setVal((localID), getData(localID), val_vec[n]);}, galois::loopname(doall_str.c_str()));
      }
    }
  }

public:
   typedef typename GraphTy::GraphNode GraphNode;
   typedef typename GraphTy::iterator iterator;
   typedef typename GraphTy::const_iterator const_iterator;
   typedef typename GraphTy::local_iterator local_iterator;
   typedef typename GraphTy::const_local_iterator const_local_iterator;
   typedef typename GraphTy::edge_iterator edge_iterator;


  //vGraph construction is collective
  // FIXME: use scalefactor to balance load
  vGraph(const std::string& filename, const std::string& partitionFolder, unsigned host, unsigned numHosts, std::vector<unsigned> scalefactor = std::vector<unsigned>())
    :GlobalObject(this), id(host),round(false)
  {
#ifdef __GALOIS_SIMULATE_COMMUNICATION__
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
    comm_mode = 0;
#endif
#endif
    std::string part_fileName = getPartitionFileName(partitionFolder,id,numHosts);
    std::string part_metaFile = getMetaFileName(partitionFolder, id, numHosts);

    galois::Graph::OfflineGraph g(part_fileName);
    num_iter_push = 0;
    num_iter_pull = 0;
    num_run = 0;
    totalNodes = g.size();
    std::cerr << "[" << id << "] SIZE ::::  " << totalNodes << "\n";
    readMetaFile(part_metaFile, localToGlobalMap_meta);
    std::cerr << "[" << id << "] MAPSIZE : " << localToGlobalMap_meta.size() << "\n";
    masterNodes.resize(numHosts);
    mirrorNodes.resize(numHosts);

#if 0
    for(auto info : localToGlobalMap_meta){
      assert(info.owner_id >= 0 && info.owner_id < numHosts);
      mirrorNodes[info.owner_id].push_back(info.global_id);

      GIDtoOwnerMap[info.global_id] = info.owner_id;
      LocalToGlobalMap[info.local_id] = info.global_id;
      GlobalToLocalMap[info.global_id] = info.local_id;
      //galois::Runtime::printOutput("[%] Owner : %\n", info.global_id, info.owner_id);
    }
#endif


    //compute owners for all nodes
    numOwned = g.size();//gid2host[id].second - gid2host[id].first;

    uint64_t numEdges = g.edge_begin(*g.end()) - g.edge_begin(*g.begin()); // depends on Offline graph impl
    std::cerr << "[" << id << "] Edge count Done " << numEdges << "\n";


    uint32_t numNodes = numOwned;
    graph.allocateFrom(numNodes, numEdges);
    //std::cerr << "Allocate done\n";

    graph.constructNodes();
    //std::cerr << "Construct nodes done\n";
    loadEdges<std::is_void<EdgeTy>::value>(g);

    setup_communication(numHosts);

#ifdef __GALOIS_SIMULATE_COMMUNICATION__
#ifndef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
    simulate_communication();
#endif
#endif
  }

  void setup_communication(unsigned numHosts) {
    galois::StatTimer StatTimer_comm_setup("COMMUNICATION_SETUP_TIME");
    StatTimer_comm_setup.start();

    for(auto info : localToGlobalMap_meta){
      assert(info.owner_id >= 0 && info.owner_id < numHosts);
      mirrorNodes[info.owner_id].push_back(info.global_id);

      GlobalVec.push_back(info.global_id);
      OwnerVec.push_back(info.owner_id);
      LocalVec.push_back(info.local_id);
      //galois::Runtime::printOutput("[%] Owner : %\n", info.global_id, info.owner_id);
    }

    //Check to make sure GlobalVec is sorted. Everything depends on it.
    assert(std::is_sorted(GlobalVec.begin(), GlobalVec.end()));
    if(!std::is_sorted(GlobalVec.begin(), GlobalVec.end())){
      std::cerr << "GlobalVec not sorted; Aborting execution\n";
      abort();
    }
    if(!std::is_sorted(LocalVec.begin(), LocalVec.end())){
      std::cerr << "LocalVec not sorted; Aborting execution\n";
      abort();
    }

    //Exchange information.
    exchange_info_init();

    for(uint32_t h = 0; h < masterNodes.size(); ++h){
       galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(masterNodes[h].size()),
           [&](uint32_t n){
           masterNodes[h][n] = G2L(masterNodes[h][n]);
           }, galois::loopname("MASTER_NODES"));
    }
    //masterNodes.resize(hostNodes.size());

    for(uint32_t h = 0; h < mirrorNodes.size(); ++h){
       galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(mirrorNodes[h].size()),
           [&](uint32_t n){
           mirrorNodes[h][n] = G2L(mirrorNodes[h][n]);
           }, galois::loopname("MIRROR_NODES"));
    }
    //mirrorNodes.resize(hostNodes.size());

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
   void loadEdges(galois::Graph::OfflineGraph & g) {
      fprintf(stderr, "Loading edge-data while creating edges.\n");
      uint64_t cur = 0;
      for (auto n = g.begin(); n != g.end(); ++n) {
           for (auto ii = g.edge_begin(*n), ee = g.edge_end(*n); ii < ee; ++ii) {
            auto gdst = g.getEdgeDst(ii);
            auto gdata = g.getEdgeData<EdgeTy>(ii);
             graph.constructEdge(cur++, gdst, gdata);
         }
           graph.fixEndEdge((*n), cur);
      }
   }
   template<bool isVoidType, typename std::enable_if<isVoidType>::type* = nullptr>
   void loadEdges(galois::Graph::OfflineGraph & g) {
      fprintf(stderr, "Loading void edge-data while creating edges.\n");
      uint64_t cur = 0;
     for(auto n = g.begin(); n != g.end(); ++n){
           for (auto ii = g.edge_begin(*n), ee = g.edge_end(*n); ii < ee; ++ii) {
            auto gdst = g.getEdgeDst(ii);
             graph.constructEdge(cur++, gdst);
         }
           graph.fixEndEdge((*n), cur);
      }
   }

   NodeTy& getData(GraphNode N, galois::MethodFlag mflag = galois::MethodFlag::WRITE) {
      auto& r = getDataImpl<BSPNode>(N, mflag);
//    auto i =galois::Runtime::NetworkInterface::ID;
      //std::cerr << i << " " << N << " " <<&r << " " << r.dist_current << "\n";
      return r;
   }

   const NodeTy& getData(GraphNode N, galois::MethodFlag mflag = galois::MethodFlag::WRITE) const {
      auto& r = getDataImpl<BSPNode>(N, mflag);
//    auto i =galois::Runtime::NetworkInterface::ID;
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


  void exchange_info_init(){
    auto& net = galois::Runtime::getSystemNetworkInterface();
    //may be reusing tag, so need a barrier
    galois::Runtime::getHostBarrier().wait();

    //send
    for (unsigned x = 0; x < net.Num; ++x) {
      if((x == id))
        continue;

      galois::Runtime::SendBuffer b;
      gSerialize(b, (uint64_t)mirrorNodes[x].size(), mirrorNodes[x]);
      net.sendTagged(x, 1, b);
      std::cout << " number of mirrors from : " << x << " : " << mirrorNodes[x].size() << "\n";
    }
    //recieve
    for (unsigned x = 0; x < net.Num; ++x) {
      if ((x == id))
        continue;
      decltype(net.recieveTagged(1, nullptr)) p;
      do {
        net.handleReceives();
        p = net.recieveTagged(1, nullptr);
      } while (!p);

      uint64_t numItems;
      galois::Runtime::gDeserialize(p->second, numItems);
      galois::Runtime::gDeserialize(p->second, masterNodes[p->first]);
      std::cout << "from : " << p->first << " -> " << numItems << " --> " << masterNodes[p->first].size() << "\n";
    }

    //may be reusing tag, so need a barrier
    galois::Runtime::getHostBarrier().wait();
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
      auto& net = galois::Runtime::getSystemNetworkInterface();

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
      auto& net = galois::Runtime::getSystemNetworkInterface();

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
      auto& net = galois::Runtime::getSystemNetworkInterface();

      std::vector<MPI_Request> requests(2 * net.Num);
      unsigned num_requests = 0;

      galois::Runtime::SendBuffer sb[net.Num];
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

         galois::Runtime::gSerialize(sb[x], val_vec);
         assert(size == sb[x].size());
         
         SyncBroadcast_send_bytes += size;
         //std::cerr << "[" << id << "]" << " mpi send to " << x << " : " << size << "\n";
         MPI_Isend(sb[x].linearData(), size, MPI_BYTE, x, 32767, MPI_COMM_WORLD, &requests[num_requests++]);
      }

      galois::Runtime::RecvBuffer rb[net.Num];
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
         galois::Runtime::gDeserialize(rb[x], val_vec);
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
      auto& net = galois::Runtime::getSystemNetworkInterface();

      std::vector<MPI_Request> requests(2 * net.Num);
      unsigned num_requests = 0;

      galois::Runtime::SendBuffer sb[net.Num];
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

         galois::Runtime::gSerialize(sb[x], val_vec);
         assert(size == sb[x].size());

         SyncReduce_send_bytes += size;
         //std::cerr << "[" << id << "]" << " mpi send to " << x << " : " << size << "\n";
         MPI_Isend(sb[x].linearData(), size, MPI_BYTE, x, 32767, MPI_COMM_WORLD, &requests[num_requests++]);
      }

      galois::Runtime::RecvBuffer rb[net.Num];
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
         galois::Runtime::gDeserialize(rb[x], val_vec);
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
  static void syncRecv(uint32_t src, galois::Runtime::RecvBuffer& buf) {
      uint32_t oid;
      void (vGraph::*fn)(galois::Runtime::RecvBuffer&);
      galois::Runtime::gDeserialize(buf, oid, fn);
      vGraph* obj = reinterpret_cast<vGraph*>(ptrForObj(oid));
      (obj->*fn)(buf);
      //--(obj->num_recv_expected);
      //std::cout << "[ " << galois::Runtime::getSystemNetworkInterface().ID << "] " << " NUM RECV EXPECTED : " << (obj->num_recv_expected) << "\n";
   }

#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
   template<typename FnTy>
#endif
   void syncRecvApplyPull(galois::Runtime::RecvBuffer& buf) {
     unsigned from_id;
     uint32_t num;
     std::string loopName;
     uint32_t num_iter_push;
     galois::Runtime::gDeserialize(buf, from_id, num);
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
     std::vector<typename FnTy::ValTy> val_vec(num);
#else
     std::vector<uint64_t> val_vec(num);
#endif
     galois::Runtime::gDeserialize(buf, val_vec);
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
   void syncRecvApplyPush(galois::Runtime::RecvBuffer& buf) {
     unsigned from_id;
     uint32_t num;
     std::string loopName;
     uint32_t num_iter_push;
     galois::Runtime::gDeserialize(buf, from_id, num);
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
     std::vector<typename FnTy::ValTy> val_vec(num);
#else
     std::vector<uint64_t> val_vec(num);
#endif
     galois::Runtime::gDeserialize(buf, val_vec);
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
      void (vGraph::*fn)(galois::Runtime::RecvBuffer&) = &vGraph::syncRecvApplyPull<FnTy>;
#else
      void (vGraph::*fn)(galois::Runtime::RecvBuffer&) = &vGraph::syncRecvApplyPull;
#endif
      galois::StatTimer StatTimer_syncBroadcast("SIMULATE_NET_BROADCAST");
      galois::Statistic SyncBroadcast_send_bytes("SIMULATE_NET_BROADCAST_SEND_BYTES");

#ifndef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
      galois::Runtime::getHostBarrier().wait();
#endif
      StatTimer_syncBroadcast.start();
      auto& net = galois::Runtime::getSystemNetworkInterface();

      for (unsigned x = 0; x < net.Num; ++x) {
         uint32_t num = masterNodes[x].size();
         if((x == id) || (num == 0))
           continue;

         galois::Runtime::SendBuffer b;
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

      galois::Runtime::getHostBarrier().wait();
      StatTimer_syncBroadcast.stop();
   }

#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
   template<typename FnTy>
#endif
   void simulate_reduce(std::string loopName) {
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
      void (vGraph::*fn)(galois::Runtime::RecvBuffer&) = &vGraph::syncRecvApplyPush<FnTy>;
#else
      void (vGraph::*fn)(galois::Runtime::RecvBuffer&) = &vGraph::syncRecvApplyPush;
#endif
      galois::StatTimer StatTimer_syncReduce("SIMULATE_NET_REDUCE");
      galois::Statistic SyncReduce_send_bytes("SIMULATE_NET_REDUCE_SEND_BYTES");

#ifndef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
      galois::Runtime::getHostBarrier().wait();
#endif
      StatTimer_syncReduce.start();
      auto& net = galois::Runtime::getSystemNetworkInterface();

      for (unsigned x = 0; x < net.Num; ++x) {
         uint32_t num = mirrorNodes[x].size();
         if((x == id) || (num == 0))
           continue;

         galois::Runtime::SendBuffer b;
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

      galois::Runtime::getHostBarrier().wait();

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
    std::string doall_str("LAMBDA::REDUCE_" + loopName + "_" + std::to_string(num_run) + "_" + std::to_string(num_iter_push));
    galois::Statistic SyncReduce_send_bytes("SEND_BYTES_REDUCE", loopName);
    galois::StatTimer StatTimer_extract("REDUCE_EXTRACT", loopName);
    galois::StatTimer StatTimer_syncReduce("REDUCE", loopName, galois::start_now);
    auto& net = galois::Runtime::getSystemNetworkInterface();

    for (unsigned x = 0; x < net.Num; ++x) {
      uint32_t num = mirrorNodes[x].size();
      if((x == id))
        continue;

      galois::Runtime::SendBuffer b;

      StatTimer_extract.start();
      if(num > 0 ){
        std::vector<typename FnTy::ValTy> val_vec(num);

        if (!FnTy::extract_reset_batch(x, &val_vec[0])) {
          galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n){
              uint32_t lid = mirrorNodes[x][n];
              auto val = FnTy::extract(lid, getData(lid));
              FnTy::reset(lid, getData(lid));
              val_vec[n] = val;
            }, galois::loopname(doall_str.c_str()));
        }

        gSerialize(b, val_vec);
      } else {
        gSerialize(b, loopName);
      }
      StatTimer_extract.stop();

      SyncReduce_send_bytes += b.size();
      net.sendTagged(x, galois::Runtime::evilPhase, b);
    }

    net.flush();

    for (unsigned x = 0; x < net.Num; ++x) {
      if ((x == id))
        continue;
      decltype(net.recieveTagged(galois::Runtime::evilPhase,nullptr)) p;
      do {
        net.handleReceives();
        p = net.recieveTagged(galois::Runtime::evilPhase, nullptr);
      } while (!p);
      syncRecvApply<FnTy>(p->first, p->second, loopName);
    }
    ++galois::Runtime::evilPhase;
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
    std::string doall_str("LAMBDA::BROADCAST_" + loopName + "_" + std::to_string(num_run) + "_" + std::to_string(num_iter_pull));
    galois::Statistic SyncBroadcast_send_bytes("SEND_BYTES_BROADCAST", loopName);
    galois::StatTimer StatTimer_extract("BROADCAST_EXTRACT", loopName);
    galois::StatTimer StatTimer_syncBroadcast("BROADCAST", loopName, galois::start_now);
    auto& net = galois::Runtime::getSystemNetworkInterface();

    for (unsigned x = 0; x < net.Num; ++x) {
      uint32_t num = masterNodes[x].size();
      if((x == id))
        continue;

      galois::Runtime::SendBuffer b;

      StatTimer_extract.start();
      if(num > 0 ){
        std::vector<typename FnTy::ValTy> val_vec(num);
        if (!FnTy::extract_batch(x, &val_vec[0])) {

          galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n){
              uint32_t localID = masterNodes[x][n];
              auto val = FnTy::extract((localID), getData(localID));
              val_vec[n] = val;
              }, galois::loopname(doall_str.c_str()));
        }
        gSerialize(b, val_vec);
      } else {
        gSerialize(b, loopName);
      }
      StatTimer_extract.stop();

      SyncBroadcast_send_bytes += b.size();
      net.sendTagged(x, galois::Runtime::evilPhase, b);
      //std::cerr << "["<<id<<"] broadcast sent to : " << x << " tag : " << galois::Runtime::evilPhase <<"\n";

    }

    net.flush();

    for (unsigned x = 0; x < net.Num; ++x) {
      if ((x == id))
        continue;
      decltype(net.recieveTagged(galois::Runtime::evilPhase,nullptr)) p;
      do {
        net.handleReceives();
        p = net.recieveTagged(galois::Runtime::evilPhase, nullptr);
      } while (!p);
      //std::cerr << "["<<id<<"] broadcast APPLY sent from : " << x << " tag : " << galois::Runtime::evilPhase <<"\n";
      syncBroadcastRecvApply<FnTy>(p->first, p->second, loopName);
    }

    ++galois::Runtime::evilPhase;
    StatTimer_syncBroadcast.stop();
  }

  uint64_t getGID(size_t nodeID) {
      return L2G(nodeID);
   }
  uint32_t getLID(uint64_t nodeID) {
      return G2L(nodeID);
   }

   unsigned getHostID(uint64_t gid) {
    auto lid = G2L(gid);
    return OwnerVec[lid];
    //return GIDtoOwnerMap[gid];
   }
   uint32_t getNumOwned() const {
      return numOwned;
   }

   uint64_t getGlobalOffset() const {
     return 0;
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

  /**For resetting num_iter_pull and push.**/
   void reset_num_iter(uint32_t runNum){
      num_iter_pull = 0;
      num_iter_push = 0;
      num_run = runNum;
   }
   uint32_t get_run_num() {
     return num_run;
   }

};
#endif//_GALOIS_DIST_vGraph_H
