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
//std::vector<std::vector<uint64_t>> masterNodes(4); // master nodes on different hosts. For sync_pull
//std::map<uint64_t, uint32_t> GIDtoOwnerMap;

/**************************************************************/

template<typename NodeTy, typename EdgeTy, bool BSPNode = false, bool BSPEdge = false>
class vGraph : public GlobalObject {

   typedef typename std::conditional<BSPNode, std::pair<NodeTy, NodeTy>, NodeTy>::type realNodeTy;
   typedef typename std::conditional<BSPEdge, std::pair<EdgeTy, EdgeTy>, EdgeTy>::type realEdgeTy;

   typedef Galois::Graph::LC_CSR_Graph<realNodeTy, realEdgeTy> GraphTy;

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
  std::vector<std::vector<size_t>> slaveNodes; // slave nodes from different hosts. For sync_push
  std::vector<std::vector<size_t>> masterNodes; // master nodes on different hosts. For sync_pull
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
 
   template<typename FnTy>
   void syncRecvApply(uint32_t from_id, Galois::Runtime::RecvBuffer& buf) {
     uint32_t num;
     std::string loopName;
     auto& net = Galois::Runtime::getSystemNetworkInterface();
     Galois::Runtime::gDeserialize(buf, loopName, num);
     std::string doall_str("LAMBDA::SYNC_PUSH_RECV_APPLY_" + loopName + "_" + std::to_string(num_run));
     Galois::Runtime::reportLoopInstance(doall_str);
     Galois::StatTimer StatTimer_set("SYNC_SET", loopName, Galois::start_now);
#if 0
     for (; num; --num) {
       size_t gid;
       typename FnTy::ValTy val;
       Galois::Runtime::gDeserialize(buf, gid, val);
       //TODO: implement master
       //assert(isMaster(gid));
       auto lid = G2L(gid);
       FnTy::reduce(lid, getData(lid), val);
     }
#endif
     assert(num == masterNodes[from_id].size());
     if(num > 0){
       std::vector<typename FnTy::ValTy> val_vec(num);
       Galois::Runtime::gDeserialize(buf, val_vec);
       //         if (!FnTy::reduce_batch(from_id, &val_vec[0])) {
       Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num),
                      [&](uint32_t n){
                        auto lid = masterNodes[from_id][n];
                        FnTy::reduce(lid, getData(lid), val_vec[n]);
                      }, Galois::loopname(doall_str.c_str()));
       //         }
     }
   }

   template<typename FnTy>
   void syncPullRecvReply(uint32_t from_id, Galois::Runtime::RecvBuffer& buf) {
     auto& net = Galois::Runtime::getSystemNetworkInterface();
     uint32_t num;
     std::string loopName;
     Galois::Runtime::gDeserialize(buf, loopName, num);
     Galois::StatTimer StatTimer_extract("SYNC_EXTRACT", loopName, Galois::start_now);
     Galois::Statistic SyncPullReply_send_bytes("SEND_BYTES_SYNC_PULL_REPLY", loopName);
     std::string doall_str("LAMBDA::SYNC_PULL_RECV_REPLY_" + loopName + "_" + std::to_string(num_run));
     Galois::Runtime::reportLoopInstance(doall_str);
     Galois::Runtime::SendBuffer b;
     assert(num == masterNodes[from_id].size());
     gSerialize(b, num);

#if 0
     /********** Serial loop: works ****************/
     for(auto n : masterNodes[from_id]){
       typename FnTy::ValTy val;
       auto localID = G2L(n);
       val = FnTy::extract((localID), getData(localID));
       
       Galois::Runtime::gSerialize(b, n, val);
     }
     
     SyncPullReply_send_bytes += b.size();
     net.sendMsg(from_id, syncRecv, b);
#endif
     
     if(num >0) {
       std::vector<typename FnTy::ValTy> val_vec(num);
       //std::cout << "["<< net.ID << "] num : " << num << "\n";
       
       //       if (!FnTy::extract_batch(from_id, &val_vec[0])) {
         Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n){
             auto localID = masterNodes[from_id][n];
             //std::cout << "["<< net.ID << "] n : " << n << "\n";
             auto val = FnTy::extract((localID), getData(localID));
             assert(n < num);
             val_vec[n] = val;
             
           }, Galois::loopname(doall_str.c_str()));
         //       }
       
       Galois::Runtime::gSerialize(b, val_vec);
     }
     StatTimer_extract.stop();
     //std::cout << "[" << net.ID << "] Serialized : sending to other host\n";
     SyncPullReply_send_bytes += b.size();
     net.sendTagged(from_id, Galois::Runtime::evilPhase + 1, b);
     //     net.sendMsg(from_id, syncRecv, b);
   }
  
  template<typename FnTy>
  void syncPullRecvApply(uint32_t from_id, Galois::Runtime::RecvBuffer& buf) {
    uint32_t num;
    std::string loopName;
    Galois::Runtime::gDeserialize(buf, loopName, num);
    std::string doall_str("LAMBDA::SYNC_PULL_RECV_APPLY_" + loopName + "_" + std::to_string(num_run));
    Galois::Runtime::reportLoopInstance(doall_str);
    Galois::StatTimer StatTimer_set("SYNC_SET", loopName, Galois::start_now);
    
    assert(num == slaveNodes[from_id].size());
    auto& net = Galois::Runtime::getSystemNetworkInterface();

    if(num > 0 ){
      std::vector<typename FnTy::ValTy> val_vec(num);
      
      Galois::Runtime::gDeserialize(buf, val_vec);

      //      if (!FnTy::setVal_batch(from_id, &val_vec[0])) {
        Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n){
            auto localID = slaveNodes[from_id][n];
            FnTy::setVal((localID), getData(localID), val_vec[n]);}, Galois::loopname(doall_str.c_str()));
        //      }  
    }
#if 0
    for (; num; --num) {
      uint64_t gid;
      typename FnTy::ValTy val;
      Galois::Runtime::gDeserialize(buf, gid, val);
      auto localId = G2L(gid);
      //std::cerr << "Applying pulled val : " << val<< "\n";
      FnTy::setVal(localId, getData(localId), val);
    }
#endif
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
    std::string part_fileName = getPartitionFileName(partitionFolder,id,numHosts);
    std::string part_metaFile = getMetaFileName(partitionFolder, id, numHosts);

    OfflineGraph g(part_fileName);
    num_iter_push = 0;
    num_iter_pull = 0;
    num_run = 0;
    totalNodes = g.size();
    std::cerr << "[" << id << "] SIZE ::::  " << totalNodes << "\n";
    readMetaFile(part_metaFile, localToGlobalMap_meta);
    std::cerr << "[" << id << "] MAPSIZE : " << localToGlobalMap_meta.size() << "\n";
    masterNodes.resize(numHosts);
    slaveNodes.resize(numHosts);

#if 0
    for(auto info : localToGlobalMap_meta){
      assert(info.owner_id >= 0 && info.owner_id < numHosts);
      slaveNodes[info.owner_id].push_back(info.global_id);

      GIDtoOwnerMap[info.global_id] = info.owner_id;
      LocalToGlobalMap[info.local_id] = info.global_id;
      GlobalToLocalMap[info.global_id] = info.local_id;
      //Galois::Runtime::printOutput("[%] Owner : %\n", info.global_id, info.owner_id);
    }
#endif

    for(auto info : localToGlobalMap_meta){
      assert(info.owner_id >= 0 && info.owner_id < numHosts);
      slaveNodes[info.owner_id].push_back(info.global_id);

      GlobalVec.push_back(info.global_id);
      OwnerVec.push_back(info.owner_id);
      LocalVec.push_back(info.local_id);
      //Galois::Runtime::printOutput("[%] Owner : %\n", info.global_id, info.owner_id);
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
       Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(masterNodes[h].size()),
           [&](uint32_t n){
           masterNodes[h][n] = G2L(masterNodes[h][n]);
           }, Galois::loopname("MASTER_NODES"));
    }
    masterNodes.resize(hostNodes.size());

    for(uint32_t h = 0; h < slaveNodes.size(); ++h){
       Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(slaveNodes[h].size()),
           [&](uint32_t n){
           slaveNodes[h][n] = G2L(slaveNodes[h][n]);
           }, Galois::loopname("SLAVE_NODES"));
    }
    slaveNodes.resize(hostNodes.size());

    for(auto x = 0; x < masterNodes.size(); ++x){
      std::string master_nodes_str = "MASTER_NODES_TO_" + std::to_string(x);
      Galois::Statistic StatMasterNodes(master_nodes_str);
      StatMasterNodes += masterNodes[x].size();
    }
    for(auto x = 0; x < slaveNodes.size(); ++x){
      std::string slave_nodes_str = "SLAVE_NODES_FROM_" + std::to_string(x);
      Galois::Statistic StatSlaveNodes(slave_nodes_str);
      StatSlaveNodes += slaveNodes[x].size();
    }


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
   }

   template<bool isVoidType, typename std::enable_if<!isVoidType>::type* = nullptr>
   void loadEdges(OfflineGraph & g) {
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
   void loadEdges(OfflineGraph & g) {
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
    auto& net = Galois::Runtime::getSystemNetworkInterface();
    //may be reusing tag, so need a barrier
    Galois::Runtime::getHostBarrier().wait();

    masterNodes.resize(net.Num);
    //send
    for (unsigned x = 0; x < net.Num; ++x) {
      if((x == id) || (slaveNodes[x].size() == 0))
        continue;

      Galois::Runtime::SendBuffer b;
      gSerialize(b, (uint64_t)slaveNodes[x].size(), slaveNodes[x]);
      //for(auto n : slaveNodes[x]){
        //gSerialize(b, n);
      //}
      net.sendTagged(x, 1, b);
      std::cout << " number of slaves from : " << x << " : " << slaveNodes[x].size() << "\n";
    }
    //recieve
    for (unsigned x = 0; x < net.Num; ++x) {
      if ((x == id) || (slaveNodes[x].size() == 0))
        continue;
      decltype(net.recieveTagged(1, nullptr)) p;
      do {
        p = net.recieveTagged(1, nullptr);
      } while (!p);

      uint64_t numItems;
      Galois::Runtime::gDeserialize(p->second, numItems);
      Galois::Runtime::gDeserialize(p->second, masterNodes[p->first]);
      std::cout << "from : " << p->first << " -> " << numItems << " --> " << masterNodes[p->first].size() << "\n";
    }
    //may be reusing tag, so need a barrier
    Galois::Runtime::getHostBarrier().wait();
   }

  template<typename FnTy>
  void sync_push(std::string loopName) {
    std::string doall_str("LAMBDA::SYNC_PUSH_" + loopName + "_" + std::to_string(num_run));
    Galois::Statistic SyncPush_send_bytes("SEND_BYTES_SYNC_PUSH", loopName);
    Galois::StatTimer StatTimer_extract("SYNC_EXTRACT", loopName);
    Galois::StatTimer StatTimer_syncPush("SYNC_PUSH", loopName, Galois::start_now);
    auto& net = Galois::Runtime::getSystemNetworkInterface();
    for (unsigned x = 0; x < net.Num; ++x) {
      uint32_t num = slaveNodes[x].size();
      if((x == id) || (num == 0))
        continue;
      
      Galois::Runtime::SendBuffer b;
      gSerialize(b, loopName,  num);
      
#if 0
      for (auto start = 0; start < slaveNodes[x].size(); ++start) {
        auto gid = slaveNodes[x][start];
        auto lid = G2L(gid);
        
        gSerialize(b, gid, FnTy::extract(lid, getData(lid)));
        FnTy::reset(lid, getData(lid));
      }
#endif
      
      StatTimer_extract.start();
      if(num > 0 ){
        std::vector<typename FnTy::ValTy> val_vec(num);
        
        //        if (!FnTy::extract_reset_batch(x, &val_vec[0])) {
          Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(num), [&](uint32_t n){
              auto lid = slaveNodes[x][n];
              auto val = FnTy::extract(lid, getData(lid));
              FnTy::reset(lid, getData(lid));
              val_vec[n] = val;
            }, Galois::loopname(doall_str.c_str()));
          //        }
        
        gSerialize(b, val_vec);
      }
      StatTimer_extract.stop();
      
      SyncPush_send_bytes += b.size();
      net.sendTagged(x, Galois::Runtime::evilPhase, b);
    }

    for (unsigned x = 0; x < net.Num; ++x) {
      if ((x == id) || (slaveNodes[x].size() == 0))
        continue;
      decltype(net.recieveTagged(Galois::Runtime::evilPhase,nullptr)) p;
      do {
        p = net.recieveTagged(Galois::Runtime::evilPhase, nullptr);
      } while (!p);
      syncRecvApply<FnTy>(p->first, p->second);
    }
    ++Galois::Runtime::evilPhase;
    //std::cout << "[" << net.ID <<"] time1 : " << time1.get() << "(msec) time2 : " << time2.get() << "(msec)\n";
  }


  template<typename FnTy>
  void sync_pull(std::string loopName) {
    Galois::Statistic SyncPull_send_bytes("SEND_BYTES_SYNC_PULL", loopName);
    Galois::StatTimer StatTimer_syncPull("SYNC_PULL", loopName, Galois::start_now);
    auto& net = Galois::Runtime::getSystemNetworkInterface();
    
    for (unsigned x = 0; x < net.Num; ++x) {
      if((x == id))
        continue;
      
      Galois::Runtime::SendBuffer b;
      gSerialize(b, loopName, (uint32_t)(slaveNodes[x].size()));
      SyncPull_send_bytes += b.size();
      net.sendTagged(x, Galois::Runtime::evilPhase, b);
    }
    
    net.flush();
    int num_recv_expected = net.Num - 1;
    while (num_recv_expected) {
      auto p = net.recieveTagged(Galois::Runtime::evilPhase, nullptr);
      if (p) {
        --num_recv_expected;
        syncPullRecvReply<FnTy>(p->first, p->second);
      }
      //         net.handleReceives();
    }
    ++Galois::Runtime::evilPhase;
    num_recv_expected = net.Num - 1;
    while (num_recv_expected) {
      auto p = net.recieveTagged(Galois::Runtime::evilPhase, nullptr);
      if (p) {
        --num_recv_expected;
        syncPullRecvApply<FnTy>(p->first, p->second);
      }
    }
    ++Galois::Runtime::evilPhase;
    assert(num_recv_expected == 0);
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
      m.num_master_nodes = (size_t *) calloc(hostNodes.size(), sizeof(size_t));;
      m.master_nodes = (size_t **) calloc(hostNodes.size(), sizeof(size_t *));;
      for(uint32_t h = 0; h < hostNodes.size(); ++h){
        m.num_master_nodes[h] = masterNodes[h].size();
        if (masterNodes[h].size() > 0) {
          m.master_nodes[h] = (size_t *) calloc(masterNodes[h].size(), sizeof(size_t));;
          std::copy(masterNodes[h].begin(), masterNodes[h].end(), m.master_nodes[h]);
        } else {
          m.master_nodes[h] = NULL;
        }
      }
      m.num_slave_nodes = (size_t *) calloc(hostNodes.size(), sizeof(size_t));;
      m.slave_nodes = (size_t **) calloc(hostNodes.size(), sizeof(size_t *));;
      for(uint32_t h = 0; h < hostNodes.size(); ++h){
        m.num_slave_nodes[h] = slaveNodes[h].size();
        if (slaveNodes[h].size() > 0) {
          m.slave_nodes[h] = (size_t *) calloc(slaveNodes[h].size(), sizeof(size_t));;
          std::copy(slaveNodes[h].begin(), slaveNodes[h].end(), m.slave_nodes[h]);
        } else {
          m.slave_nodes[h] = NULL;
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

};
#endif//_GALOIS_DIST_vGraph_H
