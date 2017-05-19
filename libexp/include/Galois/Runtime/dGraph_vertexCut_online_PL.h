/** partitioned graph wrapper for vertexCut -*- C++ -*-
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
 * @section Contains the vertex cut functionality to be used in dGraph.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Gurbinder Gill <gurbinder533@gmail.com>
 */

#include <vector>
#include <set>
#include <algorithm>
#include <unordered_map>
#include "Galois/Runtime/dGraph.h"
#include <boost/dynamic_bitset.hpp>
#include "Galois/Runtime/vecBool_bitset.h"
#include "Galois/Runtime/dGraph_edgeAssign_policy.h"
#include "Galois/Runtime/Dynamic_bitset.h"
#include "Galois/Graphs/FileGraph.h"
#include <sstream>

//template<typename NodeTy, typename EdgeTy, bool BSPNode = false, bool BSPEdge = false>
//class hGraph;

template<typename NodeTy, typename EdgeTy, bool BSPNode = false, bool BSPEdge = false>
class hGraph_vertexCut : public hGraph<NodeTy, EdgeTy, BSPNode, BSPEdge> {

  public:
    typedef hGraph<NodeTy, EdgeTy, BSPNode, BSPEdge> base_hGraph;
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


    std::vector<NodeInfo> localToGlobalMap_meta;
    std::vector<size_t> OwnerVec; //To store the ownerIDs of sorted according to the Global IDs.
    std::vector<size_t> GlobalVec; //Global Id's sorted vector.
    std::vector<std::pair<uint32_t, uint32_t>> hostNodes;

    std::vector<size_t> GlobalVec_ordered; //Global Id's sorted vector.
    // To send edges to different hosts: #Src #Dst
    std::vector<std::vector<uint64_t>> assigned_edges_perhost;


    //EXPERIMENT
    std::unordered_map<uint64_t, uint32_t> GlobalVec_map;

    //XXX: initialize to ~0
    std::vector<uint64_t> numNodes_per_host;
    std::unordered_map<uint64_t, std::vector<uint64_t>> host_edges_map;
    std::vector<uint64_t> numEdges_per_host;
    std::vector<std::pair<uint64_t, uint64_t>> gid2host;

    Galois::VecBool gid_vecBool;

    uint64_t globalOffset;
    uint32_t numNodes;
    uint64_t numEdges;

    unsigned getHostID(uint64_t gid) const {
      auto lid = G2L(gid);
      return OwnerVec[lid];
    }

    size_t getOwner_lid(size_t lid) const {
      return OwnerVec[lid];
    }

    virtual bool isLocal(uint64_t gid) const {
      return (GlobalVec_map.find(gid) != GlobalVec_map.end());
    }

    bool isOwned(uint64_t gid) const {
      for(auto i : hostNodes){
        if(i.first != ~0){
          auto iter = std::lower_bound(GlobalVec.begin() + i.first, GlobalVec.begin() + i.second, gid);
          if(iter != GlobalVec.end() && *iter == gid)
            return true;
        }
      }
      return false;
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

    std::string getPartitionFileName(const std::string & basename, unsigned hostID, unsigned num_hosts){
      std::string result = basename;
      result+= ".PART.";
      result+=std::to_string(hostID);
      result+= ".OF.";
      result+=std::to_string(num_hosts);
      return result;
    }

    std::pair<uint32_t, uint32_t> nodes_by_host(uint32_t host) const {
      return std::make_pair<uint32_t, uint32_t>(~0,~0);
    }

    std::pair<uint64_t, uint64_t> nodes_by_host_G(uint32_t host) const {
      return std::make_pair<uint64_t, uint64_t>(~0,~0);
    }


    hGraph_vertexCut(const std::string& filename, const std::string& partitionFolder,unsigned host, unsigned _numHosts, std::vector<unsigned> scalefactor, bool transpose = false) :  base_hGraph(host, _numHosts) {

      Galois::Runtime::reportStat("(NULL)", "ONLINE VERTEX CUT PL", 0, 0);

      Galois::Statistic statGhostNodes("TotalGhostNodes");
      Galois::StatTimer StatTimer_graph_construct("TIME_GRAPH_CONSTRUCT");
      Galois::StatTimer StatTimer_graph_construct_comm("TIME_GRAPH_CONSTRUCT_COMM");
      Galois::StatTimer StatTimer_distributed_edges("TIMER_DISTRIBUTE_EDGES");
      Galois::StatTimer StatTimer_distributed_edges_policy("TIMER_DISTRIBUTE_EDGES_POLICY");
      Galois::StatTimer StatTimer_distributed_edges_map("TIMER_DISTRIBUTE_EDGES_MAP");
      Galois::StatTimer StatTimer_distributed_edges_test_set_bit("TIMER_DISTRIBUTE_EDGES_TEST_SET_BIT");
      Galois::StatTimer StatTimer_distributed_edges_get_dst("TIMER_DISTRIBUTE_EDGES_GET_DST");
      Galois::StatTimer StatTimer_distributed_edges_get_edges("TIMER_DISTRIBUTE_EDGES_GET_EDGES");
      Galois::StatTimer StatTimer_distributed_edges_inner_loop("TIMER_DISTRIBUTE_EDGES_INNER_LOOP");
      Galois::StatTimer StatTimer_distributed_edges_next_src("TIMER_DISTRIBUTE_EDGES_NEXT_SRC");

      Galois::Graph::OfflineGraph g(filename);

      base_hGraph::totalNodes = g.size();
      std::cerr << "[" << base_hGraph::id << "] Total nodes : " << base_hGraph::totalNodes << "\n";
      //compute owners for all nodes
      if (scalefactor.empty() || (base_hGraph::numHosts == 1)) {
        for (unsigned i = 0; i < base_hGraph::numHosts; ++i)
          gid2host.push_back(Galois::block_range(0U, (unsigned) g.size(), i, base_hGraph::numHosts));
      } else {
        assert(scalefactor.size() == base_hGraph::numHosts);
        unsigned numBlocks = 0;
        for (unsigned i = 0; i < base_hGraph::numHosts; ++i)
          numBlocks += scalefactor[i];
        std::vector<std::pair<uint64_t, uint64_t>> blocks;
        for (unsigned i = 0; i < numBlocks; ++i)
          blocks.push_back(Galois::block_range(0U, (unsigned) g.size(), i, numBlocks));
        std::vector<unsigned> prefixSums;
        prefixSums.push_back(0);
        for (unsigned i = 1; i < base_hGraph::numHosts; ++i)
          prefixSums.push_back(prefixSums[i - 1] + scalefactor[i - 1]);
        for (unsigned i = 0; i < base_hGraph::numHosts; ++i) {
          unsigned firstBlock = prefixSums[i];
          unsigned lastBlock = prefixSums[i] + scalefactor[i] - 1;
          gid2host.push_back(std::make_pair(blocks[firstBlock].first, blocks[lastBlock].second));
        }
      }

      /*base_hGraph::totalOwnedNodes = base_hGraph::numOwned = gid2host[base_hGraph::id].second - gid2host[base_hGraph::id].first;*/
      globalOffset = gid2host[base_hGraph::id].first;
      //std::cerr << "[" << base_hGraph::id << "] Owned nodes: " << base_hGraph::numOwned << "\n";

      uint64_t numEdges_distribute = g.edge_begin(gid2host[base_hGraph::id].second) - g.edge_begin(gid2host[base_hGraph::id].first); // depends on Offline graph impl
      std::cerr << "[" << base_hGraph::id << "] Total edges to distribute : " << numEdges_distribute << "\n";


      //Go over assigned nodes and distribute edges.
      uint32_t edgeNum_threshold = 100; //node with < threshold edges is small else is big.

      assigned_edges_perhost.resize(base_hGraph::numHosts);
      for(auto src = gid2host[base_hGraph::id].first; src != gid2host[base_hGraph::id].second; ++src){
        auto num_edges = std::distance(g.edge_begin(src), g.edge_end(src));
        if(num_edges > edgeNum_threshold){
          //Assign edges for high degree nodes to the destination
          for(auto ee = g.edge_begin(src), ee_end = g.edge_end(src); ee != ee_end; ++ee){
            auto dst = g.getEdgeDst(ee);
            auto h = find_hostID(dst);
            assigned_edges_perhost[h].push_back(src);
            assigned_edges_perhost[h].push_back(dst);
          }
        }
        else{
          //keep all edges with the source node
          for(auto ee = g.edge_begin(src), ee_end = g.edge_end(src); ee != ee_end; ++ee){
            auto dst = g.getEdgeDst(ee);
            assigned_edges_perhost[base_hGraph::id].push_back(src);
            assigned_edges_perhost[base_hGraph::id].push_back(dst);
          }
        }
      }

      uint64_t check_numEdges = 0;
      for(uint32_t h = 0; h < base_hGraph::numHosts; ++h){
        std::cout << "from : " << base_hGraph::id << " to : " << h << " : edges assigned : " << assigned_edges_perhost[h].size() << "\n";
        check_numEdges += assigned_edges_perhost[h].size();
      }

      assert(check_numEdges == 2*numEdges_distribute);

      std::cerr << "exchange_edges started\n";
      exchange_edges();
      std::cerr << "exchange_edges done\n";

      fill_slaveNodes(base_hGraph::slaveNodes);
      std::cerr << "fill_slaveNodes done\n";

      std::cerr << "numNodes : " << numNodes << " , numEdges : " << numEdges << "\n";

      base_hGraph::numOwned = numNodes;

      base_hGraph::graph.allocateFrom(numNodes, numEdges);
      std::cerr << "Allocate done\n";

      base_hGraph::graph.constructNodes();
      std::cerr << "Construct nodes done\n";


      Galois::StatTimer StatTimer_load_edges("TIMER_LOAD_EDGES");

      StatTimer_load_edges.start();
      loadEdges(base_hGraph::graph);
      StatTimer_load_edges.stop();

      StatTimer_graph_construct.stop();

      std::cerr << base_hGraph::id << " Edges loaded \n";

      StatTimer_graph_construct_comm.start();
      base_hGraph::setup_communication();
      std::cerr << base_hGraph::id << " setup_communication done \n";
      StatTimer_graph_construct_comm.stop();


    }


    void exchange_edges(){
      Galois::StatTimer StatTimer_exchange_edges("EXCHANGE_EDGES_TIME");
      Galois::Runtime::getHostBarrier().wait(); // so that all hosts start the timer together

      auto& net = Galois::Runtime::getSystemNetworkInterface();

      //send and clear assigned_edges_perhost to receive from other hosts
      for (unsigned x = 0; x < net.Num; ++x) {
        if(x == base_hGraph::id) continue;

        Galois::Runtime::SendBuffer b;
        gSerialize(b, assigned_edges_perhost[x]);
        net.sendTagged(x, Galois::Runtime::evilPhase, b);
        assigned_edges_perhost[x].clear();
      }

      //receive
      for (unsigned x = 0; x < net.Num; ++x) {
        if(x == base_hGraph::id) continue;

        decltype(net.recieveTagged(Galois::Runtime::evilPhase, nullptr)) p;
        do {
          net.handleReceives();
          p = net.recieveTagged(Galois::Runtime::evilPhase, nullptr);
        } while(!p);

        Galois::Runtime::gDeserialize(p->second, assigned_edges_perhost[p->first]);
      }
      ++Galois::Runtime::evilPhase;
    }

    uint32_t find_hostID(uint64_t gid){
      for(uint32_t h = 0; h < base_hGraph::numHosts; ++h){
        if(gid >= gid2host[h].first && gid < gid2host[h].second)
          return h;
        else
          continue;
      }
      return -1;
    }

    uint32_t G2L(const uint64_t gid) const {
      return GlobalVec_map.at(gid);
    }

#if 0
    uint32_t G2L(uint64_t gid) const {
      //we can assume that GID exits and is unique. Index is localID since it is sorted.
      uint32_t found_index  = ~0;
      for(auto i : hostNodes){
        if(i.first != ~0){
          auto iter = std::lower_bound(GlobalVec.begin() + i.first, GlobalVec.begin() + i.second, gid);
          if(*iter == gid)
            return (iter - GlobalVec.begin());
        }
      }
      abort();
    }
#endif

    uint64_t L2G(uint32_t lid) const {
      return GlobalVec[lid];
    }


    template<typename GraphTy, typename std::enable_if<!std::is_void<typename GraphTy::edge_data_type>::value>::type* = nullptr>
      void loadEdges(GraphTy& graph, Galois::Graph::OfflineGraph& g) {
        fprintf(stderr, "Loading edge-data while creating edges.\n");
        uint64_t cur = 0;


        for(auto n = 0; n < base_hGraph::numOwned; ++n){
          auto gid = L2G(n);
          auto iter = std::lower_bound(GlobalVec_ordered.begin(), GlobalVec_ordered.end(), gid);
          uint32_t old_lid;
          assert(*iter == gid);
          if(*iter == gid){
            old_lid = (iter - GlobalVec_ordered.begin());
          }

          for (auto ii = g.edge_begin(old_lid), ee = g.edge_end(old_lid); ii < ee; ++ii) {
            auto gdst = g.getEdgeDst(ii);
            graph.constructEdge(cur++, G2L(GlobalVec_ordered[gdst]));
            auto gdata = g.getEdgeData<typename GraphTy::edge_data_type>(ii);
            graph.constructEdge(cur++, G2L(GlobalVec_ordered[gdst]), gdata);
          }
          graph.fixEndEdge(n, cur);
        }

#if 0
        for (auto n = g.begin(); n != g.end(); ++n) {

          for (auto ii = g.edge_begin(*n), ee = g.edge_end(*n); ii < ee; ++ii) {
            auto gdst = g.getEdgeDst(ii);
            auto gdata = g.getEdgeData<typename GraphTy::edge_data_type>(ii);
            graph.constructEdge(cur++, gdst, gdata);
          }
          graph.fixEndEdge((*n), cur);
        }
#endif
      }


    template<typename GraphTy, typename std::enable_if<std::is_void<typename GraphTy::edge_data_type>::value>::type* = nullptr>
      void loadEdges(GraphTy& graph) {
        fprintf(stderr, "Loading void edge-data while creating edges.\n");
        uint64_t cur = 0;

        //auto p = host_edges_map.begin();
        auto map_end = host_edges_map.end();
        for(auto l = 0; l < base_hGraph::numOwned; ++l){
          auto p = host_edges_map.find(L2G(l));
          if( p != map_end){
            for(auto n : (*p).second)
              graph.constructEdge(cur++, G2L(n));
          }
          graph.fixEndEdge(l, cur);
        }
      }

#if 0
    struct sort_dynamic_bitset{
      inline bool operator()(const boost::dynamic_bitset<uint32_t>& a_set, const boost::dynamic_bitset<uint32_t>& b_set){
        return a_set.count() < b_set.count();
      }
    };
    struct sort_dynamic_bitset{
      inline bool operator()(const Galois::DynamicBitSet<uint64_t>& a_set, const Galois::DynamicBitSet<uint64_t>& b_set) const{
        return a_set.bit_count() < b_set.bit_count();
      }
    };
#endif


    void fill_slaveNodes(std::vector<std::vector<size_t>>& slaveNodes){

      std::vector<std::vector<uint64_t>> GlobalVec_perHost(base_hGraph::numHosts);
      std::vector<std::vector<uint32_t>> OwnerVec_perHost(base_hGraph::numHosts);
      std::vector<uint64_t> nodesOnHost_vec;


      //Fill GlobalVec_perHost and slaveNodes vetors using assigned_edges_perhost.
      numEdges = 0;
      for(auto h = 0; h < base_hGraph::numHosts; ++h){
        for(auto i = 0; i  < assigned_edges_perhost[h].size(); i += 2){
          host_edges_map[assigned_edges_perhost[h][i]].push_back(assigned_edges_perhost[h][i + 1]);
          nodesOnHost_vec.push_back(assigned_edges_perhost[h][i]);
          nodesOnHost_vec.push_back(assigned_edges_perhost[h][i + 1]);
          numEdges++;
        }
      }


      /*base_hGraph::totalOwnedNodes = base_hGraph::numOwned = gid2host[base_hGraph::id].second - gid2host[base_hGraph::id].first;*/
      //Isolated nodes
      for(auto n = gid2host[base_hGraph::id].first; n < gid2host[base_hGraph::id].second; ++n){
        nodesOnHost_vec.push_back(n);
      }

      // Only keep unique node ids in vector
      std::sort(nodesOnHost_vec.begin(), nodesOnHost_vec.end());
      nodesOnHost_vec.erase(std::unique(nodesOnHost_vec.begin(), nodesOnHost_vec.end()), nodesOnHost_vec.end());

      numNodes = nodesOnHost_vec.size();

      for(auto i : nodesOnHost_vec){
        // Need to add both source and destination nodes
        // Source : i_pair.first
        // Destination : i_pair.second
        auto owner = find_hostID(i);
        GlobalVec_perHost[owner].push_back(i);
        OwnerVec_perHost[owner].push_back(owner);
        slaveNodes[owner].push_back(i);
      }


      hostNodes.resize(base_hGraph::numHosts);
      uint32_t counter = 0;
      for(auto i = 0; i < base_hGraph::numHosts; i++){
        if(GlobalVec_perHost[i].size() > 0){
          hostNodes[i] = std::make_pair(counter, GlobalVec_perHost[i].size() + counter);
          counter += GlobalVec_perHost[i].size();
        }
        else {
          hostNodes[i] = std::make_pair(~0, ~0);
        }
        std::sort(GlobalVec_perHost[i].begin(), GlobalVec_perHost[i].end());
        std::sort(OwnerVec_perHost[i].begin(), OwnerVec_perHost[i].end());
      }

      auto iter_insert = GlobalVec.begin();
      uint32_t c = 0;
      for(auto v : GlobalVec_perHost){
        for(auto j : v){
          GlobalVec.push_back(j);
        }
      }

      //trasfer to unordered_map
      uint32_t local_id = 0;
      for(auto v : GlobalVec){
        GlobalVec_map[v] = local_id;
        ++local_id;
      }


      OwnerVec.reserve(counter);
      c = 0;
      iter_insert = OwnerVec.begin();
      for(auto v : OwnerVec_perHost){
        for(auto j : v){
          OwnerVec.push_back(j);
        }
      }
      base_hGraph::totalOwnedNodes = GlobalVec_perHost[base_hGraph::id].size();
    }

    bool is_vertex_cut() const{
      return true;
    }

    uint64_t get_local_total_nodes() const {
      return (base_hGraph::numOwned);
    }

#if 0
    void save_meta_file(std::string file_name_prefix) const {
      std::string meta_file_str = file_name_prefix +".gr.META." + std::to_string(base_hGraph::id) + ".OF." + std::to_string(base_hGraph::numHosts);
      std::string tmp_meta_file_str = file_name_prefix +".gr.TMP." + std::to_string(base_hGraph::id) + ".OF." + std::to_string(base_hGraph::numHosts);
      std::ofstream meta_file(meta_file_str.c_str());
      std::ofstream tmp_file;
      tmp_file.open(tmp_meta_file_str.c_str());

      size_t num_nodes = (size_t)base_hGraph::numOwned;
      std::cerr << base_hGraph::id << "  NUMNODES  : " <<  num_nodes << "\n";
      meta_file.write(reinterpret_cast<char*>(&num_nodes), sizeof(num_nodes));
      for(size_t lid = 0; lid < base_hGraph::numOwned; ++lid){

      size_t gid = L2G(lid);
      size_t owner = getOwner_lid(lid);
#if 0
      //for(auto src = base_hGraph::graph.begin(), src_end = base_hGraph::graph.end(); src != src_end; ++src){
        size_t lid = (size_t)(*src);
        size_t gid = (size_t)L2G(*src);
        size_t owner = (size_t)getOwner_lid(lid);
#endif
        meta_file.write(reinterpret_cast<char*>(&gid), sizeof(gid));
        meta_file.write(reinterpret_cast<char*>(&lid), sizeof(lid));
        meta_file.write(reinterpret_cast<char*>(&owner), sizeof(owner));

        tmp_file << gid << " " << lid << " " << owner << "\n";
      }
    }
#endif

};

