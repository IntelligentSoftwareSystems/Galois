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


    //EXPERIMENT
    std::unordered_map<uint64_t, uint32_t> GlobalVec_map;

    //XXX: initialize to ~0
    //std::vector<std::vector<uint64_t>> node_mapping;
    std::vector<uint64_t> numNodes_per_host;
    //std::vector<uint64_t> Nodes_isolated;
    //std::vector<std::vector<uint64_t>> master_mapping;
    //std::vector<std::vector<uint64_t>> mirror_mapping;
    //std::unordered_map<uint64_t, std::vector<uint64_t>> host_edges_map;
    std::unordered_map<uint64_t, std::vector<uint64_t>> host_edges_map;
    //std::vector<std::vector<bool>> gid_bitVector(g.size(), std::vector<bool>(base_hGraph::numHosts, false));
    //std::vector<boost::dynamic_bitset<uint32_t>>gid_bitset;
    //std::vector<Galois::DynamicBitSet<uint64_t>>gid_bitset;
    //std::vector<std::vector<bool>>gid_bitset_vecBool;
    //std::vector<bool> gid_bitset_oneVec;
    std::vector<uint64_t> numEdges_per_host;

    Galois::VecBool gid_vecBool;



    //OfflineGraph* g;

    uint64_t globalOffset;
    //uint32_t numOwned;
    uint32_t numNodes;
    //uint32_t id;
    //uint32_t numHosts;

    unsigned getHostID(uint64_t gid) const {
      auto lid = G2L(gid);
      return OwnerVec[lid];
    }

    size_t getOwner_lid(size_t lid) const {
      return OwnerVec[lid];
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


    hGraph_vertexCut(const std::string& filename, const std::string& partitionFolder,unsigned host, unsigned _numHosts, std::vector<unsigned> scalefactor) :  base_hGraph(host, _numHosts) {

      Galois::Runtime::reportStat("(NULL)", "ONLINE VERTEX CUT", 0, 0);

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

      {

      StatTimer_graph_construct.start();
      Galois::Graph::FileGraph g;
      g.fromFile(filename);

      numNodes_per_host.resize(base_hGraph::numHosts);
      numEdges_per_host.resize(base_hGraph::numHosts, 0);
      gid_vecBool.resize(g.size() , base_hGraph::numHosts);

      /********** vertex cut begins *******************/
      srand(1);

      base_hGraph::totalNodes = g.size();

      std::cout << "Start loop over nodes\n";



      StatTimer_distributed_edges.start();
      //for(auto src = g.begin(), src_end= g.end(); src != src_end; ++src){
      for(auto src = g.begin(), src_end= g.end(); src != src_end; ++src){


      //StatTimer_distributed_edges_inner_loop.start();
    
      //StatTimer_distributed_edges_get_edges.start();
        auto e_start = g.edge_begin(*src);
        auto e_end = g.edge_end(*src);
      //StatTimer_distributed_edges_get_edges.stop();

        for(auto e = e_start; e != e_end; ++e){
	
          //StatTimer_distributed_edges_get_dst.start();
          auto dst = g.getEdgeDst(e);
          //StatTimer_distributed_edges_get_dst.stop();
          //auto assigned_host = random_edge_assignment(*src, dst, gid_vecBool, base_hGraph::numHosts);
          //StatTimer_distributed_edges_policy.start();
          auto assigned_host = balanced_edge_assignment(*src, dst, gid_vecBool, base_hGraph::numHosts, numEdges_per_host);
          //StatTimer_distributed_edges_policy.stop();
          //auto assigned_host = rand() % base_hGraph::numHosts;

          assert(assigned_host < base_hGraph::numHosts);

          //StatTimer_distributed_edges_map.start();
          // my edge to be constructed later
          if(assigned_host == base_hGraph::id){
            host_edges_map[*src].push_back(dst);
            ++base_hGraph::numOwned_edges;
          }
          //StatTimer_distributed_edges_map.stop();

          //StatTimer_distributed_edges_test_set_bit.start();
#if 0
          if(!gid_vecBool.is_set(*src, assigned_host)){
            gid_vecBool.set_bit(*src, assigned_host);
            ++numNodes_per_host[assigned_host];
            tmp_file << *src << " " << gid_vecBool.bit_count(*src) << " " << numNodes_per_host[assigned_host] << "\n";
          }

          if(!gid_vecBool.is_set(dst, assigned_host)){
            gid_vecBool.set_bit(dst, assigned_host);
            ++numNodes_per_host[assigned_host];
            tmp_file << dst << " " << gid_vecBool.bit_count(dst) << " " << numNodes_per_host[assigned_host] << "\n";
          }
#endif

          if(!gid_vecBool.set_bit_and_return(*src, assigned_host)){
            ++numNodes_per_host[assigned_host];
          }

          if(!gid_vecBool.set_bit_and_return(dst, assigned_host)){
            ++numNodes_per_host[assigned_host];
          }

          ++numEdges_per_host[assigned_host];
          //StatTimer_distributed_edges_test_set_bit.stop();
        }
      //StatTimer_distributed_edges_inner_loop.stop();

      }

      }



      //std::stringstream ss;
      //ss << "[" << base_hGraph::id << "] numNodes assigned :" << numNodes_per_host[base_hGraph::id] << "\n";
      //std::cout << ss.str();


      StatTimer_distributed_edges.stop();
      //std::cerr << "TOTAL_EDGES : " <<total_edges << "\n";

        // Assigning isolated nodes
        for(auto k = 0; k < base_hGraph::totalNodes; ++k){
          if(gid_vecBool.bit_count(k) == 0){
            ++base_hGraph::total_isolatedNodes;
            uint32_t assigned_host = 0;
            for(auto h = 1; h < base_hGraph::numHosts; ++h){
              if(numNodes_per_host[h] < numNodes_per_host[assigned_host])
                assigned_host = h;
            }
            gid_vecBool.set_bit(k, assigned_host);
            ++numNodes_per_host[assigned_host];
          }
        }


      std::cerr << "Done assigning isolated nodes\n";

      std::stringstream ss1;
      ss1 << "[" << base_hGraph::id << "] numNodes assigned After :" << numNodes_per_host[base_hGraph::id] << "\n";
      std::cout << ss1.str();

      std::cerr << "[" << base_hGraph::id << "] Total nodes : " << base_hGraph::totalNodes << "\n";

      //compute owners for all nodes
      base_hGraph::numOwned = numNodes_per_host[base_hGraph::id];
      uint32_t _numNodes = base_hGraph::numOwned;
      Galois::Runtime::reportStat("(NULL)", "OWNED_NODES", numNodes_per_host[base_hGraph::id], 0);
      //std::cerr << "[" << base_hGraph::id << "] Owned nodes : " << base_hGraph::numOwned << "\n";


      uint64_t _numEdges = numEdges_per_host[base_hGraph::id];
      Galois::Runtime::reportStat("(NULL)", "OWNED_EDGES", _numEdges, 0);
      //std::cerr << "[" << base_hGraph::id << "] Total edges : " << _numEdges << "\n";

      base_hGraph::graph.allocateFrom(_numNodes, _numEdges);
      std::cerr << "Allocate done\n";

      base_hGraph::graph.constructNodes();
      std::cerr << "Construct nodes done\n";

      fill_mirrorNodes(base_hGraph::mirrorNodes);
      std::cerr << "fill_mirrorNodes done\n";

      // free the memory 
      gid_vecBool.clear();
      std::vector<uint64_t>().swap(numEdges_per_host);
      std::vector<uint64_t>().swap(numNodes_per_host);


      Galois::StatTimer StatTimer_load_edges("TIMER_LOAD_EDGES");

      StatTimer_load_edges.start();
      loadEdges(base_hGraph::graph);
      StatTimer_load_edges.stop();

      StatTimer_graph_construct.stop();

      std::cerr << base_hGraph::id << "Edges loaded \n";

      StatTimer_graph_construct_comm.start();
      base_hGraph::setup_communication();
      StatTimer_graph_construct_comm.stop();

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


    void fill_mirrorNodes(std::vector<std::vector<size_t>>& mirrorNodes){

      std::vector<std::vector<uint64_t>> GlobalVec_perHost(base_hGraph::numHosts);
      std::vector<std::vector<uint32_t>> OwnerVec_perHost(base_hGraph::numHosts);

      // To keep track of the masters assinged.
      std::vector<uint32_t> master_load(base_hGraph::numHosts);
      // To preserve the old indicies
      std::vector<uint64_t> old_index(base_hGraph::totalNodes);
      std::iota(old_index.begin(), old_index.end(), 0);

      // sort by replication factor
      std::sort(old_index.begin(), old_index.end(), [&](uint64_t i1, uint64_t i2) {return (gid_vecBool.bit_count(i1) <  gid_vecBool.bit_count(i2));});

      uint64_t current_index = 0;
      for(auto i : old_index){
        ++current_index;
        //must be assigned to some host.
        auto num_set_bits = gid_vecBool.bit_count(i);
        assert(num_set_bits > 0);

        uint32_t first_set_pos = gid_vecBool.find_first(i);
        assert(first_set_pos != ~0);
        uint32_t owner = first_set_pos;
        uint32_t next_set_pos = first_set_pos;

        if(num_set_bits > 1){
          for(auto n = 1; n < num_set_bits; ++n){
            next_set_pos = gid_vecBool.find_next(i,next_set_pos) ;
            assert(next_set_pos != ~0);
            if(master_load[owner] > master_load[next_set_pos]){
              owner = next_set_pos;
            }
          }

          assert(owner < base_hGraph::numHosts);

        }
        ++master_load[owner];
        assert(owner < base_hGraph::numHosts);


        if(gid_vecBool.is_set(i, base_hGraph::id)){
          GlobalVec_perHost[owner].push_back(i);
          OwnerVec_perHost[owner].push_back(owner);
          mirrorNodes[owner].push_back(i);
        }
      }

#if 0
      //sort per host global vector for G2L
      for(auto i = 0; i < base_hGraph::numHosts; ++i){
        std::sort(GlobalVec_perHost[i].begin(), GlobalVec_perHost[i].end());
      }
#endif

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


      GlobalVec.reserve(counter);
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
      base_hGraph::totalOnwedNodes = GlobalVec_perHost[base_hGraph::id].size();
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

