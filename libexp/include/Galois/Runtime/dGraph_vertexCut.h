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

    bool isOwned(uint64_t gid) const {
      return (getHostID(gid) == base_hGraph::id);
    }

    bool isLocal(uint64_t gid) const {
      return (GlobalVec_map.find(gid) != GlobalVec_map.end());
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

    hGraph_vertexCut(const std::string& filename, const std::string& partitionFolder,unsigned host, unsigned _numHosts, std::vector<unsigned> scalefactor) :  base_hGraph(host, _numHosts) {


      Galois::Runtime::reportStat("(NULL)", "VERTEX CUT", 0, 0);
      Galois::Statistic statGhostNodes("TotalGhostNodes");
      Galois::StatTimer StatTimer_graph_construct("TIME_GRAPH_CONSTRUCT");
      Galois::StatTimer StatTimer_graph_construct_comm("TIME_GRAPH_CONSTRUCT_COMM");
      //id = _id;
      //numHosts = _numHosts;

      StatTimer_graph_construct.start();
      std::string part_fileName = getPartitionFileName(partitionFolder, base_hGraph::id, base_hGraph::numHosts);
      std::string part_metaFile = getMetaFileName(partitionFolder, base_hGraph::id, base_hGraph::numHosts);

      Galois::Graph::OfflineGraph g(part_fileName);
      Galois::Graph::OfflineGraph g_baseFile(filename);

      base_hGraph::totalNodes = g_baseFile.size();
      std::cerr << "[" << base_hGraph::id << "] Total nodes : " << base_hGraph::totalNodes << "\n";
      readMetaFile(part_metaFile, localToGlobalMap_meta);

      //compute owners for all nodes
      base_hGraph::numOwned = g.size();

      uint64_t _numEdges = g.edge_begin(*(g.end())) - g.edge_begin(*(g.begin())); // depends on Offline graph impl
      std::cerr << "[" << base_hGraph::id << "] Total edges : " << _numEdges << "\n";

      uint32_t _numNodes = base_hGraph::numOwned;

      base_hGraph::graph.allocateFrom(_numNodes, _numEdges);
      //std::cerr << "Allocate done\n";

      base_hGraph::graph.constructNodes();
      //std::cerr << "Construct nodes done\n";
      fill_slaveNodes(base_hGraph::slaveNodes);

      loadEdges(base_hGraph::graph, g);
      std::cerr <<"[" << base_hGraph::id << "] Edges loaded \n";
      StatTimer_graph_construct.stop();

      StatTimer_graph_construct_comm.start();
      base_hGraph::setup_communication();
      StatTimer_graph_construct_comm.stop();
    }


    uint32_t G2L(uint64_t gid) const {
      return GlobalVec_map.at(gid);
    }
#if 0
    uint32_t G2L(uint64_t gid) const {
      //we can assume that GID exits and is unique. Index is localID since it is sorted.
      for(auto i : hostNodes){
        if(i.first != ~0U){
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

          for(auto ii = g.edge_begin(old_lid), ee = g.edge_end(old_lid); ii < ee; ++ii) {
            auto gdst = g.getEdgeDst(ii);
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
      void loadEdges(GraphTy& graph, Galois::Graph::OfflineGraph& g) {
        fprintf(stderr, "Loading void edge-data while creating edges.\n");
        uint64_t cur = 0;
#if 0
        std::string output_graph = "graph_" + std::to_string(base_hGraph::id) + ".edglist";
        std::ofstream graph_file;
        graph_file.open(output_graph.c_str());

        for(auto n = g.begin(); n != g.end(); ++n){
          for(auto e = g.edge_begin(*n); e != g.edge_end(*n); e++){
            auto dst = g.getEdgeDst(e);
            graph_file << L2G(*n) << "\t" << L2G(dst) << "\n";
          }
        }

        Galois::Runtime::getHostBarrier().wait();
#endif
        assert(g.size() == GlobalVec_ordered.size());

        for(auto n = 0U; n < base_hGraph::numOwned; ++n){
          auto gid = L2G(n);
          auto iter = std::lower_bound(GlobalVec_ordered.begin(), GlobalVec_ordered.end(), gid);
          uint32_t old_lid = ~0;
          assert(*iter == gid);
          if(*iter == gid){
            old_lid = (iter - GlobalVec_ordered.begin());
          }
          assert(old_lid < g.size());
          for (auto ii = g.edge_begin(old_lid), ee = g.edge_end(old_lid); ii < ee; ++ii) {
            auto gdst = g.getEdgeDst(ii);
            graph.constructEdge(cur++, G2L(GlobalVec_ordered[gdst]));
          }
          graph.fixEndEdge(n, cur);
        }
      }

    void fill_slaveNodes(std::vector<std::vector<size_t>>& slaveNodes){

      std::vector<std::vector<size_t>> GlobalVec_perHost(base_hGraph::numHosts);
      std::vector<std::vector<size_t>> OwnerVec_perHost(base_hGraph::numHosts);
      std::vector<std::vector<size_t>> LocalVec_perHost(base_hGraph::numHosts);

      for(auto info : localToGlobalMap_meta){
        assert(info.owner_id >= 0 && info.owner_id < base_hGraph::numHosts);
        slaveNodes[info.owner_id].push_back(info.global_id);

        GlobalVec_ordered.push_back(info.global_id);
        GlobalVec_perHost[info.owner_id].push_back(info.global_id);
        OwnerVec_perHost[info.owner_id].push_back(info.owner_id);
        LocalVec_perHost[info.owner_id].push_back(info.local_id);

        //std::cerr << "[" << base_hGraph::id << "]" << " G : " << info.global_id << " , L : " << info.local_id << " , O : " << info.owner_id << "\n";
      }

      std::cerr << "[ " << base_hGraph::id <<"] : OWNED : " << GlobalVec_perHost[base_hGraph::id].size() << "\n";

      base_hGraph::totalOwnedNodes = GlobalVec_perHost[base_hGraph::id].size();

      assert(std::is_sorted(GlobalVec_ordered.begin(), GlobalVec_ordered.end()));

      hostNodes.resize(base_hGraph::numHosts);
      uint32_t counter = 0;
      for(auto i = 0U; i < base_hGraph::numHosts; i++){
        if(GlobalVec_perHost[i].size() > 0){
          hostNodes[i] = std::make_pair(counter, GlobalVec_perHost[i].size() + counter);
          counter += GlobalVec_perHost[i].size();
        }
        else {
          hostNodes[i] = std::make_pair(~0, ~0);
        }
      }

      GlobalVec.reserve(counter);
      auto iter_insert = GlobalVec.begin();
      //uint32_t c = 0;
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
      //c = 0;
      iter_insert = OwnerVec.begin();
      for(auto v : OwnerVec_perHost){
        for(auto j : v){
          OwnerVec.push_back(j);
        }
      }

#if 0
       std::cout << "[" << base_hGraph::id << "] : Global size :"  << GlobalVec.size() << "\n";
      for(auto k : hostNodes){
        std::cout << "[" << base_hGraph::id << "] : " << k.first << ", " << k.second << "\n";
      }
#endif

      assert(counter == GlobalVec.size());
#if 0
      for(auto j = 0; j < GlobalVec_perHost.size(); ++j){
        if(!std::is_sorted(GlobalVec_perHost[j].begin(), GlobalVec_perHost[j].end())){
          std::cerr << "GlobalVec_perhost not sorted; Aborting execution for : " << j << "\n";
          abort();
        }
      }
#endif
      //Check to make sure GlobalVec is sorted. Everything depends on it.
      //assert(std::is_sorted(GlobalVec.begin(), GlobalVec.end()));
      for(auto h : hostNodes) {
        if(h.first != ~0U) {
          if(!std::is_sorted(GlobalVec.begin() + h.first , GlobalVec.begin() + h.second)){
            std::cerr << "GlobalVec not sorted; Aborting execution\n";
            abort();
          }
        }
      }

#if 0
      for(auto i = 0; i < base_hGraph::numOwned; ++i){
        assert( i == G2L(L2G(i)));
      }
#endif
    }

    bool is_vertex_cut() const{
      return true;
    }

    uint64_t get_local_total_nodes() const {
      return (base_hGraph::numOwned);
    }
};

