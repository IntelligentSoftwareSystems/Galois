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
#include "Galois/Runtime/dGraph_edgeAssign_policy.h"

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



    //XXX: initialize to ~0
    std::vector<std::vector<uint64_t>> node_mapping;
    std::vector<uint64_t> numNodes_per_host;
    std::vector<uint64_t> Nodes_isolated;
    std::vector<std::vector<uint64_t>> master_mapping;
    std::vector<std::vector<uint64_t>> slave_mapping;
    std::vector<std::pair<uint64_t, uint64_t>> host_edges;
    //std::vector<std::vector<bool>> gid_bitVector(g.size(), std::vector<bool>(base_hGraph::numHosts, false));
    std::vector<boost::dynamic_bitset<uint32_t>>gid_bitset;
    std::vector<uint64_t> numEdges_per_host;




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
      return gid >= globalOffset && gid < globalOffset + base_hGraph::numOwned;
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

      Galois::Statistic statGhostNodes("TotalGhostNodes");
      //id = _id;
      //numHosts = _numHosts;

      //std::string part_fileName = getPartitionFileName(partitionFolder, base_hGraph::id, base_hGraph::numHosts);
      //std::string part_metaFile = getMetaFileName(partitionFolder, base_hGraph::id, base_hGraph::numHosts);


      Galois::Graph::OfflineGraph g(filename);

      node_mapping.resize(base_hGraph::numHosts);
      numNodes_per_host.resize(base_hGraph::numHosts);
      master_mapping.resize(base_hGraph::numHosts);
      slave_mapping.resize(base_hGraph::numHosts);
      numEdges_per_host.resize(base_hGraph::numHosts, 0);
      gid_bitset.resize(g.size(), boost::dynamic_bitset<uint32_t>(base_hGraph::numHosts));

      std::cout << "SIZE OF EACH BITSET : " << gid_bitset[0].size() << "\n";



      /********** vertex cut begins *******************/

      srand(1);
      for(auto src = g.begin(); src != g.end(); ++src){
        for(auto iter_edge = g.edge_begin(*src); iter_edge != g.edge_end(*src); ++iter_edge){
          auto dst = g.getEdgeDst(iter_edge);

          //auto assigned_host = balanced_edge_assignment(*src, dst, gid_bitset[*src], gid_bitset[dst]);
          auto assigned_host = random_edge_assignment(*src, dst, gid_bitset[*src], gid_bitset[dst], base_hGraph::numHosts);

          assert(assigned_host < base_hGraph::numHosts);

          // my edge to be constructed later
          if(assigned_host == base_hGraph::id){
            host_edges.push_back(std::make_pair(*src, dst));
          }

          // gid_bitset updated
          //if(!gid_bitset[*src][assigned_host]){
          if(!gid_bitset[*src].test(assigned_host)){
            gid_bitset[*src][assigned_host] = 1;
            ++numNodes_per_host[assigned_host];
            //node_mapping[assigned_host].push_back(*src);
          }

          //if(!gid_bitset[dst][assigned_host]){
          if(!gid_bitset[dst].test(assigned_host)){
            gid_bitset[dst][assigned_host] = 1;
            //node_mapping[assigned_host].push_back(dst);
            ++numNodes_per_host[assigned_host];
          }
          ++numEdges_per_host[assigned_host];
        }
      }


        // Assigning isolated nodes
        for(auto k = gid_bitset.begin(); k != gid_bitset.end(); ++k){
          if((*k).none()){
            uint32_t assigned_host = 0;
            for(auto h = 1; h < base_hGraph::numHosts; ++h){
              if(numNodes_per_host[h] < numNodes_per_host[assigned_host])
                assigned_host = h;
            }
            (*k)[assigned_host] = 1;
            if(assigned_host == base_hGraph::id){
              Nodes_isolated.push_back(std::distance(gid_bitset.begin(), k));
              ++numNodes_per_host[assigned_host];
            }
          }
        }

      base_hGraph::totalNodes = g.size();
      std::cerr << "[" << base_hGraph::id << "] Total nodes : " << base_hGraph::totalNodes << "\n";

      //compute owners for all nodes
      base_hGraph::numOwned = numNodes_per_host[base_hGraph::id];

      std::cerr << "[" << base_hGraph::id << "] Owned nodes : " << base_hGraph::numOwned << "\n";

      assert(host_edges.size() == numEdges_per_host[base_hGraph::id]);

      uint64_t _numEdges = host_edges.size();
      std::cerr << "[" << base_hGraph::id << "] Total edges : " << _numEdges << "\n";

      uint32_t _numNodes = base_hGraph::numOwned;

      base_hGraph::graph.allocateFrom(_numNodes, _numEdges);
      //std::cerr << "Allocate done\n";

      base_hGraph::graph.constructNodes();
      //std::cerr << "Construct nodes done\n";

      fill_slaveNodes(base_hGraph::slaveNodes);

      // sort edges based on sources
      std::sort(host_edges.begin(), host_edges.end(), [&](std::pair<uint64_t, uint64_t>& p1, std::pair<uint64_t, uint64_t>& p2){ return (G2L(p1.first) < G2L(p2.first)) ;});

      loadEdges(base_hGraph::graph, g);
      std::cerr << "Edges loaded \n";

      base_hGraph::setup_communication();

    }

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
      void loadEdges(GraphTy& graph, Galois::Graph::OfflineGraph& g) {
        fprintf(stderr, "Loading void edge-data while creating edges.\n");
        uint64_t cur = 0;

        auto p = host_edges.begin();
        for(auto l = 0; l < base_hGraph::numOwned; ++l){
            while(L2G(l) == (*p).first){
              graph.constructEdge(cur++, G2L((*p).second));
              ++p;
            }
          graph.fixEndEdge(l, cur);
        }
      }

    struct sort_dynamic_bitset{
      inline bool operator()(const boost::dynamic_bitset<uint32_t>& a_set, const boost::dynamic_bitset<uint32_t>& b_set){
        return a_set.count() < b_set.count();
      }
    };

    void fill_slaveNodes(std::vector<std::vector<size_t>>& slaveNodes){

      std::vector<std::vector<uint64_t>> GlobalVec_perHost(base_hGraph::numHosts);
      std::vector<std::vector<uint32_t>> OwnerVec_perHost(base_hGraph::numHosts);

      // To keep track of the masters assinged.
      std::vector<uint32_t> master_load(base_hGraph::numHosts);
      // To preserve the old indicies
      std::vector<uint64_t> old_index(gid_bitset.size());
      std::iota(old_index.begin(), old_index.end(), 0);

      std::sort(old_index.begin(), old_index.end(), [&](uint64_t i1, uint64_t i2) {return (gid_bitset[i1].count() <  gid_bitset[i2].count());});

      //sort vector of dynamic_bitset to select masters and slaves.
      std::sort(gid_bitset.begin(), gid_bitset.end(), sort_dynamic_bitset());

      uint64_t current_index = 0;
      for(auto bit_set = gid_bitset.begin(); bit_set != gid_bitset.end(); ++bit_set, ++current_index){
        //must be assigned to some host.
        auto num_set_bits = (*bit_set).count();
        assert(num_set_bits > 0);

        uint32_t first_set_pos = (*bit_set).find_first();
        uint32_t owner = first_set_pos;
        uint32_t next_set_pos = first_set_pos;

        if(num_set_bits > 1){
          for(auto n = 1; n < num_set_bits; ++n){
            next_set_pos = (*bit_set).find_next(next_set_pos) ;
            if(master_load[owner] > master_load[next_set_pos]){
              owner = next_set_pos;
            }
          }

          assert(owner < base_hGraph::numHosts);

        }
        ++master_load[owner];
        assert(owner < base_hGraph::numHosts);


        if((*bit_set).test(base_hGraph::id)){
          GlobalVec_perHost[owner].push_back(old_index[current_index]);
          OwnerVec_perHost[owner].push_back(owner);
          slaveNodes[owner].push_back(old_index[current_index]);
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
      }

#if 0
      if(base_hGraph::id == 0){
        for(auto i = 0; i < base_hGraph::numHosts; i++){
          std::cout << base_hGraph::id << " : hostNode " << hostNodes[i].first  << ", " << hostNodes[i].second << "\n";
        }
      }
#endif
      GlobalVec.reserve(counter);
      auto iter_insert = GlobalVec.begin();
        uint32_t c = 0;
        for(auto v : GlobalVec_perHost){
          for(auto j : v){
            GlobalVec.push_back(j);
          }
        }

        OwnerVec.reserve(counter);
        c = 0;
        iter_insert = OwnerVec.begin();
        for(auto v : OwnerVec_perHost){
          for(auto j : v){
            OwnerVec.push_back(j);
          }
        }
    }

    bool is_vertex_cut() const{
      return true;
    }

    uint64_t get_local_total_nodes() const {
      return (base_hGraph::numOwned);
    }
};

