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

class DS_vertexCut {

  public:
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
    std::vector<size_t> LocalVec; //Local Id's sorted vector.



    //OfflineGraph* g;

    uint64_t globalOffset;
    uint32_t numOwned;
    uint32_t numNodes;
    uint32_t id;
    uint32_t numHosts;

    unsigned getHostID(uint64_t gid) {
      auto lid = G2L(gid);
      return OwnerVec[lid];
    }

    bool isOwned(uint64_t gid) const {
      return gid >= globalOffset && gid < globalOffset + numOwned;
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

    std::string getPartitionFileName(const std::string& filename, const std::string & basename, unsigned hostID, unsigned num_hosts){
      std::string result = basename;
      result+= ".PART.";
      result+=std::to_string(hostID);
      result+= ".OF.";
      result+=std::to_string(num_hosts);
      return result;
    }


    void hGraph(OfflineGraph& g, const std::string& filename, const std::string& partitionFolder,unsigned host, unsigned _numHosts, std::vector<unsigned> scalefactor, uint32_t& _numNodes, uint32_t& _numOwned,uint64_t& _numEdges, uint64_t& _totalNodes, unsigned _id ){

      Galois::Statistic statGhostNodes("TotalGhostNodes");
      id = _id;
      numHosts = _numHosts;

      //std::string part_fileName = getPartitionFileName(partitionFolder,id,numHosts);
      std::string part_metaFile = getMetaFileName(partitionFolder, id, numHosts);

      //g = new OfflineGraph(part_fileName);

      _totalNodes = g.size();
      std::cerr << "[" << id << "] Total nodes : " << _totalNodes << "\n";
      readMetaFile(part_metaFile, localToGlobalMap_meta);

      //compute owners for all nodes
      numOwned = _numOwned = g.size();

      _numEdges = g.edge_begin(*(g.end())) - g.edge_begin(*(g.begin())); // depends on Offline graph impl
      std::cerr << "[" << id << "] Total edges : " << _numEdges << "\n";

      numNodes = _numNodes = numOwned;

    }

    uint32_t G2L(uint64_t gid) const {
      //we can assume that GID exits and is unique. Index is localID since it is sorted.
      auto iter = std::lower_bound(GlobalVec.begin(), GlobalVec.end(), gid);
      assert(*iter == gid);
      if(*iter == gid)
        return (iter - GlobalVec.begin());
      else
        abort();
    }

    uint64_t L2G(uint32_t lid) const {
      return GlobalVec[lid];
    }


    template<typename GraphTy, typename std::enable_if<!std::is_void<typename GraphTy::edge_data_type>::value>::type* = nullptr>
      void loadEdges(GraphTy& graph, OfflineGraph& g) {
        fprintf(stderr, "Loading edge-data while creating edges.\n");
        uint64_t cur = 0;
        for (auto n = g.begin(); n != g.end(); ++n) {
          for (auto ii = g.edge_begin(*n), ee = g.edge_end(*n); ii < ee; ++ii) {
            auto gdst = g.getEdgeDst(ii);
            auto gdata = g.getEdgeData<typename GraphTy::edge_data_type>(ii);
            graph.constructEdge(cur++, gdst, gdata);
          }
          graph.fixEndEdge((*n), cur);
        }
      }


    template<typename GraphTy, typename std::enable_if<std::is_void<typename GraphTy::edge_data_type>::value>::type* = nullptr>
      void loadEdges(GraphTy& graph, OfflineGraph& g) {
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


    void fill_mirrorNodes(std::vector<std::vector<size_t>>& mirrorNodes){
      for(auto info : localToGlobalMap_meta){
        assert(info.owner_id >= 0 && info.owner_id < numHosts);
        mirrorNodes[info.owner_id].push_back(info.global_id);

        GlobalVec.push_back(info.global_id);
        OwnerVec.push_back(info.owner_id);
        LocalVec.push_back(info.local_id);
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
    }

    bool is_vertex_cut() const{
      return true;
    }
};

