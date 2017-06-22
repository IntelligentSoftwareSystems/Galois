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


#define BATCH_MSG_SIZE 1000
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
    std::vector<uint64_t> GlobalVec; //Global Id's sorted vector.
    std::vector<std::pair<uint32_t, uint32_t>> hostNodes;

    std::vector<size_t> GlobalVec_ordered; //Global Id's sorted vector.
    // To send edges to different hosts: #Src #Dst
    std::vector<std::vector<uint64_t>> assigned_edges_perhost;
    std::vector<uint64_t> recv_assigned_edges;
    std::vector<uint64_t> num_assigned_edges_perhost;
    uint64_t num_total_edges_to_receive;


    //EXPERIMENT
    std::unordered_map<uint64_t, uint32_t> GlobalVec_map;

    //XXX: initialize to ~0
    std::vector<uint64_t> numNodes_per_host;

    //XXX: Use EdgeTy to determine if need to load edge weights or not.
    using Host_edges_map_type = typename std::conditional<!std::is_void<EdgeTy>::value, std::unordered_map<uint64_t, std::vector<std::pair<uint64_t, uint32_t>>> , std::unordered_map<uint64_t, std::vector<uint64_t>>>::type;
    Host_edges_map_type host_edges_map;
    //std::unordered_map<uint64_t, std::vector<uint64_t>> host_edges_map;
    std::vector<uint64_t> numEdges_per_host;
    std::vector<std::pair<uint64_t, uint64_t>> gid2host;
    std::vector<std::pair<uint64_t, uint64_t>> gid2host_withoutEdges;

    Galois::VecBool gid_vecBool;

    uint64_t globalOffset;
    uint32_t numNodes;
    bool isBipartite;
    uint64_t last_nodeID_withEdges_bipartite;
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
        if (i.first != (uint64_t)(~0)) {
          assert(i.first < GlobalVec.size());
          assert(i.second <= GlobalVec.size());
          auto iter = std::lower_bound(GlobalVec.begin() + i.first, GlobalVec.begin() + i.second, gid);
          if((iter != (GlobalVec.begin() + i.second)) && (*iter == gid)){
            return true;
          }
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


    hGraph_vertexCut(const std::string& filename, const std::string& partitionFolder,unsigned host, unsigned _numHosts, std::vector<unsigned> scalefactor, bool transpose = false, uint32_t VCutTheshold = 100, bool bipartite = false) :  base_hGraph(host, _numHosts) {

      Galois::Runtime::reportStat("(NULL)", "ONLINE VERTEX CUT PL", 0, 0);

      Galois::Statistic statGhostNodes("TotalGhostNodes");
      Galois::StatTimer StatTimer_graph_construct("TIME_GRAPH_CONSTRUCT");
      Galois::StatTimer StatTimer_graph_construct_comm("TIME_GRAPH_CONSTRUCT_COMM");
      Galois::StatTimer StatTimer_local_distributed_edges("TIMER_LOCAL_DISTRIBUTE_EDGES");
      Galois::StatTimer StatTimer_exchange_edges("TIMER_EXCHANGE_EDGES");
      Galois::StatTimer StatTimer_fill_local_slaveNodes("TIMER_FILL_LOCAL_SLAVENODES");
      Galois::StatTimer StatTimer_distributed_edges_test_set_bit("TIMER_DISTRIBUTE_EDGES_TEST_SET_BIT");
      Galois::StatTimer StatTimer_allocate_local_DS("TIMER_ALLOCATE_LOCAL_DS");
      Galois::StatTimer StatTimer_distributed_edges_get_edges("TIMER_DISTRIBUTE_EDGES_GET_EDGES");
      Galois::StatTimer StatTimer_distributed_edges_inner_loop("TIMER_DISTRIBUTE_EDGES_INNER_LOOP");
      Galois::StatTimer StatTimer_distributed_edges_next_src("TIMER_DISTRIBUTE_EDGES_NEXT_SRC");

      StatTimer_local_distributed_edges.start();
      Galois::Graph::OfflineGraph g(filename);
      isBipartite = bipartite;

      uint64_t numNodes_to_divide = 0;
      if (isBipartite) {
    	  for (uint64_t n = 0; n < g.size(); ++n){
    		  if(std::distance(g.edge_begin(n), g.edge_end(n))){
                  ++numNodes_to_divide;
                  last_nodeID_withEdges_bipartite = n;
    		  }
    	  }
      }
      else {
    	  numNodes_to_divide = g.size();
      }

      std::cout << "Nodes to divide : " <<  numNodes_to_divide << "\n";
      base_hGraph::totalNodes = g.size();
      base_hGraph::totalEdges = g.sizeEdges();
      std::cerr << "[" << base_hGraph::id << "] Total nodes : " << base_hGraph::totalNodes << " , Total edges : " << base_hGraph::totalEdges << "\n";
      //compute owners for all nodes
      if (scalefactor.empty() || (base_hGraph::numHosts == 1)) {
        for (unsigned i = 0; i < base_hGraph::numHosts; ++i)
          gid2host.push_back(Galois::block_range(0U, (unsigned)numNodes_to_divide, i, base_hGraph::numHosts));
      } else {
        assert(scalefactor.size() == base_hGraph::numHosts);
        unsigned numBlocks = 0;
        for (unsigned i = 0; i < base_hGraph::numHosts; ++i)
          numBlocks += scalefactor[i];
        std::vector<std::pair<uint64_t, uint64_t>> blocks;
        for (unsigned i = 0; i < numBlocks; ++i)
          blocks.push_back(Galois::block_range(0U, (unsigned)numNodes_to_divide, i, numBlocks));
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

      if(isBipartite){
    	  uint64_t numNodes_without_edges = (g.size() - numNodes_to_divide);
    	  for (unsigned i = 0; i < base_hGraph::numHosts; ++i){
    	    auto p = Galois::block_range(0U, (unsigned)numNodes_without_edges, i, base_hGraph::numHosts);
    	    std::cout << " last node : " << last_nodeID_withEdges_bipartite << ", " << p.first << " , " << p.second << "\n";
    	    gid2host_withoutEdges.push_back(std::make_pair(last_nodeID_withEdges_bipartite + p.first + 1, last_nodeID_withEdges_bipartite + p.second + 1));
          }
      }


      uint64_t numEdges_distribute = g.edge_begin(gid2host[base_hGraph::id].second) - g.edge_begin(gid2host[base_hGraph::id].first); // depends on Offline graph impl
      std::cerr << "[" << base_hGraph::id << "] Total edges to distribute : " << numEdges_distribute << "\n";


      std::stringstream ss_cout;


      /********************************************
       * Assign edges to the hosts using heuristics
       * and send/recv from other hosts.
       * ******************************************/
      ss_cout << base_hGraph::id << " : assign_send_receive_edges started\n";
      std::cerr << ss_cout.str();
      ss_cout.str(std::string());
      ss_cout.clear();
      StatTimer_exchange_edges.start();

      assign_send_receive_edges(g, numEdges_distribute, VCutTheshold);

      StatTimer_exchange_edges.stop();

      ss_cout << base_hGraph::id << " : assign_send_receive_edges done\n";
      std::cerr << ss_cout.str();
      ss_cout.str(std::string());
      ss_cout.clear();

      /*******************************************/


      /******************************************
       *Using the edges received from other hosts,
       *fill the local data structures.
       *****************************************/
      ss_cout << base_hGraph::id << " : fill_slaveNodes started\n";
      std::cerr << ss_cout.str();
      ss_cout.str(std::string());
      ss_cout.clear();
      StatTimer_fill_local_slaveNodes.start();

      fill_slaveNodes(base_hGraph::slaveNodes);

      StatTimer_fill_local_slaveNodes.stop();
      ss_cout << base_hGraph::id << " : fill_slaveNodes done\n";
      std::cerr << ss_cout.str();
      ss_cout.str(std::string());
      ss_cout.clear();

      /*****************************************/

      ss_cout << base_hGraph::id << " : numNodes : " << numNodes << " , numEdges : " << numEdges << "\n";

      base_hGraph::numOwned = numNodes;


      /******************************************
       * Allocate and construct the graph
       *****************************************/
      ss_cout << base_hGraph::id << " : Allocate local graph DS : start\n";
      std::cerr << ss_cout.str();
      ss_cout.str(std::string());
      ss_cout.clear();
      StatTimer_allocate_local_DS.start();

      base_hGraph::graph.allocateFrom(numNodes, numEdges);
      base_hGraph::graph.constructNodes();

      StatTimer_allocate_local_DS.stop();
      ss_cout << base_hGraph::id << " : Allocate local graph DS : done\n";
      std::cerr << ss_cout.str();
      ss_cout.str(std::string());
      ss_cout.clear();


      Galois::StatTimer StatTimer_load_edges("TIMER_LOAD_EDGES");

      /*****************************************
       * Load the edges in the local graph
       * constructed.
       ****************************************/
      StatTimer_load_edges.start();
      loadEdges(base_hGraph::graph);
      StatTimer_load_edges.stop();

      StatTimer_graph_construct.stop();

      ss_cout << base_hGraph::id << " : Edges loaded\n";
      std::cerr << ss_cout.str();
      ss_cout.str(std::string());
      ss_cout.clear();

      Galois::Runtime::getHostBarrier().wait();


      /*****************************************
       * Communication PreProcessing:
       * Exchange slaves and master nodes among
       * hosts
       ****************************************/
      ss_cout << base_hGraph::id << " : Setup communication start\n";
      std::cerr << ss_cout.str();
      ss_cout.str(std::string());
      ss_cout.clear();
      StatTimer_graph_construct_comm.start();

      base_hGraph::setup_communication();

      StatTimer_graph_construct_comm.stop();
      ss_cout << base_hGraph::id << " : Setup communication done\n";
      std::cerr << ss_cout.str();
      ss_cout.str(std::string());
      ss_cout.clear();

    }


    void assign_send_receive_edges(Galois::Graph::OfflineGraph& g, uint64_t numEdges_distribute, uint32_t VCutTheshold){

      assigned_edges_perhost.resize(base_hGraph::numHosts);
      num_assigned_edges_perhost.resize(base_hGraph::numHosts);

      std::stringstream ss_cout;
      ss_cout << base_hGraph::id << " : Assign_edges_phase1 and pre exchange started\n";
      std::cerr << ss_cout.str();
      ss_cout.str(std::string());
      ss_cout.clear();

      /****************************************
       * Going over edges to get initial
       * information to exchange among hosts.
       **************************************/
      assign_edges_phase1(g, numEdges_distribute, VCutTheshold);

      /****** Total edges to receive from other hosts ****/
      num_total_edges_to_receive = 0;

      /***********************************
       * Initial exchange of information
       ***********************************/
      pre_exchange_edges_messages();

      ss_cout << base_hGraph::id << " : Assign_edges_phase1 and pre exchange done\n";
      std::cerr << ss_cout.str();
      ss_cout.str(std::string());
      ss_cout.clear();


      ss_cout << base_hGraph::id << " : Galois::on_each : assign_send_edges and receive_edges started\n";
      std::cerr << ss_cout.str();
      ss_cout.str(std::string());
      ss_cout.clear();
      /*******************************************************
       * Galois:On_each loop for using multiple threads.
       * Thread 0 : Runs the assign_send_edges functions: To
       *            assign edges to hosts and send across.
       * Thread 1 : Runs the receive_edges functions: To
       *            edges assigned to this host by other hosts.
       *
       ********************************************************/
      Galois::on_each([&](unsigned tid, unsigned nthreads){
          if(tid == 0)
                assign_send_edges<EdgeTy>(g, numEdges_distribute, VCutTheshold);
          if((nthreads == 1) || (tid == 1))
                receive_edges();
          });

      ss_cout << base_hGraph::id << " : Galois::on_each : assign_send_edges and receive_edges done \n";
      std::cerr << ss_cout.str();
      ss_cout.str(std::string());
      ss_cout.clear();

      /************** Append the edges received from other hosts to the local vector *************/
      assigned_edges_perhost[base_hGraph::id].insert(assigned_edges_perhost[base_hGraph::id].begin(), recv_assigned_edges.begin(), recv_assigned_edges.end());

      ++Galois::Runtime::evilPhase;
    }

    // Just calculating the number of edges to send to other hosts
    void assign_edges_phase1(Galois::Graph::OfflineGraph& g, uint64_t numEdges_distribute, uint32_t VCutTheshold){
         //Go over assigned nodes and distribute edges.
        for(auto src = gid2host[base_hGraph::id].first; src != gid2host[base_hGraph::id].second; ++src){
        auto num_edges = std::distance(g.edge_begin(src), g.edge_end(src));
        if(num_edges > VCutTheshold){
          //Assign edges for high degree nodes to the destination
          for(auto ee = g.edge_begin(src), ee_end = g.edge_end(src); ee != ee_end; ++ee){
            auto gdst = g.getEdgeDst(ee);
            auto h = find_hostID(gdst);
            num_assigned_edges_perhost[h]++;

          }
        }
        else{
          //keep all edges with the source node
          for(auto ee = g.edge_begin(src), ee_end = g.edge_end(src); ee != ee_end; ++ee){
            num_assigned_edges_perhost[base_hGraph::id]++;
          }
        }
      }


      uint64_t check_numEdges = 0;
      for(uint32_t h = 0; h < base_hGraph::numHosts; ++h){
        std::cout << "from : " << base_hGraph::id << " to : " << h << " : edges assigned : " << assigned_edges_perhost[h].size() << "\n";
        check_numEdges += num_assigned_edges_perhost[h];
      }

      assert(check_numEdges == numEdges_distribute);
      }



    //Edge type is not void.
    template<typename GraphEdgeTy, typename std::enable_if<!std::is_void<GraphEdgeTy>::value>::type* = nullptr>
      void assign_send_edges(Galois::Graph::OfflineGraph& g, uint64_t numEdges_distribute, uint32_t VCutTheshold){

        auto& net = Galois::Runtime::getSystemNetworkInterface();
        //Go over assigned nodes and distribute edges.
        for(auto src = gid2host[base_hGraph::id].first; src != gid2host[base_hGraph::id].second; ++src){
          auto num_edges = std::distance(g.edge_begin(src), g.edge_end(src));
          if(num_edges > VCutTheshold){
            //Assign edges for high degree nodes to the destination
            for(auto ee = g.edge_begin(src), ee_end = g.edge_end(src); ee != ee_end; ++ee){
              auto gdst = g.getEdgeDst(ee);
              auto gdata = g.getEdgeData<EdgeTy>(ee);
              auto h = find_hostID(gdst);
              assigned_edges_perhost[h].push_back(src);
              assigned_edges_perhost[h].push_back(gdst);
              assigned_edges_perhost[h].push_back(gdata);
            }
          }
          else{
            //keep all edges with the source node
            for(auto ee = g.edge_begin(src), ee_end = g.edge_end(src); ee != ee_end; ++ee){
              auto gdst = g.getEdgeDst(ee);
              auto gdata = g.getEdgeData<EdgeTy>(ee);
              assigned_edges_perhost[base_hGraph::id].push_back(src);
              assigned_edges_perhost[base_hGraph::id].push_back(gdst);
              assigned_edges_perhost[base_hGraph::id].push_back(gdata);
            }
          }
          //send if reached the batch limit
          for(uint32_t h = 0; h < base_hGraph::numHosts; ++h){
            if(h == base_hGraph::id) continue;
            if(assigned_edges_perhost[h].size() >= 3*BATCH_MSG_SIZE){
              Galois::Runtime::SendBuffer b;
              uint32_t num_edge_sending = (assigned_edges_perhost[h].size()/3);
              Galois::Runtime::gSerialize(b, num_edge_sending , assigned_edges_perhost[h]);
              net.sendTagged(h, Galois::Runtime::evilPhase, b);
              assigned_edges_perhost[h].clear();
            }
          }
        }

        //send the remaining edges to hosts
        for(uint32_t h = 0; h < base_hGraph::numHosts; ++h){
          if(h == base_hGraph::id) continue;
          if(assigned_edges_perhost[h].size() > 0){
            Galois::Runtime::SendBuffer b;
            uint32_t num_edge_sending = (assigned_edges_perhost[h].size()/3);
            Galois::Runtime::gSerialize(b, num_edge_sending, assigned_edges_perhost[h]);
            net.sendTagged(h, Galois::Runtime::evilPhase, b);
            assigned_edges_perhost[h].clear();
          }
        }


        uint64_t check_numEdges = 0;
        for(uint32_t h = 0; h < base_hGraph::numHosts; ++h){
          std::cout << "from : " << base_hGraph::id << " to : " << h << " : edges assigned : " << assigned_edges_perhost[h].size() << "\n";
        check_numEdges += assigned_edges_perhost[h].size();
      }

      assert(check_numEdges == 3*numEdges_distribute);
      }

    //Edge type is void.
    template<typename GraphEdgeTy, typename std::enable_if<std::is_void<GraphEdgeTy>::value>::type* = nullptr>
      void assign_send_edges(Galois::Graph::OfflineGraph& g, uint64_t numEdges_distribute, uint32_t VCutTheshold){

        auto& net = Galois::Runtime::getSystemNetworkInterface();
        //Go over assigned nodes and distribute edges.
        for(auto src = gid2host[base_hGraph::id].first; src != gid2host[base_hGraph::id].second; ++src){
          auto num_edges = std::distance(g.edge_begin(src), g.edge_end(src));
          if(num_edges > VCutTheshold){
            //Assign edges for high degree nodes to the destination
            for(auto ee = g.edge_begin(src), ee_end = g.edge_end(src); ee != ee_end; ++ee){
              auto gdst = g.getEdgeDst(ee);
              auto h = find_hostID(gdst);
              assigned_edges_perhost[h].push_back(src);
              assigned_edges_perhost[h].push_back(gdst);
            }
          }
          else{
            //keep all edges with the source node
            for(auto ee = g.edge_begin(src), ee_end = g.edge_end(src); ee != ee_end; ++ee){
              auto gdst = g.getEdgeDst(ee);
              assigned_edges_perhost[base_hGraph::id].push_back(src);
              assigned_edges_perhost[base_hGraph::id].push_back(gdst);
            }
          }
        //send if reached the batch limit
        for(uint32_t h = 0; h < base_hGraph::numHosts; ++h){
          if(h == base_hGraph::id) continue;
          if(assigned_edges_perhost[h].size() >= 2*BATCH_MSG_SIZE){
            Galois::Runtime::SendBuffer b;
            uint32_t num_edge_sending = (assigned_edges_perhost[h].size()/2);
            Galois::Runtime::gSerialize(b, num_edge_sending, assigned_edges_perhost[h]);
            net.sendTagged(h, Galois::Runtime::evilPhase, b);
            //std::cerr << base_hGraph::id << " ] : sending : " << num_edge_sending << "\n";
            assigned_edges_perhost[h].clear();
          }
        }


        }

    //send the remaining edges to hosts
    for(uint32_t h = 0; h < base_hGraph::numHosts; ++h){
      if(h == base_hGraph::id) continue;
      if(assigned_edges_perhost[h].size() > 0){
        Galois::Runtime::SendBuffer b;
        uint32_t num_edge_sending = (assigned_edges_perhost[h].size()/2);
        Galois::Runtime::gSerialize(b, num_edge_sending, assigned_edges_perhost[h]);
        net.sendTagged(h, Galois::Runtime::evilPhase, b);
        std::cerr << base_hGraph::id << " ] : sending : " << num_edge_sending << "\n";
        assigned_edges_perhost[h].clear();
      }
    }



    uint64_t check_numEdges = 0;
    for(uint32_t h = 0; h < base_hGraph::numHosts; ++h){
      std::cout << "from : " << base_hGraph::id << " to : " << h << " : edges assigned : " << assigned_edges_perhost[h].size() << "\n";
      check_numEdges += assigned_edges_perhost[h].size();
    }
    assert(check_numEdges == 2*numEdges_distribute);
}



    void pre_exchange_edges_messages(){
      Galois::StatTimer StatTimer_exchange_edges("PRE_EXCHANGE_EDGES_TIME");
      //Galois::Runtime::getHostBarrier().wait(); // so that all hosts start the timer together

      auto& net = Galois::Runtime::getSystemNetworkInterface();

      //send and clear assigned_edges_perhost to receive from other hosts
      for (unsigned x = 0; x < net.Num; ++x) {
        if(x == base_hGraph::id) continue;

        Galois::Runtime::SendBuffer b;
        //uint64_t num_batch_ceiling = std::ceil(double(num_assigned_edges_perhost[x]/double(BATCH_MSG_SIZE));
        gSerialize(b, num_assigned_edges_perhost[x]);
        net.sendTagged(x, Galois::Runtime::evilPhase, b);
      }

      //receive
      for (unsigned x = 0; x < net.Num; ++x) {
        if(x == base_hGraph::id) continue;

        decltype(net.recieveTagged(Galois::Runtime::evilPhase, nullptr)) p;
        do {
          net.handleReceives();
          p = net.recieveTagged(Galois::Runtime::evilPhase, nullptr);
        } while(!p);

        uint64_t _num_batch_msgs = 0;
        Galois::Runtime::gDeserialize(p->second, _num_batch_msgs);
        num_total_edges_to_receive += _num_batch_msgs;
      }
      ++Galois::Runtime::evilPhase;
    }


    void receive_edges(){
      Galois::StatTimer StatTimer_exchange_edges("RECEIVE_EDGES_TIME");
      auto& net = Galois::Runtime::getSystemNetworkInterface();

      //receive the edges from other hosts
      while(num_total_edges_to_receive){

        decltype(net.recieveTagged(Galois::Runtime::evilPhase, nullptr)) p;
        do {
          net.handleReceives();
          p = net.recieveTagged(Galois::Runtime::evilPhase, nullptr);
        } while(!p);

        std::vector<uint64_t> _assigned_edges_perhost_local;
        uint32_t _num_received = 0;
        Galois::Runtime::gDeserialize(p->second, _num_received,_assigned_edges_perhost_local);
        recv_assigned_edges.insert(recv_assigned_edges.end(), _assigned_edges_perhost_local.begin(), _assigned_edges_perhost_local.end());
        num_total_edges_to_receive -= _num_received;
      }
    }

    uint32_t find_hostID(uint64_t gid){
      for(uint32_t h = 0; h < base_hGraph::numHosts; ++h){
        if(gid >= gid2host[h].first && gid < gid2host[h].second)
          return h;
        else if(isBipartite && (gid >= gid2host_withoutEdges[h].first && gid < gid2host_withoutEdges[h].second))
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
      void loadEdges(GraphTy& graph) {
        fprintf(stderr, "Loading edge-data while creating edges.\n");
        uint64_t cur = 0;

        auto map_end = host_edges_map.end();
        for(uint64_t l = 0; l < base_hGraph::numOwned; ++l){
          auto p = host_edges_map.find(L2G(l));
          if( p != map_end){
            for(auto n : (*p).second)
              graph.constructEdge(cur++, G2L(n.first), n.second);
          }
          graph.fixEndEdge(l, cur);
        }
      }

    template<typename GraphTy, typename std::enable_if<std::is_void<typename GraphTy::edge_data_type>::value>::type* = nullptr>
      void loadEdges(GraphTy& graph) {
        fprintf(stderr, "Loading void edge-data while creating edges.\n");
        uint64_t cur = 0;

        //auto p = host_edges_map.begin();
        auto map_end = host_edges_map.end();
        for (uint64_t l = 0; l < base_hGraph::numOwned; ++l){
          auto p = host_edges_map.find(L2G(l));
          if( p != map_end){
            for(auto n : (*p).second)
              graph.constructEdge(cur++, G2L(n));
          }
          graph.fixEndEdge(l, cur);
        }
      }

    //Non Void edge_data_type
    template<typename GraphEdgeTy, typename std::enable_if<!std::is_void<GraphEdgeTy>::value>::type* = nullptr>
    void fill_edge_map(std::vector<uint64_t>& nodesOnHost_vec){
      numEdges = 0;
      for(uint32_t h = 0; h < base_hGraph::numHosts; ++h){
        for(uint64_t i = 0; i  < assigned_edges_perhost[h].size(); i += 3){
          host_edges_map[assigned_edges_perhost[h][i]].push_back(std::make_pair(assigned_edges_perhost[h][i + 1],assigned_edges_perhost[h][i + 2]));

          nodesOnHost_vec.push_back(assigned_edges_perhost[h][i]);
          nodesOnHost_vec.push_back(assigned_edges_perhost[h][i + 1]);
          numEdges++;
        }
      }
    }

    //Void edge_data_type
    template<typename GraphEdgeTy, typename std::enable_if<std::is_void<GraphEdgeTy>::value>::type* = nullptr>
    void fill_edge_map(std::vector<uint64_t>& nodesOnHost_vec){
      numEdges = 0;
      for(uint32_t h = 0; h < base_hGraph::numHosts; ++h){
        for(uint64_t  i = 0; i  < assigned_edges_perhost[h].size(); i += 2) {
          host_edges_map[assigned_edges_perhost[h][i]].push_back(assigned_edges_perhost[h][i + 1]);

          nodesOnHost_vec.push_back(assigned_edges_perhost[h][i]);
          nodesOnHost_vec.push_back(assigned_edges_perhost[h][i + 1]);
          numEdges++;
        }
      }
    }

    void fill_slaveNodes(std::vector<std::vector<size_t>>& slaveNodes){

      std::vector<std::vector<uint64_t>> GlobalVec_perHost(base_hGraph::numHosts);
      std::vector<std::vector<uint32_t>> OwnerVec_perHost(base_hGraph::numHosts);
      std::vector<uint64_t> nodesOnHost_vec;


      fill_edge_map<EdgeTy>(nodesOnHost_vec);
      //Fill GlobalVec_perHost and slaveNodes vetors using assigned_edges_perhost.

      //Isolated nodes
      for(auto n = gid2host[base_hGraph::id].first; n < gid2host[base_hGraph::id].second; ++n){
        nodesOnHost_vec.push_back(n);
      }

      if(isBipartite){
        // Isolated nodes for bipartite graphs nodes without edges.
        for(auto n = gid2host_withoutEdges[base_hGraph::id].first; n < gid2host_withoutEdges[base_hGraph::id].second; ++n){
          nodesOnHost_vec.push_back(n);
        }
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
      for(uint32_t i = 0; i < base_hGraph::numHosts; i++){
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

