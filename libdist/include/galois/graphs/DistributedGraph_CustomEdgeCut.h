/* This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

/** 
 * @file DistributedGraph_CustomEdgeCut.h
 *
 * Implements the custom edge cut partitioning scheme for DistributedGraph.
 */
#ifndef _GALOIS_DIST_HGRAPHCUSTOMEDGCUT_H
#define _GALOIS_DIST_HGRAPHCUSTOMEDGCUT_H

#include "galois/graphs/DistributedGraph.h"
#include <sstream>

namespace galois {
namespace graphs {

/**
 * Distributed graph that partitions based on a manual assignment of nodes
 * to hosts.
 *
 * @tparam NodeTy type of node data for the graph
 * @tparam EdgeTy type of edge data for the graph
 *
 * @todo fully document and clean up code
 * @warning not meant for public use + not fully documented yet
 */
template<typename NodeTy, typename EdgeTy>
class DistGraphCustomEdgeCut : public DistGraph<NodeTy, EdgeTy> {
  constexpr static const char* const GRNAME = "dGraph_customEdgeCut";
  public:
    //! typedef for base DistGraph class
    using base_DistGraph = DistGraph<NodeTy, EdgeTy>;

    //! store the ownerIDs sorted according to the global IDs
    std::vector<size_t> OwnerVec; 
    std::vector<std::pair<uint32_t, uint32_t>> hostNodes;

    std::vector<size_t> GlobalVec_ordered; //!< global ids sorted vector
    // To send edges to different hosts: #Src #Dst
    std::vector<std::vector<uint64_t>> assigned_edges_perhost;
    std::vector<uint64_t> recv_assigned_edges;
    std::vector<uint64_t> assignedNodes;
    uint64_t num_total_edges_to_receive;
    uint64_t numOwned = 0;

    //! GID = localToGlobalVector[LID]
    std::vector<uint64_t> localToGlobalVector;
    //! LID = globalToLocalMap[GID]
    std::unordered_map<uint64_t, uint32_t> globalToLocalMap;
    //! vector that stored custom vertex ID map
    std::vector<int32_t> vertexIDMap;

    uint64_t globalOffset;
    uint32_t numNodes;
    bool isBipartite;
    uint64_t numEdges;

    //! @todo this is broken
    //! @warning this is broken; use at own risk
    unsigned getHostID(uint64_t gid) const {
      auto lid = G2L(gid);
      return OwnerVec[lid];
    }

    bool isOwned(uint64_t gid) const {
      //assert(isLocal(gid));
      if (isLocal(gid) && (globalToLocalMap.at(gid) < numOwned)) {
        return true;
      }
      return false;
    }

    virtual bool isLocal(uint64_t gid) const {
      assert(gid < base_DistGraph::numGlobalNodes);
      return (globalToLocalMap.find(gid) != globalToLocalMap.end());
    }
  
    virtual uint32_t G2L(uint64_t gid) const {
      assert(isLocal(gid));
      return globalToLocalMap.at(gid);
    }
  
    virtual uint64_t L2G(uint32_t lid) const {
      return localToGlobalVector[lid];
    }

    /** 
     * Reading vertexIDMap binary file
     * Assuming that vertexIDMap binary file contains int32_t entries.
     */
    bool readVertexIDMappingFile(const std::string& vertexIDMap_filename, 
                                 std::vector<int32_t>&vertexIDMap, 
                                 uint32_t num_entries_to_read, 
                                 uint32_t startLoc){
      std::ifstream meta_file(vertexIDMap_filename, std::ifstream::binary);

      if (!meta_file.is_open()) {
        std::cerr << "Unable to open file " << vertexIDMap_filename << "! Exiting!\n";
        return false;
      }

      meta_file.seekg(startLoc * sizeof(int32_t), meta_file.beg);
      meta_file.read(reinterpret_cast<char*>(&vertexIDMap[0]), 
                     sizeof(int32_t)*(num_entries_to_read));
      galois::gPrint("[", base_DistGraph::id, "] Number of nodes read :: ", 
                     num_entries_to_read, "\n");
      return true;
    }

    std::pair<uint32_t, uint32_t> nodes_by_host(uint32_t host) const {
      return std::make_pair<uint32_t, uint32_t>(~0,~0);
    }

    std::pair<uint64_t, uint64_t> nodes_by_host_G(uint32_t host) const {
      return std::make_pair<uint64_t, uint64_t>(~0,~0);
    }

    /**
     * Constructor for Vertex Cut
     */
    DistGraphCustomEdgeCut(const std::string& filename, 
               const std::string& partitionFolder,
               unsigned host, unsigned _numHosts, 
               std::vector<unsigned>& scalefactor, 
               const std::string& vertexIDMap_filename, 
               bool transpose = false, 
               uint32_t VCutThreshold = 100, 
               bool bipartite = false) : base_DistGraph(host, _numHosts) {
      if (!scalefactor.empty()) {
        if (base_DistGraph::id == 0) {
          std::cerr << "WARNING: scalefactor not supported for custom-cuts\n";
        }
        scalefactor.clear();
      }

      if (vertexIDMap_filename.empty()) {
        if (base_DistGraph::id == 0) {
          std::cerr << "WARNING: no vertexIDMap_filename provided for custom-cuts\n";
        }
        abort();
      }

      galois::runtime::reportParam("(NULL)", "CUSTOM EDGE CUT", "0");

      galois::CondStatTimer<MORE_DIST_STATS> Tgraph_construct(
        "GraphPartitioningTime", GRNAME
      );

      Tgraph_construct.start();

      galois::graphs::OfflineGraph g(filename);
      isBipartite = bipartite;

      base_DistGraph::numGlobalNodes = g.size();
      base_DistGraph::numGlobalEdges = g.sizeEdges();
      base_DistGraph::computeMasters(g, scalefactor, isBipartite);

      // Read the vertexIDMap_filename for masters.
      auto startLoc = base_DistGraph::gid2host[base_DistGraph::id].first;
      auto num_entries_to_read = 
        (base_DistGraph::gid2host[base_DistGraph::id].second - 
         base_DistGraph::gid2host[base_DistGraph::id].first);

      vertexIDMap.resize(num_entries_to_read);
      readVertexIDMappingFile(vertexIDMap_filename, vertexIDMap, 
                              num_entries_to_read, startLoc);

      uint64_t nodeBegin = base_DistGraph::gid2host[base_DistGraph::id].first;
      typename galois::graphs::OfflineGraph::edge_iterator edgeBegin = 
        g.edge_begin(nodeBegin);

      uint64_t nodeEnd = base_DistGraph::gid2host[base_DistGraph::id].second;
      typename galois::graphs::OfflineGraph::edge_iterator edgeEnd = 
        g.edge_begin(nodeEnd);

      galois::Timer edgeInspectionTimer;
      edgeInspectionTimer.start();

      galois::graphs::BufferedGraph<EdgeTy> mpiGraph;
      mpiGraph.loadPartialGraph(filename, nodeBegin, nodeEnd, *edgeBegin, 
                                *edgeEnd, base_DistGraph::numGlobalNodes,
      base_DistGraph::numGlobalEdges);

      mpiGraph.resetReadCounters();

      uint64_t numEdges_distribute = edgeEnd - edgeBegin; 
      fprintf(stderr, "[%u] Total edges to distribute : %lu\n",
              base_DistGraph::id, numEdges_distribute);

      /********************************************
       * Assign edges to the hosts using heuristics
       * and send/recv from other hosts.
       * ******************************************/
      std::vector<uint64_t> prefixSumOfEdges;
      assign_edges_phase1(g, mpiGraph, numEdges_distribute, vertexIDMap,
                          prefixSumOfEdges, base_DistGraph::mirrorNodes,
                          edgeInspectionTimer);

      base_DistGraph::numOwned = numOwned;
      base_DistGraph::numNodesWithEdges = numNodes;

      if (base_DistGraph::numOwned > 0) {
        base_DistGraph::beginMaster = G2L(localToGlobalVector[0]); 
          //base_DistGraph::gid2host[base_DistGraph::id].first);
      } else {
        base_DistGraph::beginMaster = 0;
      }

      /******************************************
       * Allocate and construct the graph
       *****************************************/
      base_DistGraph::graph.allocateFrom(numNodes, numEdges);
      base_DistGraph::graph.constructNodes();

      auto& base_graph = base_DistGraph::graph;
      galois::do_all(
        galois::iterate((uint32_t)0, numNodes),
        [&] (auto n) {
          base_graph.fixEndEdge(n, prefixSumOfEdges[n]);
        },
        #if MORE_DIST_STATS
        galois::loopname("EdgeLoading"),
        #endif
        galois::no_stats()
      );

      base_DistGraph::printStatistics();

      loadEdges(base_DistGraph::graph, mpiGraph, numEdges_distribute);

      mpiGraph.resetAndFree();

      /*******************************************/

      galois::runtime::getHostBarrier().wait();

      if (transpose && (numNodes > 0)) {
        base_DistGraph::graph.transpose();
        base_DistGraph::transposed = true;
      } 

      galois::CondStatTimer<MORE_DIST_STATS> Tthread_ranges("ThreadRangesTime",
                                                            GRNAME);

      Tthread_ranges.start();
      base_DistGraph::determineThreadRanges();
      Tthread_ranges.stop();

      base_DistGraph::determineThreadRangesMaster();
      base_DistGraph::determineThreadRangesWithEdges();
      base_DistGraph::initializeSpecificRanges();

      base_DistGraph::edgesEqualMasters(); // edges should be masters

      Tgraph_construct.stop();

      /*****************************************
       * Communication PreProcessing:
       * Exchange mirrors and master nodes among
       * hosts
       ****************************************/
      galois::CondStatTimer<MORE_DIST_STATS> Tgraph_construct_comm(
        "GraphCommSetupTime", GRNAME
      );

      Tgraph_construct_comm.start();
      base_DistGraph::setup_communication();
      Tgraph_construct_comm.stop();
    }

    //! from https:://graphics.stanford.edu/~seander/bithacks.html
    unsigned int findTrailingZeros(unsigned int v) {
      unsigned int c;     // c will be the number of zero bits on the right,
                          // so if v is 1101000 (base 2), then c will be 3
      // NOTE: if 0 == v, then c = 31.
      if (v & 0x1) {
        // special case for odd v (assumed to happen half of the time)
        c = 0;
      } else {
        c = 1;
      
        if ((v & 0xffff) == 0) {  
          v >>= 16;  
          c += 16;
        }
      
        if ((v & 0xff) == 0) {  
          v >>= 8;  
          c += 8;
        }
      
        if ((v & 0xf) == 0) {  
          v >>= 4;
          c += 4;
        }
      
        if ((v & 0x3) == 0) {  
          v >>= 2;
          c += 2;
        }
      
        c -= v & 0x1;
      }	
      
      return c;
    }

    /**
     * Find the communication partner of a particular host for a particular
     * round during metadata sending.
     *
     * @param roundNum current round number
     * @param hostID the id of this machine
     * @param numHosts number of hosts total
     * @returns the communication partner of the host 
     */
    unsigned findCommPartner(unsigned roundNum, unsigned hostID, 
                             unsigned numHosts) {
      if (roundNum % 2 == 1) {
        // odd round = odd hosts +, even hosts -
        if (hostID % 2 == 0) {
          return (hostID + roundNum) % numHosts;
        } else {
          return (hostID + numHosts - roundNum) % numHosts;
        }
      } else { 
        // even round slightly more complex
        unsigned toDivideBy = (unsigned)1 << findTrailingZeros(roundNum);
        //printf("divide by %u\n", toDivideBy);
        unsigned myGroup = hostID / toDivideBy;

        if (myGroup % 2 == 0) { // even group
          return (hostID + roundNum) % numHosts;
        } else { // odd group
          return (hostID + numHosts - roundNum) % numHosts;
        }
      }
    }

    template<typename GraphTy>
    void loadEdges(GraphTy& graph, galois::graphs::BufferedGraph<EdgeTy>& mpiGraph,
                   uint64_t numEdges_distribute) {
      if (base_DistGraph::id == 0) {
        if (std::is_void<typename GraphTy::edge_data_type>::value) {
          fprintf(stderr, "Loading void edge-data while creating edges.\n");
        } else {
          fprintf(stderr, "Loading edge-data while creating edges.\n");
        }
      }

      galois::Timer timer;
      timer.start();
      mpiGraph.resetReadCounters();

      assigned_edges_perhost.resize(base_DistGraph::numHosts);

      assign_load_send_edges(graph, mpiGraph, numEdges_distribute);

      std::atomic<uint64_t> edgesToReceive;
      edgesToReceive.store(num_total_edges_to_receive);

      galois::on_each(
        [&](unsigned tid, unsigned nthreads) {
          receive_edges(graph, edgesToReceive);
        });

      base_DistGraph::increment_evilPhase();

      timer.stop();
      galois::gPrint("[", base_DistGraph::id, "] Edge loading time: ", timer.get_usec()/1000000.0f, 
          " seconds to read ", mpiGraph.getBytesRead(), " bytes (",
          mpiGraph.getBytesRead()/(float)timer.get_usec(), " MBPS)\n");
    }

    //! Calculating the number of edges to send to other hosts
    //! @todo Function way too long; split into helper functions + calls
    void assign_edges_phase1(galois::graphs::OfflineGraph& g, 
                             galois::graphs::BufferedGraph<EdgeTy>& mpiGraph, 
                             uint64_t numEdges_distribute, 
                             std::vector<int32_t>& vertexIDMap, 
                             std::vector<uint64_t>& prefixSumOfEdges, 
                             std::vector<std::vector<size_t>>& mirrorNodes,
                             galois::Timer& edgeInspectionTimer) {
      // Go over assigned nodes and distribute edges.
      std::vector<std::vector<uint64_t>> numOutgoingEdges(base_DistGraph::numHosts);
      std::vector<galois::DynamicBitSet> hasIncomingEdge(base_DistGraph::numHosts);
      std::vector<galois::GAccumulator<uint64_t>> 
          num_assigned_edges_perhost(base_DistGraph::numHosts);
      std::vector<galois::GAccumulator<uint32_t>> 
          num_assigned_nodes_perhost(base_DistGraph::numHosts);
      num_total_edges_to_receive = 0;

      auto numNodesAssigned = base_DistGraph::gid2host[base_DistGraph::id].second - 
                              base_DistGraph::gid2host[base_DistGraph::id].first;

      for (uint32_t i = 0; i < base_DistGraph::numHosts; ++i) {
        numOutgoingEdges[i].assign(numNodesAssigned, 0);
        hasIncomingEdge[i].resize(base_DistGraph::numGlobalNodes);
      }

      mpiGraph.resetReadCounters();

      uint64_t globalOffset = base_DistGraph::gid2host[base_DistGraph::id].first;

      auto& net = galois::runtime::getSystemNetworkInterface();

      if ((net.Num % 2) != 0) {
        galois::gWarn("CUSTOM EDGE CUT ONLY SUPPORTS POWER OF 2 HOST #S");
      }

      galois::do_all(
        galois::iterate(base_DistGraph::gid2host[base_DistGraph::id].first,
                        base_DistGraph::gid2host[base_DistGraph::id].second),
        [&] (auto src) {
          auto ee = mpiGraph.edgeBegin(src);
          auto ee_end = mpiGraph.edgeEnd(src);
          auto num_edges = std::distance(ee, ee_end);
          auto h = this->find_hostID(src - globalOffset);
          assert(h < net.Num);
          /*
           * numOutgoingEdges starts at 1, let the receive side know that
           * src is owned by the host h. Therefore, to get the actual number
           * of edges we have to substract 1.
           */
          numOutgoingEdges[h][src - globalOffset] = 1;
          num_assigned_nodes_perhost[h] += 1;
          num_assigned_edges_perhost[h] += num_edges;
          numOutgoingEdges[h][src - globalOffset] += num_edges;

          for (; ee != ee_end; ++ee) {
            auto gdst = mpiGraph.edgeDestination(*ee);
            hasIncomingEdge[h].set(gdst);
          }
        },
        #if MORE_DIST_STATS
        galois::loopname("EdgeInspection"),
        #endif
        galois::no_stats()
      );

      // time should have been started outside of this loop
      edgeInspectionTimer.stop();

      galois::gPrint("[", base_DistGraph::id, "] Edge inspection time: ",
                     edgeInspectionTimer.get_usec()/1000000.0f, " seconds to read ",
                     mpiGraph.getBytesRead(), " bytes (",
                     mpiGraph.getBytesRead()/(float)edgeInspectionTimer.get_usec(),
                     " MBPS)\n");

      uint64_t check_numEdges = 0;
      for (uint32_t h = 0; h < base_DistGraph::numHosts; ++h) {
        check_numEdges += num_assigned_edges_perhost[h].reduce();
      }
      galois::gPrint("[", base_DistGraph::id, "] check_numEdges done\n");

      assert(check_numEdges == numEdges_distribute);

      numOwned = num_assigned_nodes_perhost[base_DistGraph::id].reduce();

      galois::DynamicBitSet sentHosts;
      galois::DynamicBitSet recvHosts;
      sentHosts.resize(net.Num);
      recvHosts.resize(net.Num);

      sentHosts.set(base_DistGraph::id);
      recvHosts.set(base_DistGraph::id);

      /****** Exchange numOutgoingEdges sets *********/
      // send and clear assigned_edges_perhost to receive from other hosts
      galois::gPrint("[", base_DistGraph::id, "] Starting send/recv of the data\n");

      for (unsigned roundNum = 1; roundNum < net.Num; roundNum++) {
        // find comm partner
        unsigned commPartner = findCommPartner(roundNum, base_DistGraph::id,
                                               net.Num);

        galois::gDebug("[", base_DistGraph::id, "] Round ", roundNum, ", comm "
                       "partner is ", commPartner, "\n");

        // send my data off to comm partner
        galois::runtime::SendBuffer b;
        galois::runtime::gSerialize(b, num_assigned_nodes_perhost[commPartner].reduce());
        galois::runtime::gSerialize(b, num_assigned_edges_perhost[commPartner].reduce());
        galois::runtime::gSerialize(b, numOutgoingEdges[commPartner]);
        numOutgoingEdges[commPartner].clear();
        galois::runtime::gSerialize(b, hasIncomingEdge[commPartner]);
        net.sendTagged(commPartner, galois::runtime::evilPhase, b);
        b.getVec().clear();

        // expect data from comm partner back
        decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;
        do {
          p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
        } while(!p);

        if (p->first != commPartner) {
          GALOIS_DIE("wrong comm partner");
        }

        uint32_t num_nodes_from_host = 0;
        uint64_t num_edges_from_host = 0;
        galois::runtime::gDeserialize(p->second, num_nodes_from_host);
        galois::runtime::gDeserialize(p->second, num_edges_from_host);
        galois::runtime::gDeserialize(p->second, numOutgoingEdges[p->first]);
        galois::runtime::gDeserialize(p->second, hasIncomingEdge[p->first]);
        num_total_edges_to_receive += num_edges_from_host;
        numOwned += num_nodes_from_host;

        sentHosts.set(commPartner);
        recvHosts.set(commPartner);

        base_DistGraph::increment_evilPhase();
      }

      if (sentHosts.count() != net.Num) {
        GALOIS_DIE("not sent to everyone");
      }
      if (recvHosts.count() != net.Num) {
        GALOIS_DIE("not recv from everyone");
      }

      //for (unsigned x = 0; x < net.Num; ++x) {
      //  if(x == base_DistGraph::id) continue;
      //  galois::runtime::SendBuffer b;
      //  galois::runtime::gSerialize(b, num_assigned_nodes_perhost[x].reduce());
      //  galois::runtime::gSerialize(b, num_assigned_edges_perhost[x].reduce());
      //  galois::runtime::gSerialize(b, numOutgoingEdges[x]);
      //  galois::runtime::gSerialize(b, hasIncomingEdge[x]);
      //  net.sendTagged(x, galois::runtime::evilPhase, b);
      //}
      //net.flush();
      //galois::gPrint("[", base_DistGraph::id, "] Sent the data\n");

      // receive
      //for (unsigned x = 0; x < net.Num; ++x) {
      //  if(x == base_DistGraph::id) continue;
      //  decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;
      //  do {
      //    p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
      //  } while(!p);

      //  uint32_t num_nodes_from_host = 0;
      //  uint64_t num_edges_from_host = 0;
      //  galois::runtime::gDeserialize(p->second, num_nodes_from_host);
      //  galois::runtime::gDeserialize(p->second, num_edges_from_host);
      //  galois::runtime::gDeserialize(p->second, numOutgoingEdges[p->first]);
      //  galois::runtime::gDeserialize(p->second, hasIncomingEdge[p->first]);
      //  num_total_edges_to_receive += num_edges_from_host;
      //  numOwned += num_nodes_from_host;
      //}
      //base_DistGraph::increment_evilPhase();

      galois::gPrint("[", base_DistGraph::id, "] Metadata exchange done\n");

      for (unsigned x = 0; x < net.Num; ++x) {
        if(x == base_DistGraph::id) continue;

        assert(hasIncomingEdge[base_DistGraph::id].size() == 
               hasIncomingEdge[x].size());

        hasIncomingEdge[base_DistGraph::id].bitwise_or(hasIncomingEdge[x]);
      }

      galois::gPrint("[", base_DistGraph::id, "] Start: Fill local and global "
                     "vectors\n");
      numNodes = 0;
      numEdges = 0;
      localToGlobalVector.reserve(numOwned);
      globalToLocalMap.reserve(numOwned);
      uint64_t src = 0;
      for(uint32_t i = 0; i < base_DistGraph::numHosts; ++i){
        for(unsigned j = 0; j < numOutgoingEdges[i].size(); ++j){
          if(numOutgoingEdges[i][j] > 0){
            /* Subtract 1, since we added 1 before sending to know the
             * existence of nodes which do not have outgoing edges but
             * are still owned.
             */
            numEdges += (numOutgoingEdges[i][j] - 1);
            localToGlobalVector.push_back(src);
            globalToLocalMap[src] = numNodes++;
            prefixSumOfEdges.push_back(numEdges);
          }
          ++src;
        }
      }
      galois::gPrint("[", base_DistGraph::id, "] End: Fill local and global "
                     "vectors\n");

      /* At this point numNodes should be equal to the number of
       * nodes owned by the host.
       */
      assert(numNodes == numOwned);
      assert(localToGlobalVector.size() == numOwned);

      galois::gPrint("[", base_DistGraph::id, "] Start: Fill Ghosts\n");
      /* In a separate loop for ghosts, so that all the masters can be assigned
       * contigous local ids.
       */
      for(uint64_t i = 0; i < base_DistGraph::numGlobalNodes; ++i){
        /*
         * if it has incoming edges on this host, then it is a ghosts node.
         * Node should be created for this locally.
         * NOTE: Since this is edge cut, this ghost will not have outgoing edges,
         * therefore, will not add to the prefixSumOfEdges.
         */
        if (hasIncomingEdge[base_DistGraph::id].test(i) && !isOwned(i)){
          localToGlobalVector.push_back(i);
          globalToLocalMap[i] = numNodes++;
          prefixSumOfEdges.push_back(numEdges);
        }
      }

      galois::gPrint("[", base_DistGraph::id, "] End: Fill Ghosts\n");

      uint32_t numGhosts = (localToGlobalVector.size() - numOwned);
      std::vector<uint32_t> mirror_mapping_to_hosts;
      if(numGhosts > 0){
        mirror_mapping_to_hosts.resize(numGhosts);
      }

      galois::gPrint("[", base_DistGraph::id, "] Start: assignedNodes send\n");

      /****** Exchange assignedNodes: All to all *********/
      for (unsigned x = 0; x < net.Num; ++x) {
        if(x == base_DistGraph::id) continue;
        galois::runtime::SendBuffer b;
#if 0
        galois::runtime::gSerialize(b, numOwned);
        for(uint32_t i = 0; i < numOwned; ++i) {
          if(base_DistGraph::id == 0)
            std::cerr << localToGlobalVector[i] << "\n";
          galois::runtime::gSerialize(b,localToGlobalVector[i]);
        }
#endif
        std::vector<uint64_t> temp_vec(localToGlobalVector.begin(), localToGlobalVector.begin() + numOwned);
        galois::runtime::gSerialize(b, temp_vec);
        net.sendTagged(x, galois::runtime::evilPhase, b);
      }
      galois::gPrint("[", base_DistGraph::id, "] Start: assignedNodes receive\n");

      galois::gPrint("[", base_DistGraph::id, "] End: assignedNodes send\n");
      net.flush();

      galois::gPrint("[", base_DistGraph::id, "] Start: assignedNodes receive\n");
      //receive
      for (unsigned x = 0; x < net.Num; ++x) {
        if(x == base_DistGraph::id) continue;

        decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;
        do {
          p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
        } while(!p);

        std::vector<uint64_t> temp_vec;
        galois::runtime::gDeserialize(p->second, temp_vec);

        /*
         * This vector should be sorted. find_hostID expects a sorted vector.
         */
        assert(std::is_sorted(temp_vec.begin(), temp_vec.end()));


        uint32_t from_hostID = p->first;

        //update mirror to hosts mapping.
        galois::do_all(
            galois::iterate(localToGlobalVector.begin() + numOwned,
                            localToGlobalVector.end()),
          [&] (auto src) {
              auto h = this->find_hostID(temp_vec, src, from_hostID);
              if(h < std::numeric_limits<uint32_t>::max()){
                mirror_mapping_to_hosts[this->G2L(src) - numOwned] = h;
              }
          },
          #if MORE_DIST_STATS
          galois::loopname("MirrorToHostAssignment"),
          #endif
          galois::no_stats()
        );
      }
      galois::gPrint("[", base_DistGraph::id, "] End: assignedNodes receive\n");
      base_DistGraph::increment_evilPhase();

      // fill mirror nodes
      for (uint32_t i = 0; i < (localToGlobalVector.size() - numOwned); ++i) {
        mirrorNodes[mirror_mapping_to_hosts[i]].push_back(localToGlobalVector[numOwned + i]);
      }

      fprintf(stderr, "[%u] Resident nodes : %u , Resident edges : %lu\n", 
              base_DistGraph::id, numNodes, numEdges);
    }

    // Helper functions
    uint32_t find_hostID(uint64_t offset) {
      assert(offset < vertexIDMap.size());
      return vertexIDMap[offset];
      return std::numeric_limits<uint32_t>::max();
    }

    uint32_t find_hostID(std::vector<uint64_t>& vec, uint64_t gid, 
                         uint32_t from_hostID) {
      auto iter = std::lower_bound(vec.begin(), vec.end(), gid);
      if((*iter == gid) && (iter != vec.end())){
        return from_hostID;
      }
      return std::numeric_limits<uint32_t>::max();
    }

    // Edge type is not void.
    template<typename GraphTy, 
             typename std::enable_if<
               !std::is_void<typename GraphTy::edge_data_type>::value
             >::type* = nullptr>
      void assign_load_send_edges(GraphTy& graph, 
                                  galois::graphs::BufferedGraph<EdgeTy>& mpiGraph, 
                                  uint64_t numEdges_distribute) {
        using DstVecType = std::vector<std::vector<uint64_t>>;
        galois::substrate::PerThreadStorage<DstVecType> 
            gdst_vecs(base_DistGraph::numHosts);

        using DataVecType = 
            std::vector<std::vector<typename GraphTy::edge_data_type>>;
        galois::substrate::PerThreadStorage<DataVecType> 
            gdata_vecs(base_DistGraph::numHosts);

        using SendBufferVecTy = std::vector<galois::runtime::SendBuffer>; 
        galois::substrate::PerThreadStorage<SendBufferVecTy> 
          sendBuffers(base_DistGraph::numHosts);

        auto& net = galois::runtime::getSystemNetworkInterface();
        uint64_t globalOffset = base_DistGraph::gid2host[base_DistGraph::id].first;

        const unsigned& id = this->base_DistGraph::id;
        const unsigned& numHosts = this->base_DistGraph::numHosts;

        // Go over assigned nodes and distribute edges.
        galois::do_all(
          galois::iterate(base_DistGraph::gid2host[base_DistGraph::id].first,
                          base_DistGraph::gid2host[base_DistGraph::id].second),
          [&] (auto src) {
            auto ee = mpiGraph.edgeBegin(src);
            auto ee_end = mpiGraph.edgeEnd(src);

            auto& gdst_vec = *gdst_vecs.getLocal();
            auto& gdata_vec = *gdata_vecs.getLocal();

            for (unsigned i = 0; i < numHosts; ++i) {
              gdst_vec[i].clear();
              gdata_vec[i].clear();
              //gdst_vec[i].reserve(std::distance(ii, ee));
            }

            auto h = this->find_hostID(src - globalOffset);
            if (h != id) {
              // Assign edges for high degree nodes to the destination
              for(; ee != ee_end; ++ee){
                auto gdst = mpiGraph.edgeDestination(*ee);
                auto gdata = mpiGraph.edgeData(*ee);
                gdst_vec[h].push_back(gdst);
                gdata_vec[h].push_back(gdata);
              }
            } else {
              /*
               * If source is owned, all outgoing edges belong to this host
               */
              assert(this->isOwned(src));
              uint32_t lsrc = 0;
              uint64_t cur = 0;
              lsrc = this->G2L(src);
              cur = *graph.edge_begin(lsrc, galois::MethodFlag::UNPROTECTED);
              //keep all edges with the source node
              for(; ee != ee_end; ++ee){
                auto gdst = mpiGraph.edgeDestination(*ee);
                uint32_t ldst = this->G2L(gdst);
                auto gdata = mpiGraph.edgeData(*ee);
                graph.constructEdge(cur++, ldst, gdata);
              }
              assert(cur == (*graph.edge_end(lsrc)));
            }

            // send 
            for (uint32_t h = 0; h < numHosts; ++h) {
              if (h == id) continue;
              if (gdst_vec[h].size()) {
                auto& sendBuffer = (*sendBuffers.getLocal())[h];
                galois::runtime::gSerialize(sendBuffer, src, gdst_vec[h], 
                                            gdata_vec[h]);
                if (sendBuffer.size() > edgePartitionSendBufSize) {
                  net.sendTagged(h, galois::runtime::evilPhase, sendBuffer);
                  sendBuffer.getVec().clear();
                }
              }
            }
          },
          #if MORE_DIST_STATS
          galois::loopname("EdgeLoading"),
          #endif
          galois::no_stats()
        );

        // flush buffers
        for (unsigned threadNum = 0; 
             threadNum < sendBuffers.size(); 
             ++threadNum) {
          auto& sbr = *sendBuffers.getRemote(threadNum);
          for (unsigned h = 0; h < this->base_DistGraph::numHosts; ++h) {
            if (h == this->base_DistGraph::id) continue;
            auto& sendBuffer = sbr[h];
            if (sendBuffer.size() > 0) {
              net.sendTagged(h, galois::runtime::evilPhase, sendBuffer);
              sendBuffer.getVec().clear();
            }
          }
        }

        net.flush();
      }

    // Edge type is void.
    template<typename GraphTy, 
             typename std::enable_if<
               std::is_void<typename GraphTy::edge_data_type>::value
             >::type* = nullptr>
      void assign_load_send_edges(GraphTy& graph, 
                                  galois::graphs::BufferedGraph<EdgeTy>& mpiGraph, 
                                  uint64_t numEdges_distribute) {
        using DstVecType = std::vector<std::vector<uint64_t>>;
        galois::substrate::PerThreadStorage<DstVecType> 
            gdst_vecs(base_DistGraph::numHosts);

        using SendBufferVecTy = std::vector<galois::runtime::SendBuffer>; 
        galois::substrate::PerThreadStorage<SendBufferVecTy> 
          sendBuffers(base_DistGraph::numHosts);

        auto& net = galois::runtime::getSystemNetworkInterface();
        uint64_t globalOffset = base_DistGraph::gid2host[base_DistGraph::id].first;

        const unsigned& id = this->base_DistGraph::id;
        const unsigned& numHosts = this->base_DistGraph::numHosts;

        // Go over assigned nodes and distribute edges.
        galois::do_all(
          galois::iterate(base_DistGraph::gid2host[base_DistGraph::id].first,
                          base_DistGraph::gid2host[base_DistGraph::id].second),
          [&] (auto src) {
            auto ee = mpiGraph.edgeBegin(src);
            auto ee_end = mpiGraph.edgeEnd(src);

            auto& gdst_vec = *gdst_vecs.getLocal();

            for (unsigned i = 0; i < numHosts; ++i) {
              gdst_vec[i].clear();
            }

            auto h = this->find_hostID(src - globalOffset);
            if (h != id) {
              //Assign edges for high degree nodes to the destination
              for (; ee != ee_end; ++ee) {
                auto gdst = mpiGraph.edgeDestination(*ee);
                gdst_vec[h].push_back(gdst);
              }
            } else {
              /*
               * If source is owned, all outgoing edges belong to this host
               */
              assert(this->isOwned(src));
              uint32_t lsrc = 0;
              uint64_t cur = 0;
              lsrc = this->G2L(src);
              cur = *graph.edge_begin(lsrc, galois::MethodFlag::UNPROTECTED);
              //keep all edges with the source node
              for(; ee != ee_end; ++ee){
                auto gdst = mpiGraph.edgeDestination(*ee);
                uint32_t ldst = this->G2L(gdst);
                graph.constructEdge(cur++, ldst);
              }
              assert(cur == (*graph.edge_end(lsrc)));
            }

            // send 
            for (uint32_t h = 0; h < numHosts; ++h) {
              if (h == id) continue;
              if (gdst_vec[h].size()) {
                auto& sendBuffer = (*sendBuffers.getLocal())[h];
                galois::runtime::gSerialize(sendBuffer, src, gdst_vec[h]);
                if (sendBuffer.size() > edgePartitionSendBufSize) {
                  net.sendTagged(h, galois::runtime::evilPhase, sendBuffer);
                  sendBuffer.getVec().clear();
                }
              }
            }
          },
          #if MORE_DIST_STATS
          galois::loopname("EdgeLoading"),
          #endif
          galois::no_stats()
        );

        // flush buffers
        for (unsigned threadNum = 0; 
             threadNum < sendBuffers.size(); 
             ++threadNum) {
          auto& sbr = *sendBuffers.getRemote(threadNum);
          for (unsigned h = 0; h < this->base_DistGraph::numHosts; ++h) {
            if (h == this->base_DistGraph::id) continue;
            auto& sendBuffer = sbr[h];
            if (sendBuffer.size() > 0) {
              net.sendTagged(h, galois::runtime::evilPhase, sendBuffer);
              sendBuffer.getVec().clear();
            }
          }
        }

        net.flush();
      }


    template<typename GraphTy>
    void receive_edges(GraphTy& graph, std::atomic<uint64_t>& edgesToReceive) {
      galois::StatTimer StatTimer_exchange_edges("RECEIVE_EDGES_TIME", GRNAME);
      auto& net = galois::runtime::getSystemNetworkInterface();

      // receive the edges from other hosts
      while (edgesToReceive) {
        decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;
        p = net.recieveTagged(galois::runtime::evilPhase, nullptr);

        if (p) {
          auto& receiveBuffer = p->second;

          while (receiveBuffer.r_size() > 0) {
            uint64_t _src;
            std::vector<uint64_t> _gdst_vec;
            galois::runtime::gDeserialize(receiveBuffer, _src, _gdst_vec);
            edgesToReceive -= _gdst_vec.size();
            assert(isOwned(_src));
            uint32_t lsrc = G2L(_src);
            uint64_t cur = *graph.edge_begin(lsrc, 
                                             galois::MethodFlag::UNPROTECTED);
            uint64_t cur_end = *graph.edge_end(lsrc);
            assert((cur_end - cur) == _gdst_vec.size());

            deserializeEdges(graph, receiveBuffer, _gdst_vec, cur, cur_end);
          }
        }
      }
    }

  template<typename GraphTy,
           typename std::enable_if<
             !std::is_void<typename GraphTy::edge_data_type>::value
           >::type* = nullptr>
  void deserializeEdges(GraphTy& graph, galois::runtime::RecvBuffer& b, 
      std::vector<uint64_t>& gdst_vec, uint64_t& cur, uint64_t& cur_end) {
    std::vector<typename GraphTy::edge_data_type> gdata_vec;
    galois::runtime::gDeserialize(b, gdata_vec);
    uint64_t i = 0;
    while (cur < cur_end) {
      auto gdata = gdata_vec[i];
      uint64_t gdst = gdst_vec[i++];
      uint32_t ldst = G2L(gdst);
      graph.constructEdge(cur++, ldst, gdata);
    }
  }

  template<typename GraphTy,
           typename std::enable_if<
             std::is_void<typename GraphTy::edge_data_type>::value
           >::type* = nullptr>
  void deserializeEdges(GraphTy& graph, galois::runtime::RecvBuffer& b, 
      std::vector<uint64_t>& gdst_vec, uint64_t& cur, uint64_t& cur_end) {
    uint64_t i = 0;
    while (cur < cur_end) {
      uint64_t gdst = gdst_vec[i++];
      uint32_t ldst = G2L(gdst);
      graph.constructEdge(cur++, ldst);
    }
  }

  /**
   * @returns the total number of nodes; master + slaves created on the local host.
   */
  uint64_t get_local_total_nodes() const {
    return (base_DistGraph::numOwned);
  }

  void reset_bitset(typename base_DistGraph::SyncType syncType, void (*bitset_reset_range)(size_t, size_t)) const {
    size_t first_owned = 0;
    size_t last_owned = 0;

    if (base_DistGraph::numOwned > 0) {
      first_owned = G2L(localToGlobalVector[0]);
      last_owned = G2L(localToGlobalVector[numOwned - 1]);
      assert(first_owned <= last_owned);
      assert((last_owned - first_owned + 1) == base_DistGraph::numOwned);
    } 

    if (syncType == base_DistGraph::syncBroadcast) { // reset masters
      // only reset if we actually own something
      if (base_DistGraph::numOwned > 0)
        bitset_reset_range(first_owned, last_owned);
    } else { // reset mirrors
      assert(syncType == base_DistGraph::syncReduce);

      if (base_DistGraph::numOwned > 0) {
        if (first_owned > 0) {
          bitset_reset_range(0, first_owned - 1);
        }
        if (last_owned < (numNodes - 1)) {
          bitset_reset_range(last_owned + 1, numNodes - 1);
        }
      } else {
        // only time we care is if we have ghost nodes, i.e. 
        // numNodes is non-zero
        if (numNodes > 0) {
          bitset_reset_range(0, numNodes - 1);
        }
      }
    }
  }

  std::vector<std::pair<uint32_t,uint32_t>> getMirrorRanges() const {
    size_t first_owned = 0;
    size_t last_owned = 0;

    if (base_DistGraph::numOwned > 0) {
      first_owned = G2L(localToGlobalVector[0]);
      last_owned = G2L(localToGlobalVector[numOwned - 1]);
      assert(first_owned <= last_owned);
      assert((last_owned - first_owned + 1) == base_DistGraph::numOwned);
    }

    std::vector<std::pair<uint32_t, uint32_t>> mirrorRanges_vec;
    if(base_DistGraph::numOwned > 0) {
      if (first_owned > 0) {
        mirrorRanges_vec.push_back(std::make_pair(0, first_owned));
      }
      if (last_owned < (numNodes - 1)) {
        mirrorRanges_vec.push_back(std::make_pair(last_owned + 1, numNodes));
      }
    } else {
      if (numNodes > 0) {
        mirrorRanges_vec.push_back(std::make_pair(0, numNodes));
      }
    }
    return mirrorRanges_vec;
  }

  void print_string(std::string s) {
    std::stringstream ss_cout;
    ss_cout << base_DistGraph::id << s << "\n";
    std::cerr << ss_cout.str();
  }

  bool is_vertex_cut() const{
    return false;
  }
};

} // end namespace graphs
} // end namespace galois
#endif
