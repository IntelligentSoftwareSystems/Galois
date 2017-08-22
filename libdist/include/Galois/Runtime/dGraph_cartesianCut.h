/** partitioned graph wrapper for cartesianCut -*- C++ -*-
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
 * @section Contains the cartesian/grid vertex-cut functionality to be used in dGraph.
 *
 * @author Roshan Dathathri <roshan@cs.utexas.edu>
 * @author Gurbinder Gill <gill@cs.utexas.edu>
 */
#ifndef _GALOIS_DIST_HGRAPHCC_H
#define _GALOIS_DIST_HGRAPHCC_H

#include <vector>
#include <set>
#include <algorithm>
#include <unordered_map>
#include "Galois/Runtime/dGraph.h"
#include "Galois/Runtime/OfflineGraph.h"
#include "Galois/Runtime/Serialize.h"
#include "Galois/Runtime/Tracer.h"
#include "Galois/DoAllWrap.h"

template<typename NodeTy, typename EdgeTy, bool columnBlocked = false, bool moreColumnHosts = false, bool BSPNode = false, bool BSPEdge = false, unsigned DecomposeFactor = 4>
class hGraph_cartesianCut : public hGraph<NodeTy, EdgeTy, BSPNode, BSPEdge> {
public:
  typedef hGraph<NodeTy, EdgeTy, BSPNode, BSPEdge> base_hGraph;

private:
  unsigned numRowHosts;
  unsigned numColumnHosts;

  unsigned numVirtualHosts;

  // only for checkerboard partitioning, i.e., columnBlocked = true
  uint32_t dummyOutgoingNodes; // nodes without outgoing edges that are stored with nodes having outgoing edges (to preserve original ordering locality)

  // factorize numHosts such that difference between factors is minimized
#if 0
  template<bool isVirtual = false, typename std::enable_if<!isVirtual>::type* = nullptr>
  void factorize_hosts() {
    numColumnHosts = sqrt(base_hGraph::numHosts);
    while ((base_hGraph::numHosts % numColumnHosts) != 0) numColumnHosts--;
    numRowHosts = base_hGraph::numHosts/numColumnHosts;
    assert(numRowHosts>=numColumnHosts);
    if (moreColumnHosts) {
      std::swap(numRowHosts, numColumnHosts);
    }
    if (base_hGraph::id == 0) {
      std::cerr << "Cartesian grid: " << numRowHosts << " x " << numColumnHosts << "\n";
    }
  }
#endif

  void factorize_hosts() {
    numVirtualHosts = base_hGraph::numHosts*DecomposeFactor;
    numColumnHosts = sqrt(base_hGraph::numHosts);
    while ((base_hGraph::numHosts % numColumnHosts) != 0) numColumnHosts--;
    numRowHosts = base_hGraph::numHosts/numColumnHosts;
    assert(numRowHosts>=numColumnHosts);
    if (moreColumnHosts) {
      std::swap(numRowHosts, numColumnHosts);
    }

    numRowHosts = numRowHosts*DecomposeFactor;
    if (base_hGraph::id == 0) {
      std::cerr << "Cartesian grid: " << numRowHosts << " x " << numColumnHosts << " ,  using over decomposition by factor : " << DecomposeFactor << "\n";
    }
  }

  unsigned virtual2RealHost(unsigned virutalHostID){
    return virutalHostID%base_hGraph::numHosts;
  }

  //template<bool isVirtual = false, typename std::enable_if<!isVirtual>::type* = nullptr>
  unsigned gridRowID() const {
    return (base_hGraph::id / numColumnHosts);
  }

  //template<bool isVirtual, typename std::enable_if<isVirtual>::type* = nullptr>
  //unsigned gridRowID() const {
    //return (base_hGraph::id / numColumnHosts);
  //}


  //template<bool isVirtual = false, typename std::enable_if<!isVirtual>::type* = nullptr>
  unsigned gridRowID(unsigned id) const {
    return (id / numColumnHosts);
  }

  //template<bool isVirtual, typename std::enable_if<isVirtual>::type* = nullptr>
  //unsigned gridRowID(unsigned id) const {
    //return (id / numColumnHosts);
  //}

  unsigned gridColumnID() const {
    return (base_hGraph::id % numColumnHosts);
  }

  unsigned gridColumnID(unsigned id) const {
    return (id % numColumnHosts);
  }

  unsigned getBlockID(uint64_t gid) const {
    return getHostID(gid)%base_hGraph::numHosts;
  }

  unsigned getColumnHostIDOfBlock(uint32_t blockID) const {
    if (columnBlocked) {
      return (blockID / numRowHosts); // blocked, contiguous
    } else {
      return (blockID % numColumnHosts); // round-robin, non-contiguous
    }
  }

  unsigned getColumnHostID(uint64_t gid) const {
    assert(gid < base_hGraph::totalNodes);
    uint32_t blockID = getBlockID(gid);
    return getColumnHostIDOfBlock(blockID);
  }

  uint32_t getColumnIndex(uint64_t gid) const {
    assert(gid < base_hGraph::totalNodes);
    auto blockID = getBlockID(gid);
    auto h = getColumnHostIDOfBlock(blockID);
    uint32_t columnIndex = 0;
    for (auto b = 0U; b <= blockID; ++b) {
      if (getColumnHostIDOfBlock(b) == h) {
        uint64_t start, end;
        std::tie(start, end) = base_hGraph::gid2host[b];
        if (gid < end) {
          columnIndex += gid - start;
          break; // redundant
        } else {
          columnIndex += end - start;
        }
      }
    }
    return columnIndex;
  }

  // called only for those hosts with which it shares nodes
  bool isNotCommunicationPartner(unsigned host, typename base_hGraph::SyncType syncType, WriteLocation writeLocation, ReadLocation readLocation) {
    if (syncType == base_hGraph::syncReduce) {
      switch(writeLocation) {
        case writeSource:
          return (gridRowID() != gridRowID(host));
        case writeDestination:
          return (gridColumnID() != gridColumnID(host));
        case writeAny:
          assert((gridRowID() == gridRowID(host)) || (gridColumnID() == gridColumnID(host)));
          return ((gridRowID() != gridRowID(host)) && (gridColumnID() != gridColumnID(host))); // false
        default:
          assert(false);
      }
    } else { // syncBroadcast
      switch(readLocation) {
        case readSource:
          return (gridRowID() != gridRowID(host));
        case readDestination:
          return (gridColumnID() != gridColumnID(host));
        case readAny:
          assert((gridRowID() == gridRowID(host)) || (gridColumnID() == gridColumnID(host)));
          return ((gridRowID() != gridRowID(host)) && (gridColumnID() != gridColumnID(host))); // false
        default:
          assert(false);
      }
    }
    return false;
  }

public:
  // GID = localToGlobalVector[LID]
  std::vector<uint64_t> localToGlobalVector; // TODO use LargeArray instead
  // LID = globalToLocalMap[GID]
  std::unordered_map<uint64_t, uint32_t> globalToLocalMap;

  uint32_t numNodes;
  uint64_t numEdges;

  // Return the ID to which gid belongs after patition.
  unsigned getHostID(uint64_t gid) const {
    assert(gid < base_hGraph::totalNodes);
    //for (auto h = 0U; h < base_hGraph::numHosts; ++h) {
    for (auto h = 0U; h < numVirtualHosts; ++h) {
      uint64_t start, end;
      std::tie(start, end) = base_hGraph::gid2host[h];
      if (gid >= start && gid < end) {
        return h;
      }
    }
    assert(false);
    return base_hGraph::numHosts;
  }

  // Return if gid is Owned by local host.
  bool isOwned(uint64_t gid) const {
    uint64_t start, end;
    for(auto d = 0; d < DecomposeFactor; ++d){
      std::tie(start, end) = base_hGraph::gid2host[base_hGraph::id + d*base_hGraph::numHosts];
      if(gid >= start && gid < end)
        return true;
    }
    return false;
  }

  // Return is gid is present locally (owned or mirror).
  virtual bool isLocal(uint64_t gid) const {
    assert(gid < base_hGraph::totalNodes);
    if (isOwned(gid)) return true;
    return (globalToLocalMap.find(gid) != globalToLocalMap.end());
  }

  virtual uint32_t G2L(uint64_t gid) const {
    assert(isLocal(gid));
    return globalToLocalMap.at(gid);
  }

  virtual uint64_t L2G(uint32_t lid) const {
    return localToGlobalVector[lid];
  }

  // requirement: for all X and Y,
  // On X, nothingToSend(Y) <=> On Y, nothingToRecv(X)
  // Note: templates may not be virtual, so passing types as arguments
  virtual bool nothingToSend(unsigned host, typename base_hGraph::SyncType syncType, WriteLocation writeLocation, ReadLocation readLocation) {
    auto &sharedNodes = (syncType == base_hGraph::syncReduce) ? base_hGraph::mirrorNodes : base_hGraph::masterNodes;
    if (sharedNodes[host].size() > 0) {
      if (syncType == base_hGraph::syncReduce) { // does not match processor grid
        return false;
      } else {
        return isNotCommunicationPartner(host, syncType, writeLocation, readLocation);
      }
    }
    return true;
  }
  virtual bool nothingToRecv(unsigned host, typename base_hGraph::SyncType syncType, WriteLocation writeLocation, ReadLocation readLocation) {
    auto &sharedNodes = (syncType == base_hGraph::syncReduce) ? base_hGraph::masterNodes : base_hGraph::mirrorNodes;
    if (sharedNodes[host].size() > 0) {
      if (syncType == base_hGraph::syncReduce) { // does not match processor grid
        return false;
      } else {
        return isNotCommunicationPartner(host, syncType, writeLocation, readLocation);
      }
    }
    return true;
  }

  /** 
   * Constructor for Cartesian Cut graph
   */
  hGraph_cartesianCut(const std::string& filename, 
              const std::string& partitionFolder, unsigned host, 
              unsigned _numHosts, std::vector<unsigned> scalefactor, 
              bool transpose = false) : base_hGraph(host, _numHosts) {
    if (transpose) {
      GALOIS_DIE("ERROR: transpose not supported for cartesian vertex-cuts");
    }

    Galois::Statistic statGhostNodes("TotalGhostNodes");
    Galois::StatTimer StatTimer_graph_construct("TIME_GRAPH_CONSTRUCT");
    StatTimer_graph_construct.start();
    Galois::StatTimer StatTimer_graph_construct_comm("TIME_GRAPH_CONSTRUCT_COMM");

    // only used to determine node splits among hosts; abandonded later
    // for the FileGraph which mmaps appropriate regions of memory
    Galois::Graph::OfflineGraph g(filename);

    base_hGraph::totalNodes = g.size();
    if (base_hGraph::id == 0) {
      std::cerr << "Total nodes : " << base_hGraph::totalNodes << "\n";
    }
    factorize_hosts();

    base_hGraph::computeMasters(g, scalefactor, false, DecomposeFactor);

    // at this point gid2Host has pairs for how to split nodes among
    // hosts; pair has begin and end
    std::vector<uint64_t> nodeBegin(DecomposeFactor);
    std::vector<uint64_t> nodeEnd(DecomposeFactor);
    std::vector<typename Galois::Graph::OfflineGraph::edge_iterator> edgeBegin(DecomposeFactor); 
    std::vector<typename Galois::Graph::OfflineGraph::edge_iterator> edgeEnd(DecomposeFactor); 
    std::cerr << " SIZE : " << base_hGraph::gid2host.size() << "\n";
    for(auto d = 0; d < DecomposeFactor; ++d){
      nodeBegin[d] = base_hGraph::gid2host[base_hGraph::id + d*base_hGraph::numHosts].first;
      edgeBegin[d] = g.edge_begin(nodeBegin[d]);
      nodeEnd[d] = base_hGraph::gid2host[base_hGraph::id + d*base_hGraph::numHosts].second;
      edgeEnd[d] = g.edge_begin(nodeEnd[d]);
    }

    // file graph that is mmapped for much faster reading; will use this
    // when possible from now on in the code
    std::vector<Galois::Graph::FileGraph> fileGraph(DecomposeFactor);

    for(auto d = 0; d < DecomposeFactor; ++d){
      std::stringstream ss;
      ss <<"Host : " << base_hGraph::id<<" : " << nodeBegin[d] << " , " << nodeEnd[d] << "\n";
      std::cout << ss.str();
      fileGraph[d].partFromFile(filename,
        std::make_pair(boost::make_counting_iterator<uint64_t>(nodeBegin[d]), 
                     boost::make_counting_iterator<uint64_t>(nodeEnd[d])),
        std::make_pair(edgeBegin[d], edgeEnd[d]));
    }

    std::vector<uint64_t> prefixSumOfEdges;
    loadStatistics(g, fileGraph, prefixSumOfEdges); // first pass of the graph file

    std::cerr << "[" << base_hGraph::id << "] Owned nodes: " << 
                 base_hGraph::totalOwnedNodes << "\n";

    std::cerr << "[" << base_hGraph::id << "] Ghost nodes: " << 
                 numNodes - base_hGraph::totalOwnedNodes << "\n";

    std::cerr << "[" << base_hGraph::id << "] Nodes which have edges: " << 
                 base_hGraph::numOwned << "\n";

    std::cerr << "[" << base_hGraph::id << "] Total edges : " << 
                 numEdges << "\n";

    base_hGraph::numNodes = numNodes;
    base_hGraph::numNodesWithEdges = base_hGraph::numOwned; // numOwned = #nodeswithedges
    if (numNodes > 0) {
    base_hGraph::beginMaster = G2L(base_hGraph::gid2host[base_hGraph::id].first);
    base_hGraph::endMaster = G2L(base_hGraph::gid2host[base_hGraph::id + (DecomposeFactor-1)*base_hGraph::numHosts].second - 1) + 1;

      //base_hGraph::beginMaster = G2L(base_hGraph::gid2host[base_hGraph::id].first);
      //base_hGraph::endMaster = G2L(base_hGraph::gid2host[base_hGraph::id].second - 1) + 1;

      //assert(numEdges > 0);
      assert(prefixSumOfEdges.size() == numNodes);

      if (!edgeNuma) {
        base_hGraph::graph.allocateFrom(numNodes, numEdges);
      } else {
        printf("Edge based NUMA division on\n");
        //base_hGraph::graph.allocateFrom(numNodes, numEdges, prefixSumOfEdges);
        base_hGraph::graph.allocateFromByNode(numNodes, numEdges, 
                                              prefixSumOfEdges);
      }

      //std::cerr << "Allocate done\n";

      base_hGraph::graph.constructNodes();

      //std::cerr << "Construct nodes done\n";
      auto beginIter = boost::make_counting_iterator((uint32_t)0);
      auto endIter = boost::make_counting_iterator(numNodes);
      auto& base_graph = base_hGraph::graph;
      Galois::Runtime::do_all_coupled(
        Galois::Runtime::makeStandardRange(beginIter, endIter),
        [&] (auto n) {
          base_graph.fixEndEdge(n, prefixSumOfEdges[n]);
        },
        std::make_tuple(
          Galois::loopname("EdgeLoading"),
          Galois::timeit()
        )
      );
    }

    loadEdges(base_hGraph::graph, g, fileGraph); // second pass of the graph file
    std::cerr << "[" << base_hGraph::id << "] Edges loaded \n";

    fill_mirrorNodes(base_hGraph::mirrorNodes);

    // TODO revise how this works and make it consistent across cuts
    if (!edgeNuma) {
      Galois::StatTimer StatTimer_thread_ranges("TIME_THREAD_RANGES");
      StatTimer_thread_ranges.start();
      base_hGraph::determine_thread_ranges(numNodes, prefixSumOfEdges);
      StatTimer_thread_ranges.stop();
    }

    base_hGraph::determine_thread_ranges_master();
    base_hGraph::determine_thread_ranges_with_edges();
    base_hGraph::initialize_specific_ranges();

    StatTimer_graph_construct.stop();

    StatTimer_graph_construct_comm.start();
    base_hGraph::setup_communication();
    StatTimer_graph_construct_comm.stop();
  }

  void loadStatistics(Galois::Graph::OfflineGraph& g, 
      std::vector<Galois::Graph::FileGraph>& fileGraph, 
      std::vector<uint64_t>& prefixSumOfEdges) {

    base_hGraph::totalOwnedNodes = 0;
    for(auto d = 0; d < DecomposeFactor; ++d)
      base_hGraph::totalOwnedNodes += base_hGraph::gid2host[base_hGraph::id + d*base_hGraph::numHosts].second - base_hGraph::gid2host[base_hGraph::id + d*base_hGraph::numHosts].first;

    std::vector<Galois::DynamicBitSet> hasIncomingEdge(numColumnHosts);
    for (unsigned i = 0; i < numColumnHosts; ++i) {
      uint64_t columnBlockSize = 0;
      //for (auto b = 0U; b < base_hGraph::numHosts; ++b) {
      for (auto b = 0U; b < numVirtualHosts; ++b) {
        if (getColumnHostIDOfBlock(b) == i) {
          uint64_t start, end;
          std::tie(start, end) = base_hGraph::gid2host[b];
          columnBlockSize += end - start;
        }
      }
      hasIncomingEdge[i].resize(columnBlockSize);
    }

    std::vector<std::vector<std::vector<uint64_t>>> numOutgoingEdges(DecomposeFactor);
    for(auto d = 0; d < DecomposeFactor; ++d){
      numOutgoingEdges[d].resize(numColumnHosts);
      for (unsigned i = 0; i < numColumnHosts; ++i) {
        numOutgoingEdges[d][i].assign((base_hGraph::gid2host[base_hGraph::id + d*base_hGraph::numHosts].second - base_hGraph::gid2host[base_hGraph::id + d*base_hGraph::numHosts].first), 0);
      }
    }

    for(auto d = 0; d < DecomposeFactor; ++d){
      Galois::Timer timer;
      timer.start();
      fileGraph[d].reset_byte_counters();

      uint64_t rowOffset = base_hGraph::gid2host[base_hGraph::id + d*base_hGraph::numHosts].first;
      auto beginIter = boost::make_counting_iterator(base_hGraph::gid2host[base_hGraph::id + d*base_hGraph::numHosts].first);
      auto endIter = boost::make_counting_iterator(base_hGraph::gid2host[base_hGraph::id + d*base_hGraph::numHosts].second);
      Galois::Runtime::do_all_coupled(
          Galois::Runtime::makeStandardRange(beginIter, endIter),
          [&] (auto src) {
          auto ii = fileGraph[d].edge_begin(src);
          auto ee = fileGraph[d].edge_end(src);
          for (; ii < ee; ++ii) {
          auto dst = fileGraph[d].getEdgeDst(ii);
          auto h = this->getColumnHostID(dst);
          hasIncomingEdge[h].set(this->getColumnIndex(dst));
          numOutgoingEdges[d][h][src - rowOffset]++;
          }
          },
          std::make_tuple(
            Galois::loopname("EdgeInspection"),
            Galois::timeit()
            )
          );
      timer.stop();
      fprintf(stderr, "[%u] Edge inspection time : %f seconds to read %lu bytes (%f MBPS)\n", 
          base_hGraph::id, timer.get_usec()/1000000.0f, fileGraph[d].num_bytes_read(), fileGraph[d].num_bytes_read()/(float)timer.get_usec());
    }

    auto& net = Galois::Runtime::getSystemNetworkInterface();
    for (unsigned i = 0; i < numColumnHosts; ++i) {
      unsigned h = (gridRowID() * numColumnHosts) + i;
      if (h == base_hGraph::id) continue;
      Galois::Runtime::SendBuffer b;
      for(auto d = 0; d < DecomposeFactor; ++d){
        Galois::Runtime::gSerialize(b, numOutgoingEdges[d][i]);
      }
      Galois::Runtime::gSerialize(b, hasIncomingEdge[i]);
      net.sendTagged(h, Galois::Runtime::evilPhase, b);
    }
    net.flush();

    for (unsigned i = 1; i < numColumnHosts; ++i) {
      decltype(net.recieveTagged(Galois::Runtime::evilPhase, nullptr)) p;
      do {
        net.handleReceives();
        p = net.recieveTagged(Galois::Runtime::evilPhase, nullptr);
      } while (!p);
      unsigned h = (p->first % numColumnHosts);
      auto& b = p->second;
      for(auto d = 0; d < DecomposeFactor; ++d){
        Galois::Runtime::gDeserialize(b, numOutgoingEdges[d][h]);
      }
      Galois::Runtime::gDeserialize(b, hasIncomingEdge[h]);
    }
    ++Galois::Runtime::evilPhase;

    for (unsigned i = 1; i < numColumnHosts; ++i) {
      hasIncomingEdge[0].bitwise_or(hasIncomingEdge[i]);
    }

    auto max_nodes = hasIncomingEdge[0].size();
    for (unsigned i = 0; i < numColumnHosts; ++i) {
      for(auto d = 0; d < DecomposeFactor; ++d){
        max_nodes += numOutgoingEdges[d][i].size();
      }
    }
    localToGlobalVector.reserve(max_nodes);
    globalToLocalMap.reserve(max_nodes);
    prefixSumOfEdges.reserve(max_nodes);

    dummyOutgoingNodes = 0;
    numNodes = 0;
    numEdges = 0;

    for(auto d = 0; d < DecomposeFactor; ++d){
      unsigned leaderHostID = gridRowID(base_hGraph::id + d*base_hGraph::numHosts) * numColumnHosts;
      unsigned hostID = (base_hGraph::id + d*base_hGraph::numHosts);
      uint64_t src = base_hGraph::gid2host[hostID].first;
      unsigned i = gridColumnID();
      for (uint32_t j = 0; j < numOutgoingEdges[d][i].size(); ++j) {
        numEdges += numOutgoingEdges[d][i][j];
        localToGlobalVector.push_back(src);
        assert(globalToLocalMap.find(src) == globalToLocalMap.end());
        globalToLocalMap[src] = numNodes++;
        prefixSumOfEdges.push_back(numEdges);
        ++src;
      }
    }

    for(auto d = 0; d < DecomposeFactor; ++d){
      unsigned leaderHostID = gridRowID(base_hGraph::id + d*base_hGraph::numHosts) * numColumnHosts;
      for (unsigned i = 0; i < numColumnHosts; ++i) {
        unsigned hostID = leaderHostID + i;
        if(virtual2RealHost(hostID) == base_hGraph::id) continue;
        uint64_t src = base_hGraph::gid2host[hostID].first;
        for (uint32_t j = 0; j < numOutgoingEdges[d][i].size(); ++j) {
          bool createNode = false;
          if(numOutgoingEdges[d][i][j] > 0){
            createNode = true;
            numEdges += numOutgoingEdges[d][i][j];
          }else if ((gridColumnID(base_hGraph::id + i*base_hGraph::numHosts) == getColumnHostID(src))
              && hasIncomingEdge[0].test(getColumnIndex(src))) {
            if (columnBlocked) {
              ++dummyOutgoingNodes;
            } else {
              //std::cerr << leaderHostID <<  " : SOURCE : " <<  src << "\n";
              assert(false); // should be owned
              fprintf(stderr, "WARNING: Partitioning of vertices resulted in some inconsistency");
            }
            createNode = true;
          }

          if (createNode) {
            localToGlobalVector.push_back(src);
            assert(globalToLocalMap.find(src) == globalToLocalMap.end());
            globalToLocalMap[src] = numNodes++;
            prefixSumOfEdges.push_back(numEdges);
          }
          ++src;
        }
      }
    }

    base_hGraph::numOwned = numNodes; // number of nodes for which there are outgoing edges
    for (unsigned i = 0; i < numRowHosts; ++i) {
      unsigned hostID, hostID_virtual;
      if (columnBlocked) {
        hostID_virtual = (gridColumnID() * numRowHosts) + i;
      } else {
        hostID_virtual = (i * numColumnHosts) + gridColumnID();
        //hostID_virtual = (i * numColumnHosts) + gridColumnID(base_hGraph::id + d*base_hGraph::numHosts);
      }
      if (virtual2RealHost(hostID_virtual) == (base_hGraph::id)) continue;
      if (columnBlocked) {
        bool skip = false;
        for(auto d = 0; d < DecomposeFactor; ++d){
          unsigned leaderHostID = gridRowID(base_hGraph::id + i*base_hGraph::numHosts) * numColumnHosts;
          if ((hostID_virtual >= leaderHostID) && (hostID_virtual < (leaderHostID + numColumnHosts))) 
            skip = true;
        }
        if(skip)
          continue;
      }
      uint64_t dst = base_hGraph::gid2host[hostID_virtual].first;
      uint64_t dst_end = base_hGraph::gid2host[hostID_virtual].second;
      for (; dst < dst_end; ++dst) {
        if (hasIncomingEdge[0].test(getColumnIndex(dst))) {
          localToGlobalVector.push_back(dst);
          assert(globalToLocalMap.find(dst) == globalToLocalMap.end());
          globalToLocalMap[dst] = numNodes++;
          prefixSumOfEdges.push_back(numEdges);
        }
      }
    }
    }

    template<typename GraphTy>
  void loadEdges(GraphTy& graph, 
                 Galois::Graph::OfflineGraph& g,
                 std::vector<Galois::Graph::FileGraph>& fileGraph) {
    if (base_hGraph::id == 0) {
      if (std::is_void<typename GraphTy::edge_data_type>::value) {
        fprintf(stderr, "Loading void edge-data while creating edges.\n");
      } else {
        fprintf(stderr, "Loading edge-data while creating edges.\n");
      }
    }

    Galois::Timer timer;
    timer.start();
    for(auto d = 0; d < DecomposeFactor; ++d){
      fileGraph[d].reset_byte_counters();
    }

    uint32_t numNodesWithEdges;
    numNodesWithEdges = base_hGraph::totalOwnedNodes + dummyOutgoingNodes;
    // TODO: try to parallelize this better
    Galois::on_each([&](unsigned tid, unsigned nthreads){
      if (tid == 0) loadEdgesFromFile(graph, g, fileGraph);
      // using multiple threads to receive is mostly slower and leads to a deadlock or hangs sometimes
      if ((nthreads == 1) || (tid == 1)) receiveEdges(graph, numNodesWithEdges);
    });
    ++Galois::Runtime::evilPhase;

    timer.stop();
    for(auto d = 0; d < DecomposeFactor; ++d){
      fprintf(stderr, "[%u] Edge loading time : %f seconds to read %lu bytes (%f MBPS)\n", 
          base_hGraph::id, timer.get_usec()/1000000.0f, fileGraph[d].num_bytes_read(), fileGraph[d].num_bytes_read()/(float)timer.get_usec());
    }
  }

  template<typename GraphTy, typename std::enable_if<!std::is_void<typename GraphTy::edge_data_type>::value>::type* = nullptr>
  void loadEdgesFromFile(GraphTy& graph, 
                         Galois::Graph::OfflineGraph& g,
                         std::vector<Galois::Graph::FileGraph>& fileGraph) {

    //XXX h_offset not correct
    for(auto d = 0; d < DecomposeFactor; ++d){
      //h_offset is virual hostID for DecomposeFactor > 1.
      unsigned h_offset = gridRowID() * numColumnHosts;
      auto& net = Galois::Runtime::getSystemNetworkInterface();
      std::vector<std::vector<uint64_t>> gdst_vec(numColumnHosts);
      std::vector<std::vector<typename GraphTy::edge_data_type>> gdata_vec(numColumnHosts);

      auto ee = fileGraph[d].edge_begin(base_hGraph::gid2host[base_hGraph::id + d*base_hGraph::numHosts].first);
      for (auto n = base_hGraph::gid2host[base_hGraph::id + d*base_hGraph::numHosts].first; n < base_hGraph::gid2host[base_hGraph::id + d*base_hGraph::numHosts].second; ++n) {
        uint32_t lsrc = 0;
        uint64_t cur = 0;
        if (isLocal(n)) {
          lsrc = G2L(n);
          cur = *graph.edge_begin(lsrc, Galois::MethodFlag::UNPROTECTED);
        }
        auto ii = ee;
        ee = fileGraph[d].edge_end(n);
        for (unsigned i = 0; i < numColumnHosts; ++i) {
          gdst_vec[i].clear();
          gdata_vec[i].clear();
          gdst_vec[i].reserve(std::distance(ii, ee));
          gdata_vec[i].reserve(std::distance(ii, ee));
        }
        for (; ii < ee; ++ii) {
          uint64_t gdst = fileGraph[d].getEdgeDst(ii);
          auto gdata = fileGraph[d].getEdgeData<typename GraphTy::edge_data_type>(ii);
          int i = getColumnHostID(gdst);
          if ((h_offset + i) == (base_hGraph::id)) {
            assert(isLocal(n));
            uint32_t ldst = G2L(gdst);
            graph.constructEdge(cur++, ldst, gdata);
          } else {
            gdst_vec[i].push_back(gdst);
            gdata_vec[i].push_back(gdata);
          }
        }
        for (unsigned i = 0; i < numColumnHosts; ++i) {
          if (gdst_vec[i].size() > 0) {
            Galois::Runtime::SendBuffer b;
            Galois::Runtime::gSerialize(b, n);
            Galois::Runtime::gSerialize(b, gdst_vec[i]);
            Galois::Runtime::gSerialize(b, gdata_vec[i]);
            net.sendTagged(h_offset + i, Galois::Runtime::evilPhase, b);
          }
        }
        if (isLocal(n)) {
          assert(cur == (*graph.edge_end(lsrc)));
        }
      }
      net.flush();
    }
  }

  template<typename GraphTy, typename std::enable_if<std::is_void<typename GraphTy::edge_data_type>::value>::type* = nullptr>
  void loadEdgesFromFile(GraphTy& graph, 
                         Galois::Graph::OfflineGraph& g,
                         std::vector<Galois::Graph::FileGraph>& fileGraph) {
    for(auto d = 0; d < DecomposeFactor; ++d){
      //h_offset is virual hostID for DecomposeFactor > 1.
      unsigned h_offset = gridRowID() * numColumnHosts;
      auto& net = Galois::Runtime::getSystemNetworkInterface();
      std::vector<std::vector<uint64_t>> gdst_vec(numColumnHosts);

      auto ee = fileGraph[d].edge_begin(base_hGraph::gid2host[base_hGraph::id + d*base_hGraph::numHosts].first);
      for (auto n = base_hGraph::gid2host[base_hGraph::id + d*base_hGraph::numHosts].first; n < base_hGraph::gid2host[base_hGraph::id + d*base_hGraph::numHosts].second; ++n) {
        uint32_t lsrc = 0;
        uint64_t cur = 0;
        if (isLocal(n)) {
          lsrc = G2L(n);
          cur = *graph.edge_begin(lsrc, Galois::MethodFlag::UNPROTECTED);
        }
        auto ii = ee;
        ee = fileGraph[d].edge_end(n);
        for (unsigned i = 0; i < numColumnHosts; ++i) {
          gdst_vec[i].clear();
          gdst_vec[i].reserve(std::distance(ii, ee));
        }
        for (; ii < ee; ++ii) {
          uint64_t gdst = fileGraph[d].getEdgeDst(ii);
          int i = getColumnHostID(gdst);
          if ((h_offset + i) == (base_hGraph::id)) {
            assert(isLocal(n));
            uint32_t ldst = G2L(gdst);
            graph.constructEdge(cur++, ldst);
          } else {
            gdst_vec[i].push_back(gdst);
          }
        }
        for (unsigned i = 0; i < numColumnHosts; ++i) {
          if (gdst_vec[i].size() > 0) {
            Galois::Runtime::SendBuffer b;
            Galois::Runtime::gSerialize(b, n);
            Galois::Runtime::gSerialize(b, gdst_vec[i]);
            //unsigned h_offset_real = virtual2RealHost(h_offset);
            net.sendTagged(h_offset + i, Galois::Runtime::evilPhase, b);
          }
        }
        if (isLocal(n)) {
          assert(cur == (*graph.edge_end(lsrc)));
        }
      }
      net.flush();
    }
  }

  template<typename GraphTy>
  void receiveEdges(GraphTy& graph, uint32_t& numNodesWithEdges) {
    auto& net = Galois::Runtime::getSystemNetworkInterface();
    while (numNodesWithEdges < base_hGraph::numOwned) {
      decltype(net.recieveTagged(Galois::Runtime::evilPhase, nullptr)) p;
      net.handleReceives();
      p = net.recieveTagged(Galois::Runtime::evilPhase, nullptr);
      if (p) {
        auto& rb = p->second;
        uint64_t n;
        Galois::Runtime::gDeserialize(rb, n);
        std::vector<uint64_t> gdst_vec;
        Galois::Runtime::gDeserialize(rb, gdst_vec);
        assert(isLocal(n));
        uint32_t lsrc = G2L(n);
        uint64_t cur = *graph.edge_begin(lsrc, Galois::MethodFlag::UNPROTECTED);
        uint64_t cur_end = *graph.edge_end(lsrc);
        assert((cur_end - cur) == gdst_vec.size());
        deserializeEdges(graph, rb, gdst_vec, cur, cur_end);
        ++numNodesWithEdges;
      }
    }
  }

  template<typename GraphTy, typename std::enable_if<!std::is_void<typename GraphTy::edge_data_type>::value>::type* = nullptr>
  void deserializeEdges(GraphTy& graph, Galois::Runtime::RecvBuffer& b, 
      std::vector<uint64_t>& gdst_vec, uint64_t& cur, uint64_t& cur_end) {
    std::vector<typename GraphTy::edge_data_type> gdata_vec;
    Galois::Runtime::gDeserialize(b, gdata_vec);
    uint64_t i = 0;
    while (cur < cur_end) {
      auto gdata = gdata_vec[i];
      uint64_t gdst = gdst_vec[i++];
      uint32_t ldst = G2L(gdst);
      graph.constructEdge(cur++, ldst, gdata);
    }
  }

  template<typename GraphTy, typename std::enable_if<std::is_void<typename GraphTy::edge_data_type>::value>::type* = nullptr>
  void deserializeEdges(GraphTy& graph, Galois::Runtime::RecvBuffer& b, 
      std::vector<uint64_t>& gdst_vec, uint64_t& cur, uint64_t& cur_end) {
    uint64_t i = 0;
    while (cur < cur_end) {
      uint64_t gdst = gdst_vec[i++];
      uint32_t ldst = G2L(gdst);
      graph.constructEdge(cur++, ldst);
    }
  }

  void fill_mirrorNodes(std::vector<std::vector<size_t>>& mirrorNodes){
    // mirrors for outgoing edges
    for(auto d = 0; d < DecomposeFactor; ++d){
      for(unsigned i = 0; i < numColumnHosts; ++i){
        //unsigned hostID = (gridRowID() * numColumnHosts) + i;
        unsigned hostID_virtual = (gridRowID(base_hGraph::id + d*base_hGraph::numHosts) * numColumnHosts) + i;
        if (hostID_virtual == (base_hGraph::id + d*base_hGraph::numHosts)) continue;
        uint64_t src = base_hGraph::gid2host[hostID_virtual].first;
        uint64_t src_end = base_hGraph::gid2host[hostID_virtual].second;
        unsigned hostID_real = virtual2RealHost(hostID_virtual);
        mirrorNodes[hostID_real].reserve(mirrorNodes[hostID_real].size() + src_end - src);
        for (; src < src_end; ++src) {
          if (globalToLocalMap.find(src) != globalToLocalMap.end()) {
            mirrorNodes[hostID_real].push_back(src);
          }
        }
      }
    }
    // mirrors for incoming edges
    for(auto d = 0; d < DecomposeFactor; ++d){
      unsigned leaderHostID = gridRowID(base_hGraph::id + d*base_hGraph::numHosts) * numColumnHosts;
      for (unsigned i = 0; i < numRowHosts; ++i) {
        unsigned hostID_virtual;
        if (columnBlocked) {
          hostID_virtual = (gridColumnID(base_hGraph::id + d*base_hGraph::numHosts) * numRowHosts) + i;
        } else {
          hostID_virtual = (i * numColumnHosts) + gridColumnID(base_hGraph::id + d*base_hGraph::numHosts);
        }
        if (hostID_virtual == (base_hGraph::id + d*base_hGraph::numHosts)) continue;
        if (columnBlocked) {
          if ((hostID_virtual >= leaderHostID) && (hostID_virtual < (leaderHostID + numColumnHosts))) continue;
        }
        uint64_t dst = base_hGraph::gid2host[hostID_virtual].first;
        uint64_t dst_end = base_hGraph::gid2host[hostID_virtual].second;
        unsigned hostID_real = virtual2RealHost(hostID_virtual);
        mirrorNodes[hostID_real].reserve(mirrorNodes[hostID_real].size() + dst_end - dst);
        for (; dst < dst_end; ++dst) {
          if (globalToLocalMap.find(dst) != globalToLocalMap.end()) {
            mirrorNodes[hostID_real].push_back(dst);
          }
        }
      }
    }
  }

  std::string getPartitionFileName(const std::string& filename, const std::string & basename, unsigned hostID, unsigned num_hosts){
    return filename;
  }

  bool is_vertex_cut() const{
    if ((numRowHosts == 1) || (numColumnHosts == 1)) return false; // IEC or OEC
    return true;
  }

  /**
   * Returns the start and end of master nodes in local graph.
   */
  uint64_t get_local_total_nodes() const {
    return numNodes;
  }

  void reset_bitset(typename base_hGraph::SyncType syncType, 
                    void (*bitset_reset_range)(size_t, size_t)) const {
    assert(base_hGraph::beginMaster < base_hGraph::endMaster);
    assert((base_hGraph::endMaster - base_hGraph::beginMaster) == base_hGraph::totalOwnedNodes);
    if (syncType == base_hGraph::syncBroadcast) { // reset masters
      bitset_reset_range(base_hGraph::beginMaster, base_hGraph::endMaster-1);
    } else { // reset mirrors
      assert(syncType == base_hGraph::syncReduce);
      if (base_hGraph::beginMaster > 0) {
        bitset_reset_range(0, base_hGraph::beginMaster - 1);
      }
      if (base_hGraph::endMaster < numNodes) {
        bitset_reset_range(base_hGraph::endMaster, numNodes - 1);
      }
    }
  }
};
#endif
