/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
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

#ifndef _GALOIS_DIST_HGRAPHCC_H
#define _GALOIS_DIST_HGRAPHCC_H

#include "galois/graphs/DistributedGraph.h"

namespace galois {
namespace graphs {

template<typename NodeTy, typename EdgeTy, bool columnBlocked = false, 
         bool moreColumnHosts = false, unsigned DecomposeFactor = 1>
class DistGraph_cartesianCut : public DistGraph<NodeTy, EdgeTy> {
  constexpr static const char* const GRNAME = "dGraph_cartesianCut";

  using VectorOfVector64 = std::vector<std::vector<uint64_t>>;

public:
  typedef DistGraph<NodeTy, EdgeTy> base_DistGraph;

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
    numColumnHosts = sqrt(base_DistGraph::numHosts);
    while ((base_DistGraph::numHosts % numColumnHosts) != 0) numColumnHosts--;
    numRowHosts = base_DistGraph::numHosts/numColumnHosts;
    assert(numRowHosts>=numColumnHosts);
    if (moreColumnHosts) {
      std::swap(numRowHosts, numColumnHosts);
    }
    if (base_DistGraph::id == 0) {
      galois::gPrint("Cartesian grid: ", numRowHosts, " x ", numColumnHosts, "\n");
    }
  }
#endif

  void factorizeHosts() {
    numVirtualHosts = base_DistGraph::numHosts*DecomposeFactor;
    numColumnHosts = sqrt(base_DistGraph::numHosts);

    while ((base_DistGraph::numHosts % numColumnHosts) != 0) numColumnHosts--;

    numRowHosts = base_DistGraph::numHosts/numColumnHosts;
    assert(numRowHosts>=numColumnHosts);

    if (moreColumnHosts) {
      std::swap(numRowHosts, numColumnHosts);
    }

    numRowHosts = numRowHosts*DecomposeFactor;
    if (base_DistGraph::id == 0) {
      galois::gPrint("Cartesian grid: ", numRowHosts, " x ", numColumnHosts, "\n");
      galois::gPrint("Decomposition factor: ", DecomposeFactor, "\n");
    }
  }

  unsigned virtual2RealHost(unsigned virutalHostID){
    return virutalHostID%base_DistGraph::numHosts;
  }

  //template<bool isVirtual = false, typename std::enable_if<!isVirtual>::type* = nullptr>
  unsigned gridRowID() const {
    return (base_DistGraph::id / numColumnHosts);
  }

  //template<bool isVirtual, typename std::enable_if<isVirtual>::type* = nullptr>
  //unsigned gridRowID() const {
    //return (base_DistGraph::id / numColumnHosts);
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
    return (base_DistGraph::id % numColumnHosts);
  }

  unsigned gridColumnID(unsigned id) const {
    return (id % numColumnHosts);
  }

  unsigned getBlockID(uint64_t gid) const {
    return getHostID(gid)%base_DistGraph::numHosts;
  }

  unsigned getColumnHostIDOfBlock(uint32_t blockID) const {
    if (columnBlocked) {
      return (blockID / numRowHosts); // blocked, contiguous
    } else {
      return (blockID % numColumnHosts); // round-robin, non-contiguous
    }
  }

  unsigned getColumnHostID(uint64_t gid) const {
    assert(gid < base_DistGraph::numGlobalNodes);
    uint32_t blockID = getBlockID(gid);
    return getColumnHostIDOfBlock(blockID);
  }

  uint32_t getColumnIndex(uint64_t gid) const {
    assert(gid < base_DistGraph::numGlobalNodes);
    auto blockID = getBlockID(gid);
    auto h = getColumnHostIDOfBlock(blockID);
    uint32_t columnIndex = 0;
    for (auto b = 0U; b <= blockID; ++b) {
      if (getColumnHostIDOfBlock(b) == h) {
        uint64_t start, end;
        std::tie(start, end) = base_DistGraph::gid2host[b];
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
  bool isNotCommunicationPartner(unsigned host,
                                 typename base_DistGraph::SyncType syncType,
                                 WriteLocation writeLocation,
                                 ReadLocation readLocation) {
    if (syncType == base_DistGraph::syncReduce) {
      switch(writeLocation) {
        case writeSource:
          return (gridRowID() != gridRowID(host));
        case writeDestination:
          return (gridColumnID() != gridColumnID(host));
        case writeAny:
          assert((gridRowID() == gridRowID(host)) || 
                 (gridColumnID() == gridColumnID(host)));
          return ((gridRowID() != gridRowID(host)) && 
                  (gridColumnID() != gridColumnID(host))); // false
        default:
          assert(false);
      }
    } else { // syncBroadcast
      switch(readLocation) {
        case readSource:
          if (base_DistGraph::currentBVFlag != nullptr) {
            galois::runtime::make_dst_invalid(base_DistGraph::currentBVFlag);
          }

          return (gridRowID() != gridRowID(host));
        case readDestination:
          if (base_DistGraph::currentBVFlag != nullptr) {
            galois::runtime::make_src_invalid(base_DistGraph::currentBVFlag);
          }

          return (gridColumnID() != gridColumnID(host));
        case readAny:
          assert((gridRowID() == gridRowID(host)) || 
                 (gridColumnID() == gridColumnID(host)));
          return ((gridRowID() != gridRowID(host)) && 
                  (gridColumnID() != gridColumnID(host))); // false
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
    assert(gid < base_DistGraph::numGlobalNodes);
    //for (auto h = 0U; h < base_DistGraph::numHosts; ++h) {
    for (auto h = 0U; h < numVirtualHosts; ++h) {
      uint64_t start, end;
      std::tie(start, end) = base_DistGraph::gid2host[h];
      if (gid >= start && gid < end) {
        return h;
      }
    }
    assert(false);
    return base_DistGraph::numHosts;
  }

  // Return if gid is Owned by local host.
  bool isOwned(uint64_t gid) const {
    uint64_t start, end;
    for(unsigned d = 0; d < DecomposeFactor; ++d){
      std::tie(start, end) = base_DistGraph::gid2host[base_DistGraph::id + d*base_DistGraph::numHosts];
      if(gid >= start && gid < end)
        return true;
    }
    return false;
  }

  // Return is gid is present locally (owned or mirror).
  virtual bool isLocal(uint64_t gid) const {
    assert(gid < base_DistGraph::numGlobalNodes);
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
  virtual bool nothingToSend(unsigned host, typename base_DistGraph::SyncType syncType, WriteLocation writeLocation, ReadLocation readLocation) {
    auto &sharedNodes = (syncType == base_DistGraph::syncReduce) ? base_DistGraph::mirrorNodes : base_DistGraph::masterNodes;
    if (sharedNodes[host].size() > 0) {
      if (columnBlocked) { // does not match processor grid
        return false;
      } else {
        return isNotCommunicationPartner(host, syncType, writeLocation,
                                         readLocation);
      }
    }
    return true;
  }
  virtual bool nothingToRecv(unsigned host, typename base_DistGraph::SyncType syncType, WriteLocation writeLocation, ReadLocation readLocation) {
    auto &sharedNodes = (syncType == base_DistGraph::syncReduce) ? base_DistGraph::masterNodes : base_DistGraph::mirrorNodes;
    if (sharedNodes[host].size() > 0) {
      if (columnBlocked) { // does not match processor grid
        return false;
      } else {
        return isNotCommunicationPartner(host, syncType, writeLocation,
                                         readLocation);
      }
    }
    return true;
  }

  /** 
   * Constructor for Cartesian Cut graph
   */
  DistGraph_cartesianCut(const std::string& filename, 
              const std::string& partitionFolder, unsigned host, 
              unsigned _numHosts, std::vector<unsigned>& scalefactor, 
              bool transpose = false, bool readFromFile = false,
              std::string localGraphFileName = "local_graph") 
      : base_DistGraph(host, _numHosts) {
    if (transpose) {
      GALOIS_DIE("Transpose not supported for cartesian vertex-cuts");
    }

    galois::StatTimer Tgraph_construct("TIME_GRAPH_CONSTRUCT", GRNAME);
    Tgraph_construct.start();

    if (readFromFile) {
      galois::gPrint("[", base_DistGraph::id, "] Reading local graph from "
                     "file : ", localGraphFileName, "\n");
      base_DistGraph::read_local_graph_from_file(localGraphFileName);
      Tgraph_construct.stop();
      return;
    }

    // only used to determine node splits among hosts; abandonded later
    // for the BufferedGraph
    galois::graphs::OfflineGraph g(filename);

    base_DistGraph::numGlobalNodes = g.size();
    base_DistGraph::numGlobalEdges = g.sizeEdges();

    factorizeHosts();

    base_DistGraph::computeMasters(g, scalefactor, false, DecomposeFactor);

    // at this point gid2Host has pairs for how to split nodes among
    // hosts; pair has begin and end
    std::vector<uint64_t> nodeBegin(DecomposeFactor);
    std::vector<uint64_t> nodeEnd(DecomposeFactor);
    std::vector<typename galois::graphs::OfflineGraph::edge_iterator> 
        edgeBegin(DecomposeFactor); 
    std::vector<typename galois::graphs::OfflineGraph::edge_iterator> 
        edgeEnd(DecomposeFactor); 

    for (unsigned d = 0; d < DecomposeFactor; ++d) {
      nodeBegin[d] = base_DistGraph::gid2host[base_DistGraph::id + 
                                              d*base_DistGraph::numHosts].first;
      nodeEnd[d] = base_DistGraph::gid2host[base_DistGraph::id + 
                                            d*base_DistGraph::numHosts].second;
      edgeBegin[d] = g.edge_begin(nodeBegin[d]);
      edgeEnd[d] = g.edge_begin(nodeEnd[d]);
    }

    galois::Timer inspectionTimer;

    inspectionTimer.start();

    // graph that loads assigned region into memory
    std::vector<galois::graphs::BufferedGraph<EdgeTy>> bufGraph(DecomposeFactor);

    for (unsigned d = 0; d < DecomposeFactor; ++d) {
      //galois::gPrint("Host : ", base_DistGraph::id, " : ", nodeBegin[d], " , ", 
      //               nodeEnd[d], "\n");
      bufGraph[d].loadPartialGraph(filename, nodeBegin[d], nodeEnd[d],
                                   *edgeBegin[d], *edgeEnd[d], 
                                   base_DistGraph::numGlobalNodes,
                                   base_DistGraph::numGlobalEdges);
    }

    std::vector<uint64_t> prefixSumOfEdges;

    // first pass of the graph file
    loadStatistics(bufGraph, prefixSumOfEdges, inspectionTimer); 

    // allocate memory for our underlying graph representation
    base_DistGraph::graph.allocateFrom(numNodes, numEdges);

    assert(prefixSumOfEdges.size() == numNodes);

    if (numNodes > 0) {
      base_DistGraph::graph.constructNodes();

      auto& base_graph = base_DistGraph::graph;
      galois::do_all(
        galois::iterate((uint32_t)0, numNodes),
        [&] (auto n) {
          base_graph.fixEndEdge(n, prefixSumOfEdges[n]);
        },
        galois::no_stats(),
        galois::loopname("EdgeLoading"));

    } 

    if (base_DistGraph::numOwned != 0) {
      base_DistGraph::beginMaster = 
        G2L(base_DistGraph::gid2host[base_DistGraph::id].first);
    } else {
      // no owned nodes; therefore, empty masters
      base_DistGraph::beginMaster = 0; 
    }

    base_DistGraph::printStatistics();

    loadEdges(base_DistGraph::graph, bufGraph); // second pass of the graph file

    if (columnBlocked) {
      // like an unconstrained vertex-cut
      base_DistGraph::numNodesWithEdges = numNodes; // all nodes because it is not optimized
    }

    for (unsigned d = 0; d < DecomposeFactor; ++d) {
      bufGraph[d].resetAndFree();
    }

    fillMirrorNodes(base_DistGraph::mirrorNodes);

    galois::StatTimer Tthread_ranges("TIME_THREAD_RANGES", GRNAME);
    Tthread_ranges.start();
    base_DistGraph::determineThreadRanges();
    Tthread_ranges.stop();

    base_DistGraph::determineThreadRangesMaster();
    base_DistGraph::determineThreadRangesWithEdges();
    base_DistGraph::initializeSpecificRanges();

    Tgraph_construct.stop();

    galois::StatTimer Tgraph_construct_comm("TIME_GRAPH_CONSTRUCT_COMM",
                                            GRNAME);
    Tgraph_construct_comm.start();
    base_DistGraph::setup_communication();
    Tgraph_construct_comm.stop();
  }

  void loadStatistics(
    std::vector<galois::graphs::BufferedGraph<EdgeTy>>& bufGraph,
    std::vector<uint64_t>& prefixSumOfEdges,
    galois::Timer& inspectionTimer
  ) {
    base_DistGraph::numOwned = 0;
    for (unsigned d = 0; d < DecomposeFactor; ++d) {
      base_DistGraph::numOwned += 
        base_DistGraph::gid2host[base_DistGraph::id + d*base_DistGraph::numHosts].second - 
        base_DistGraph::gid2host[base_DistGraph::id + d*base_DistGraph::numHosts].first;
    }

    std::vector<galois::DynamicBitSet> hasIncomingEdge(numColumnHosts);

    for (unsigned i = 0; i < numColumnHosts; ++i) {
      uint64_t columnBlockSize = 0;
      for (auto b = 0U; b < numVirtualHosts; ++b) {
        if (getColumnHostIDOfBlock(b) == i) {
          uint64_t start, end;
          std::tie(start, end) = base_DistGraph::gid2host[b];
          columnBlockSize += end - start;
        }
      }
      hasIncomingEdge[i].resize(columnBlockSize);
    }

    std::vector<VectorOfVector64> numOutgoingEdges(DecomposeFactor);

    for (unsigned d = 0; d < DecomposeFactor; ++d) {
      numOutgoingEdges[d].resize(numColumnHosts);
      for (unsigned i = 0; i < numColumnHosts; ++i) {
        numOutgoingEdges[d][i].assign(
          (base_DistGraph::gid2host[base_DistGraph::id + 
                                    d * base_DistGraph::numHosts].second - 
           base_DistGraph::gid2host[base_DistGraph::id + 
                                    d * base_DistGraph::numHosts].first), 0
        );
      }
    }

    for (unsigned d = 0; d < DecomposeFactor; ++d) {
      bufGraph[d].resetReadCounters();

      uint64_t rowOffset = 
        base_DistGraph::gid2host[base_DistGraph::id + 
                                 d * base_DistGraph::numHosts].first;

      galois::do_all(
        galois::iterate(base_DistGraph::gid2host[base_DistGraph::id + 
                                              d*base_DistGraph::numHosts].first,
                        base_DistGraph::gid2host[base_DistGraph::id + 
                                              d*base_DistGraph::numHosts].second),
        [&] (auto src) {
          auto ii = bufGraph[d].edgeBegin(src);
          auto ee = bufGraph[d].edgeEnd(src);
          for (; ii < ee; ++ii) {
            auto dst = bufGraph[d].edgeDestination(*ii);
            auto h = this->getColumnHostID(dst);
            hasIncomingEdge[h].set(this->getColumnIndex(dst));
            numOutgoingEdges[d][h][src - rowOffset]++;
          }
        },
        galois::no_stats(),
        galois::loopname("EdgeInspection")
      );
    }

    inspectionTimer.stop();

    uint64_t allBytesRead = 0;
    for (unsigned d = 0; d < DecomposeFactor; ++d) {
      allBytesRead += bufGraph[d].getBytesRead();
    }

    galois::gPrint("[", base_DistGraph::id, "] Edge inspection time: ", 
                   inspectionTimer.get_usec()/1000000.0f, 
                   " seconds to read ", allBytesRead, " bytes (",
                   allBytesRead / (float)inspectionTimer.get_usec(), " MBPS)\n");

    auto& net = galois::runtime::getSystemNetworkInterface();
    for (unsigned i = 0; i < numColumnHosts; ++i) {
      unsigned h = (gridRowID() * numColumnHosts) + i;
      if (h == base_DistGraph::id) continue;
      galois::runtime::SendBuffer b;
      for(unsigned d = 0; d < DecomposeFactor; ++d){
        galois::runtime::gSerialize(b, numOutgoingEdges[d][i]);
      }
      galois::runtime::gSerialize(b, hasIncomingEdge[i]);
      net.sendTagged(h, galois::runtime::evilPhase, b);
    }
    net.flush();

    for (unsigned i = 1; i < numColumnHosts; ++i) {
      decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;
      do {
        p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
      } while (!p);
      unsigned h = (p->first % numColumnHosts);
      auto& b = p->second;
      for(unsigned d = 0; d < DecomposeFactor; ++d){
        galois::runtime::gDeserialize(b, numOutgoingEdges[d][h]);
      }
      galois::runtime::gDeserialize(b, hasIncomingEdge[h]);
    }
    base_DistGraph::increment_evilPhase();

    for (unsigned i = 1; i < numColumnHosts; ++i) {
      hasIncomingEdge[0].bitwise_or(hasIncomingEdge[i]);
    }

    auto max_nodes = hasIncomingEdge[0].size();
    for (unsigned i = 0; i < numColumnHosts; ++i) {
      for (unsigned d = 0; d < DecomposeFactor; ++d) {
        max_nodes += numOutgoingEdges[d][i].size();
      }
    }
    localToGlobalVector.reserve(max_nodes);
    globalToLocalMap.reserve(max_nodes);
    prefixSumOfEdges.reserve(max_nodes);

    dummyOutgoingNodes = 0;
    numNodes = 0;
    numEdges = 0;

    for (unsigned d = 0; d < DecomposeFactor; ++d) {
      //unsigned leaderHostID = gridRowID(base_DistGraph::id + d*base_DistGraph::numHosts) * numColumnHosts;
      unsigned hostID = (base_DistGraph::id + d*base_DistGraph::numHosts);
      uint64_t src = base_DistGraph::gid2host[hostID].first;
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

    for (unsigned d = 0; d < DecomposeFactor; ++d) {
      unsigned leaderHostID = gridRowID(base_DistGraph::id + d*base_DistGraph::numHosts) * 
                              numColumnHosts;
      for (unsigned i = 0; i < numColumnHosts; ++i) {
        unsigned hostID = leaderHostID + i;
        if(virtual2RealHost(hostID) == base_DistGraph::id) continue;
        uint64_t src = base_DistGraph::gid2host[hostID].first;
        for (uint32_t j = 0; j < numOutgoingEdges[d][i].size(); ++j) {
          bool createNode = false;
          if (numOutgoingEdges[d][i][j] > 0) {
            createNode = true;
            numEdges += numOutgoingEdges[d][i][j];
          } else if ((gridColumnID(base_DistGraph::id + i*base_DistGraph::numHosts) == 
                     getColumnHostID(src)) && 
                     hasIncomingEdge[0].test(getColumnIndex(src))) {
            if (columnBlocked) {
              ++dummyOutgoingNodes;
            } else {
              galois::gWarn("Partitioning of vertices resulted in some inconsistency");
              assert(false); // should be owned
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

    base_DistGraph::numNodesWithEdges = numNodes;

    for (unsigned i = 0; i < numRowHosts; ++i) {
      //unsigned hostID;
      unsigned hostID_virtual;
      if (columnBlocked) {
        hostID_virtual = (gridColumnID() * numRowHosts) + i;
      } else {
        hostID_virtual = (i * numColumnHosts) + gridColumnID();
        //hostID_virtual = (i * numColumnHosts) + gridColumnID(base_DistGraph::id + d*base_DistGraph::numHosts);
      }
      if (virtual2RealHost(hostID_virtual) == (base_DistGraph::id)) continue;
      if (columnBlocked) {
        bool skip = false;
        for (unsigned d = 0; d < DecomposeFactor; ++d) {
          unsigned leaderHostID = gridRowID(base_DistGraph::id + d * base_DistGraph::numHosts) *
                                  numColumnHosts;
          if ((hostID_virtual >= leaderHostID) && 
              (hostID_virtual < (leaderHostID + numColumnHosts))) {
            skip = true;
          }
        }
        if (skip) continue;
      }

      uint64_t dst = base_DistGraph::gid2host[hostID_virtual].first;
      uint64_t dst_end = base_DistGraph::gid2host[hostID_virtual].second;
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
                 std::vector<galois::graphs::BufferedGraph<EdgeTy>>& bufGraph) {
    if (base_DistGraph::id == 0) {
      if (std::is_void<typename GraphTy::edge_data_type>::value) {
        galois::gPrint("Loading void edge-data while creating edges\n");
      } else {
        galois::gPrint("Loading edge-data while creating edges\n");
      }
    }

    galois::Timer timer;
    timer.start();
    for (unsigned d = 0; d < DecomposeFactor; ++d) {
      bufGraph[d].resetReadCounters();
    }

    std::atomic<uint32_t> numNodesWithEdges;
    numNodesWithEdges = base_DistGraph::numOwned + dummyOutgoingNodes;
    loadEdgesFromFile(graph, bufGraph, numNodesWithEdges);
    galois::on_each([&](unsigned tid, unsigned nthreads){
      receiveEdges(graph, numNodesWithEdges);
    });
    base_DistGraph::increment_evilPhase();

    timer.stop();
    for (unsigned d = 0; d < DecomposeFactor; ++d) {
      galois::gPrint("[", base_DistGraph::id, "] Edge loading time: ", 
                     timer.get_usec()/1000000.0f, 
                     " seconds to read ", bufGraph[d].getBytesRead(), 
                     " bytes (", 
                     bufGraph[d].getBytesRead()/(float)timer.get_usec(), 
                     " MBPS)\n");
    }
  }

  template<typename GraphTy, 
           typename std::enable_if<
             !std::is_void<typename GraphTy::edge_data_type>::value
           >::type* = nullptr>
  void loadEdgesFromFile(GraphTy& graph, 
                         std::vector<galois::graphs::BufferedGraph<EdgeTy>>& bufGraph,
                         std::atomic<uint32_t>& numNodesWithEdges) {
    auto& net = galois::runtime::getSystemNetworkInterface();

    //XXX h_offset not correct
    for (unsigned d = 0; d < DecomposeFactor; ++d) {
      //h_offset is virual hostID for DecomposeFactor > 1.
      unsigned h_offset = gridRowID() * numColumnHosts;
      galois::substrate::PerThreadStorage<VectorOfVector64> gdst_vecs(numColumnHosts);
      typedef std::vector<std::vector<typename GraphTy::edge_data_type>> DataVecVecTy;
      galois::substrate::PerThreadStorage<DataVecVecTy> gdata_vecs(numColumnHosts);
      typedef std::vector<galois::runtime::SendBuffer> SendBufferVecTy;
      galois::substrate::PerThreadStorage<SendBufferVecTy> sb(numColumnHosts);

      const unsigned& id = base_DistGraph::id; // manually copy it because it is protected
      galois::do_all(
        galois::iterate(base_DistGraph::gid2host[base_DistGraph::id + 
                                              d*base_DistGraph::numHosts].first,
                        base_DistGraph::gid2host[base_DistGraph::id + 
                                              d*base_DistGraph::numHosts].second),
        [&] (auto n) {
          auto& gdst_vec = *gdst_vecs.getLocal();
          auto& gdata_vec = *gdata_vecs.getLocal();
          uint32_t lsrc = 0;
          uint64_t cur = 0;
          if (this->isLocal(n)) {
            lsrc = this->G2L(n);
            cur = *graph.edge_begin(lsrc, galois::MethodFlag::UNPROTECTED);
          }
          auto ii = bufGraph[d].edgeBegin(n);
          auto ee = bufGraph[d].edgeEnd(n);
          for (unsigned i = 0; i < numColumnHosts; ++i) {
            gdst_vec[i].clear();
            gdata_vec[i].clear();
            gdst_vec[i].reserve(std::distance(ii, ee));
            gdata_vec[i].reserve(std::distance(ii, ee));
          }
          for (; ii < ee; ++ii) {
            uint64_t gdst = bufGraph[d].edgeDestination(*ii);
            auto gdata = bufGraph[d].edgeData(*ii);
            int i = this->getColumnHostID(gdst);
            if ((h_offset + i) == (id)) {
              assert(this->isLocal(n));
              uint32_t ldst = this->G2L(gdst);
              graph.constructEdge(cur++, ldst, gdata);
            } else {
              gdst_vec[i].push_back(gdst);
              gdata_vec[i].push_back(gdata);
            }
          }
          for (unsigned i = 0; i < numColumnHosts; ++i) {
            if (gdst_vec[i].size() > 0) {
              auto& b = (*sb.getLocal())[i];
              galois::runtime::gSerialize(b, n);
              galois::runtime::gSerialize(b, gdst_vec[i]);
              galois::runtime::gSerialize(b, gdata_vec[i]);
              if (b.size() > edgePartitionSendBufSize) {
                net.sendTagged(h_offset + i, galois::runtime::evilPhase, b);
                b.getVec().clear();
              }
            }
          }
          if (this->isLocal(n)) {
            assert(cur == (*graph.edge_end(lsrc)));
          }

          // TODO don't have to receive every iteration
          auto buffer = net.recieveTagged(galois::runtime::evilPhase, nullptr);
          this->processReceivedEdgeBuffer(buffer, graph, numNodesWithEdges);
        },
        galois::no_stats(),
        galois::loopname("EdgeLoading"));

      for (unsigned t = 0; t < sb.size(); ++t) {
        auto& sbr = *sb.getRemote(t);
        for (unsigned i = 0; i < numColumnHosts; ++i) {
          auto& b = sbr[i];
          if (b.size() > 0) {
            net.sendTagged(h_offset + i, galois::runtime::evilPhase, b);
            b.getVec().clear();
          }
        }
      }
    }
    net.flush();
  }

  template<typename GraphTy, 
           typename std::enable_if<
             std::is_void<typename GraphTy::edge_data_type>::value
           >::type* = nullptr>
  void loadEdgesFromFile(GraphTy& graph, 
                         std::vector<galois::graphs::BufferedGraph<EdgeTy>>& bufGraph,
                         std::atomic<uint32_t>& numNodesWithEdges) {
    auto& net = galois::runtime::getSystemNetworkInterface();

    for(unsigned d = 0; d < DecomposeFactor; ++d){
      //h_offset is virual hostID for DecomposeFactor > 1.
      unsigned h_offset = gridRowID() * numColumnHosts;
      galois::substrate::PerThreadStorage<VectorOfVector64> gdst_vecs(numColumnHosts);
      typedef std::vector<galois::runtime::SendBuffer> SendBufferVecTy;
      galois::substrate::PerThreadStorage<SendBufferVecTy> sb(numColumnHosts);

      const unsigned& id = base_DistGraph::id; // manually copy it because it is protected
      galois::do_all(
        galois::iterate(base_DistGraph::gid2host[base_DistGraph::id + 
                                              d*base_DistGraph::numHosts].first,
                        base_DistGraph::gid2host[base_DistGraph::id + 
                                              d*base_DistGraph::numHosts].second),
        [&] (auto n) {
          auto& gdst_vec = *gdst_vecs.getLocal();
          uint32_t lsrc = 0;
          uint64_t cur = 0;
          if (this->isLocal(n)) {
            lsrc = this->G2L(n);
            cur = *graph.edge_begin(lsrc, galois::MethodFlag::UNPROTECTED);
          }
          auto ii = bufGraph[d].edgeBegin(n);
          auto ee = bufGraph[d].edgeEnd(n);
          for (unsigned i = 0; i < numColumnHosts; ++i) {
            gdst_vec[i].clear();
            gdst_vec[i].reserve(std::distance(ii, ee));
          }
          for (; ii < ee; ++ii) {
            uint64_t gdst = bufGraph[d].edgeDestination(*ii);
            int i = this->getColumnHostID(gdst);
            if ((h_offset + i) == (id)) {
              assert(this->isLocal(n));
              uint32_t ldst = this->G2L(gdst);
              graph.constructEdge(cur++, ldst);
            } else {
              gdst_vec[i].push_back(gdst);
            }
          }
          for (unsigned i = 0; i < numColumnHosts; ++i) {
            if (gdst_vec[i].size() > 0) {
              auto& b = (*sb.getLocal())[i];
              galois::runtime::gSerialize(b, n);
              galois::runtime::gSerialize(b, gdst_vec[i]);
              //unsigned h_offset_real = virtual2RealHost(h_offset);
              if (b.size() > edgePartitionSendBufSize) {
                net.sendTagged(h_offset + i, galois::runtime::evilPhase, b);
                b.getVec().clear();
              }
            }
          }
          if (this->isLocal(n)) {
            assert(cur == (*graph.edge_end(lsrc)));
          }

          // TODO don't have to receive every iteration
          auto buffer = net.recieveTagged(galois::runtime::evilPhase, nullptr);
          this->processReceivedEdgeBuffer(buffer, graph, numNodesWithEdges);
        },
        galois::no_stats(),
        galois::loopname("EdgeLoading"));

      for (unsigned t = 0; t < sb.size(); ++t) {
        auto& sbr = *sb.getRemote(t);
        for (unsigned i = 0; i < numColumnHosts; ++i) {
          auto& b = sbr[i];
          if (b.size() > 0) {
            net.sendTagged(h_offset + i, galois::runtime::evilPhase, b);
            b.getVec().clear();
          }
        }
      }
    }
    net.flush();
  }

  // used below
  template<typename T>
  #if __GNUC__ > 5 || (__GNUC__ == 5 && __GNUC_MINOR__ > 1)
  using optional_t = std::experimental::optional<T>;
  #else
  using optional_t = boost::optional<T>;
  #endif
  template<typename GraphTy>
  void processReceivedEdgeBuffer(
    optional_t<std::pair<uint32_t, galois::runtime::RecvBuffer>>& buffer,
    GraphTy& graph, std::atomic<uint32_t>& numNodesWithEdges
  ) {
    if (buffer) {
      auto& rb = buffer->second;

      while (rb.r_size() > 0) {
        uint64_t n;
        std::vector<uint64_t> gdst_vec;
        galois::runtime::gDeserialize(rb, n);
        galois::runtime::gDeserialize(rb, gdst_vec);
        assert(isLocal(n));
        uint32_t lsrc = G2L(n);
        uint64_t cur = *graph.edge_begin(lsrc, galois::MethodFlag::UNPROTECTED);
        uint64_t cur_end = *graph.edge_end(lsrc);
        assert((cur_end - cur) == gdst_vec.size());
        deserializeEdges(graph, rb, gdst_vec, cur, cur_end);
        ++numNodesWithEdges;
      }
    }
  }

  template<typename GraphTy>
  void receiveEdges(GraphTy& graph, std::atomic<uint32_t>& numNodesWithEdges) {
    auto& net = galois::runtime::getSystemNetworkInterface();

    // receive edges for all mirror nodes
    while (numNodesWithEdges < base_DistGraph::numNodesWithEdges) {
      decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;
      p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
      processReceivedEdgeBuffer(p, graph, numNodesWithEdges);
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

  void fillMirrorNodes(std::vector<std::vector<size_t>>& mirrorNodes) {
    // mirrors for outgoing edges
    for (unsigned d = 0; d < DecomposeFactor; ++d) {
      for (unsigned i = 0; i < numColumnHosts; ++i) {
        //unsigned hostID = (gridRowID() * numColumnHosts) + i;
        unsigned hostID_virtual = (gridRowID(base_DistGraph::id + d*base_DistGraph::numHosts) * 
                                   numColumnHosts) + i;
        if (hostID_virtual == (base_DistGraph::id + d*base_DistGraph::numHosts)) continue;
        uint64_t src = base_DistGraph::gid2host[hostID_virtual].first;
        uint64_t src_end = base_DistGraph::gid2host[hostID_virtual].second;
        unsigned hostID_real = virtual2RealHost(hostID_virtual);
        mirrorNodes[hostID_real].reserve(mirrorNodes[hostID_real].size() + 
                                         src_end - src);
        for (; src < src_end; ++src) {
          if (globalToLocalMap.find(src) != globalToLocalMap.end()) {
            mirrorNodes[hostID_real].push_back(src);
          }
        }
      }
    }

    // mirrors for incoming edges
    for (unsigned d = 0; d < DecomposeFactor; ++d) {
      unsigned leaderHostID = gridRowID(base_DistGraph::id + d*base_DistGraph::numHosts) * 
                              numColumnHosts;
      for (unsigned i = 0; i < numRowHosts; ++i) {
        unsigned hostID_virtual;
        if (columnBlocked) {
          hostID_virtual = (gridColumnID(base_DistGraph::id + d*base_DistGraph::numHosts) * 
                            numRowHosts) + i;
        } else {
          hostID_virtual = (i * numColumnHosts) + 
                           gridColumnID(base_DistGraph::id + d*base_DistGraph::numHosts);
        }
        if (hostID_virtual == (base_DistGraph::id + d*base_DistGraph::numHosts)) continue;
        if (columnBlocked) {
          if ((hostID_virtual >= leaderHostID) && 
              (hostID_virtual < (leaderHostID + numColumnHosts))) continue;
        }
        uint64_t dst = base_DistGraph::gid2host[hostID_virtual].first;
        uint64_t dst_end = base_DistGraph::gid2host[hostID_virtual].second;
        unsigned hostID_real = virtual2RealHost(hostID_virtual);
        mirrorNodes[hostID_real].reserve(mirrorNodes[hostID_real].size() + 
                                         dst_end - dst);
        for (; dst < dst_end; ++dst) {
          if (globalToLocalMap.find(dst) != globalToLocalMap.end()) {
            mirrorNodes[hostID_real].push_back(dst);
          }
        }
      }
    }
  }

  std::string getPartitionFileName(const std::string& filename,
                                   const std::string&, unsigned, unsigned) {
    return filename;
  }

  bool is_vertex_cut() const {
    if (moreColumnHosts) {
      // IEC and OEC will be reversed, so do not handle it as an edge-cut
      if ((numRowHosts == 1) && (numColumnHosts == 1)) return false;
    } else {
      // IEC or OEC
      if ((numRowHosts == 1) || (numColumnHosts == 1)) return false;
    }
    return true;
  }

  void reset_bitset(typename base_DistGraph::SyncType syncType, 
                    void (*bitset_reset_range)(size_t, size_t)) const {
    if (base_DistGraph::numOwned != 0) {
      auto endMaster = base_DistGraph::beginMaster + base_DistGraph::numOwned;
      if (syncType == base_DistGraph::syncBroadcast) { // reset masters
        bitset_reset_range(base_DistGraph::beginMaster, endMaster-1);
      } else { // reset mirrors
        assert(syncType == base_DistGraph::syncReduce);
        if (base_DistGraph::beginMaster > 0) {
          bitset_reset_range(0, base_DistGraph::beginMaster - 1);
        }
        if (endMaster < numNodes) {
          bitset_reset_range(endMaster, numNodes - 1);
        }
      }
    }
  }

  std::vector<std::pair<uint32_t,uint32_t>> getMirrorRanges() const {
    std::vector<std::pair<uint32_t, uint32_t>> mirrorRanges_vec;
    if(base_DistGraph::beginMaster > 0)
      mirrorRanges_vec.push_back(std::make_pair(0, base_DistGraph::beginMaster));
    auto endMaster = base_DistGraph::beginMaster + base_DistGraph::numOwned;
    if (endMaster < numNodes) {
      mirrorRanges_vec.push_back(std::make_pair(endMaster, numNodes));
    }
    return mirrorRanges_vec;
  }

  /*
   * This function serializes the local data structures using boost binary archive.
   */
  virtual void boostSerializeLocalGraph(boost::archive::binary_oarchive& ar, 
                                        const unsigned int version = 0) const {
    // unsigned ints
    ar << numNodes;
    ar << numRowHosts;
    ar << numColumnHosts;

    // maps and vectors
    ar << localToGlobalVector;
    ar << globalToLocalMap;
  }

  /*
   * This function DeSerializes the local data structures using boost binary archive.
   */
  virtual void boostDeSerializeLocalGraph(boost::archive::binary_iarchive& ar, 
                                          const unsigned int version = 0) {
    // unsigned ints
    ar >> numNodes;
    ar >> numRowHosts;
    ar >> numColumnHosts;

    // maps and vectors
    ar >> localToGlobalVector;
    ar >> globalToLocalMap;
  }
};

} // end namespace graphs
} // end namespace galois
#endif
