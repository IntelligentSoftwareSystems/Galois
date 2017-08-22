/** partitioned graph wrapper for jaggedCut -*- C++ -*-
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
 * @section Contains the 2d jagged vertex-cut functionality to be used in dGraph.
 *
 * @author Roshan Dathathri <roshan@cs.utexas.edu>
 */
#ifndef _GALOIS_DIST_HGRAPHJVC_H
#define _GALOIS_DIST_HGRAPHJVC_H

#include <vector>
#include <set>
#include <algorithm>
#include <unordered_map>
#include "Galois/Runtime/dGraph.h"
#include "Galois/Runtime/OfflineGraph.h"
#include "Galois/Runtime/Serialize.h"
#include "Galois/Runtime/Tracer.h"
#include "Galois/Runtime/CompilerHelperFunctions.h"
#include "Galois/DoAllWrap.h"

template<typename NodeTy, typename EdgeTy, bool columnBlocked = false, bool moreColumnHosts = false, bool BSPNode = false, bool BSPEdge = false>
class hGraph_jaggedCut : public hGraph<NodeTy, EdgeTy, BSPNode, BSPEdge> {
public:
  typedef hGraph<NodeTy, EdgeTy, BSPNode, BSPEdge> base_hGraph;

private:
  unsigned numRowHosts;
  unsigned numColumnHosts;

  uint32_t dummyOutgoingNodes; // nodes without outgoing edges that are stored with nodes having outgoing edges (to preserve original ordering locality)

  std::vector<std::vector<std::pair<uint64_t, uint64_t>>> jaggedColumnMap;

  // factorize numHosts such that difference between factors is minimized
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

  unsigned gridRowID() const {
    return (base_hGraph::id / numColumnHosts);
  }

  unsigned gridRowID(unsigned id) const {
    return (id / numColumnHosts);
  }

  unsigned gridColumnID() const {
    return (base_hGraph::id % numColumnHosts);
  }

  unsigned gridColumnID(unsigned id) const {
    return (id % numColumnHosts);
  }

  unsigned getBlockID(unsigned rowID, uint64_t gid) const {
    assert(gid < base_hGraph::totalNodes);
    for (auto h = 0U; h < base_hGraph::numHosts; ++h) {
      uint64_t start, end;
      std::tie(start, end) = jaggedColumnMap[rowID][h];
      if (gid >= start && gid < end) {
        return h;
      }
    }
    assert(false);
    return base_hGraph::numHosts;
  }

  unsigned getColumnHostIDOfBlock(uint32_t blockID) const {
    if (columnBlocked) {
      return (blockID / numRowHosts); // blocked, contiguous
    } else {
      return (blockID % numColumnHosts); // round-robin, non-contiguous
    }
  }

  unsigned getColumnHostID(unsigned rowID, uint64_t gid) const {
    assert(gid < base_hGraph::totalNodes);
    uint32_t blockID = getBlockID(rowID, gid);
    return getColumnHostIDOfBlock(blockID);
  }

  uint32_t getColumnIndex(unsigned rowID, uint64_t gid) const {
    assert(gid < base_hGraph::totalNodes);
    auto blockID = getBlockID(rowID, gid);
    auto h = getColumnHostIDOfBlock(blockID);
    uint32_t columnIndex = 0;
    for (auto b = 0U; b <= blockID; ++b) {
      if (getColumnHostIDOfBlock(b) == h) {
        uint64_t start, end;
        std::tie(start, end) = jaggedColumnMap[rowID][b];
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
        case writeAny:
          // columns do not match processor grid
          return false; 
        default:
          assert(false);
      }
    } else { // syncBroadcast
      switch(readLocation) {
        case readSource:
          return (gridRowID() != gridRowID(host));
        case readDestination:
        case readAny:
          // columns do not match processor grid
          return false; 
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
    for (auto h = 0U; h < base_hGraph::numHosts; ++h) {
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
    std::tie(start, end) = base_hGraph::gid2host[base_hGraph::id];
    return gid >= start && gid < end;
  }

  // Return if gid is present locally (owned or mirror).
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
      return isNotCommunicationPartner(host, syncType, writeLocation, readLocation);
    }
    return true;
  }
  virtual bool nothingToRecv(unsigned host, typename base_hGraph::SyncType syncType, WriteLocation writeLocation, ReadLocation readLocation) {
    auto &sharedNodes = (syncType == base_hGraph::syncReduce) ? base_hGraph::masterNodes : base_hGraph::mirrorNodes;
    if (sharedNodes[host].size() > 0) {
      return isNotCommunicationPartner(host, syncType, writeLocation, readLocation);
    }
    return true;
  }

  /** 
   * Constructor for jagged Cut graph
   */
  hGraph_jaggedCut(const std::string& filename, 
              const std::string& partitionFolder, unsigned host, 
              unsigned _numHosts, std::vector<unsigned> scalefactor, 
              bool transpose = false) : base_hGraph(host, _numHosts) {
    if (transpose) {
      GALOIS_DIE("ERROR: transpose not supported for jagged vertex-cuts");
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

    base_hGraph::computeMasters(g, scalefactor, false);

    // at this point gid2Host has pairs for how to split nodes among
    // hosts; pair has begin and end
    uint64_t nodeBegin = base_hGraph::gid2host[base_hGraph::id].first;
    typename Galois::Graph::OfflineGraph::edge_iterator edgeBegin = 
      g.edge_begin(nodeBegin);

    uint64_t nodeEnd = base_hGraph::gid2host[base_hGraph::id].second;
    typename Galois::Graph::OfflineGraph::edge_iterator edgeEnd = 
      g.edge_begin(nodeEnd);
    
    // file graph that is mmapped for much faster reading; will use this
    // when possible from now on in the code
    Galois::Graph::FileGraph fileGraph;

    fileGraph.partFromFile(filename,
      std::make_pair(boost::make_counting_iterator<uint64_t>(nodeBegin), 
                     boost::make_counting_iterator<uint64_t>(nodeEnd)),
      std::make_pair(edgeBegin, edgeEnd));

    determineJaggedColumnMapping(g, fileGraph); // first pass of the graph file

    std::vector<uint64_t> prefixSumOfEdges; // TODO use LargeArray
    loadStatistics(g, fileGraph, prefixSumOfEdges); // second pass of the graph file

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

    assert(prefixSumOfEdges.size() == numNodes);

    if (!edgeNuma) {
      base_hGraph::graph.allocateFrom(numNodes, numEdges);
    } else {
      printf("Edge based NUMA division on\n");
      //base_hGraph::graph.allocateFrom(numNodes, numEdges, prefixSumOfEdges);
      base_hGraph::graph.allocateFromByNode(numNodes, numEdges, 
                                            prefixSumOfEdges);
    }

    if (numNodes > 0) {
      //assert(numEdges > 0);
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

    if (base_hGraph::totalOwnedNodes != 0) {
      base_hGraph::beginMaster = G2L(base_hGraph::gid2host[base_hGraph::id].first);
      base_hGraph::endMaster = G2L(base_hGraph::gid2host[base_hGraph::id].second - 1) + 1;
    } else {
      // no owned nodes, therefore empty masters
      base_hGraph::beginMaster = 0; 
      base_hGraph::endMaster = 0;
    }

    loadEdges(base_hGraph::graph, g, fileGraph); // third pass of the graph file
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

private:

  void determineJaggedColumnMapping(Galois::Graph::OfflineGraph& g, 
                      Galois::Graph::FileGraph& fileGraph) {
    Galois::Timer timer;
    timer.start();
    fileGraph.reset_byte_counters();
    auto beginIter = boost::make_counting_iterator(base_hGraph::gid2host[base_hGraph::id].first);
    auto endIter = boost::make_counting_iterator(base_hGraph::gid2host[base_hGraph::id].second);
    std::vector<std::atomic<uint64_t>> indegree(base_hGraph::totalNodes); // TODO use LargeArray
    Galois::Runtime::do_all_coupled(
      Galois::Runtime::makeStandardRange(beginIter, endIter),
      [&] (auto src) {
        auto ii = fileGraph.edge_begin(src);
        auto ee = fileGraph.edge_end(src);
        for (; ii < ee; ++ii) {
          auto dst = fileGraph.getEdgeDst(ii);
          Galois::atomicAdd(indegree[dst], (uint64_t)1);
        }
      },
      std::make_tuple(
        Galois::loopname("CalculateIndegree"),
        Galois::timeit()
      )
    );
    timer.stop();
    fprintf(stderr, "[%u] In-degree calculation time : %f seconds to read %lu bytes (%f MBPS)\n", 
        base_hGraph::id, timer.get_usec()/1000000.0f, fileGraph.num_bytes_read(), fileGraph.num_bytes_read()/(float)timer.get_usec());

    // TODO move this to a common helper function
    std::vector<uint64_t> prefixSumOfInEdges(base_hGraph::totalNodes); // TODO use LargeArray
    auto& activeThreads = Galois::Runtime::activeThreads;
    std::vector<uint64_t> prefixSumOfThreadBlocks(activeThreads, 0);
    Galois::on_each([&](unsigned tid, unsigned nthreads) {
        assert(nthreads == activeThreads);
        auto range = Galois::block_range(beginIter, endIter,
          tid, nthreads);
        auto begin = *range.first;
        auto end = *range.second;
        // find prefix sum of each block
        if (begin < end) {
          prefixSumOfInEdges[begin] = indegree[begin];
        }
        for (auto i = begin + 1; i < end; ++i) {
          prefixSumOfInEdges[i] = prefixSumOfInEdges[i-1] + indegree[i];
        }
        if (begin < end) {
          prefixSumOfThreadBlocks[tid] = prefixSumOfInEdges[end - 1];
        } else {
          prefixSumOfThreadBlocks[tid] = 0;
        }
    });
    for (unsigned int i = 1; i < activeThreads; ++i) {
      prefixSumOfThreadBlocks[i] += prefixSumOfThreadBlocks[i-1];
    }
    Galois::on_each([&](unsigned tid, unsigned nthreads) {
        assert(nthreads == activeThreads);
        if (tid > 0) {
          auto range = Galois::block_range(beginIter, endIter,
            tid, nthreads);
          // update prefix sum from previous block
          for (auto i = (*range.first) + 1; i < *range.second; ++i) {
            prefixSumOfInEdges[i] += prefixSumOfThreadBlocks[tid-1];
          }
        }
    });

    jaggedColumnMap.resize(numColumnHosts);
    for (unsigned i = 0; i < base_hGraph::numHosts; ++i) {
      // TODO use divideByNode() instead
      // partition based on indegree-count only
      auto pair = Galois::prefix_range(prefixSumOfInEdges, 
          (uint64_t)0U, prefixSumOfInEdges.size(),
          i, base_hGraph::numHosts);
      jaggedColumnMap[gridColumnID()].push_back(pair);
    }

    auto& net = Galois::Runtime::getSystemNetworkInterface();
    for (unsigned i = 0; i < numColumnHosts; ++i) {
      unsigned h = (gridRowID() * numColumnHosts) + i;
      if (h == base_hGraph::id) continue;
      Galois::Runtime::SendBuffer b;
      Galois::Runtime::gSerialize(b, jaggedColumnMap[gridColumnID()]);
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
      Galois::Runtime::gDeserialize(b, jaggedColumnMap[h]);
    }
    ++Galois::Runtime::evilPhase;
  }

  void loadStatistics(Galois::Graph::OfflineGraph& g, 
                      Galois::Graph::FileGraph& fileGraph, 
                      std::vector<uint64_t>& prefixSumOfEdges) {
    base_hGraph::totalOwnedNodes = base_hGraph::gid2host[base_hGraph::id].second - base_hGraph::gid2host[base_hGraph::id].first;

    std::vector<Galois::DynamicBitSet> hasIncomingEdge(numColumnHosts);
    for (unsigned i = 0; i < numColumnHosts; ++i) {
      uint64_t columnBlockSize = 0;
      for (auto b = 0U; b < base_hGraph::numHosts; ++b) {
        if (getColumnHostIDOfBlock(b) == i) {
          uint64_t start, end;
          std::tie(start, end) = jaggedColumnMap[gridColumnID()][b];
          columnBlockSize += end - start;
        }
      }
      hasIncomingEdge[i].resize(columnBlockSize);
    }

    std::vector<std::vector<uint64_t> > numOutgoingEdges(numColumnHosts);
    for (unsigned i = 0; i < numColumnHosts; ++i) {
      numOutgoingEdges[i].assign(base_hGraph::totalOwnedNodes, 0);
    }

    Galois::Timer timer;
    timer.start();
    fileGraph.reset_byte_counters();
    uint64_t rowOffset = base_hGraph::gid2host[base_hGraph::id].first;
    auto beginIter = boost::make_counting_iterator(base_hGraph::gid2host[base_hGraph::id].first);
    auto endIter = boost::make_counting_iterator(base_hGraph::gid2host[base_hGraph::id].second);
    Galois::Runtime::do_all_coupled(
      Galois::Runtime::makeStandardRange(beginIter, endIter),
      [&] (auto src) {
        auto ii = fileGraph.edge_begin(src);
        auto ee = fileGraph.edge_end(src);
        for (; ii < ee; ++ii) {
          auto dst = fileGraph.getEdgeDst(ii);
          auto h = this->getColumnHostID(this->gridColumnID(), dst);
          hasIncomingEdge[h].set(this->getColumnIndex(this->gridColumnID(), dst));
          numOutgoingEdges[h][src - rowOffset]++;
        }
      },
      std::make_tuple(
        Galois::loopname("EdgeInspection"),
        Galois::timeit()
      )
    );
    timer.stop();
    fprintf(stderr, "[%u] Edge inspection time : %f seconds to read %lu bytes (%f MBPS)\n", 
        base_hGraph::id, timer.get_usec()/1000000.0f, fileGraph.num_bytes_read(), fileGraph.num_bytes_read()/(float)timer.get_usec());

    auto& net = Galois::Runtime::getSystemNetworkInterface();
    for (unsigned i = 0; i < numColumnHosts; ++i) {
      unsigned h = (gridRowID() * numColumnHosts) + i;
      if (h == base_hGraph::id) continue;
      Galois::Runtime::SendBuffer b;
      Galois::Runtime::gSerialize(b, numOutgoingEdges[i]);
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
      Galois::Runtime::gDeserialize(b, numOutgoingEdges[h]);
      Galois::Runtime::gDeserialize(b, hasIncomingEdge[h]);
    }
    ++Galois::Runtime::evilPhase;

    auto max_nodes = hasIncomingEdge[0].size(); // imprecise
    for (unsigned i = 0; i < numColumnHosts; ++i) {
      max_nodes += numOutgoingEdges[i].size();
    }
    localToGlobalVector.reserve(max_nodes);
    globalToLocalMap.reserve(max_nodes);
    prefixSumOfEdges.reserve(max_nodes);
    unsigned leaderHostID = gridRowID() * numColumnHosts;
    uint64_t src = base_hGraph::gid2host[leaderHostID].first;
    uint64_t src_end = base_hGraph::gid2host[leaderHostID+numColumnHosts-1].second;
    dummyOutgoingNodes = 0;
    numNodes = 0;
    numEdges = 0;
    for (unsigned i = 0; i < numColumnHosts; ++i) {
      for (uint32_t j = 0; j < numOutgoingEdges[i].size(); ++j) {
        bool createNode = false;
        if (numOutgoingEdges[i][j] > 0) {
          createNode = true;
          numEdges += numOutgoingEdges[i][j];
        } else if (isOwned(src)) {
          createNode = true;
        } else {
          for (unsigned k = 0; k < numColumnHosts; ++k) {
            if (k == gridColumnID()) continue;
            auto h = getColumnHostID(k, src);
            if (h == gridColumnID()) {
              if (hasIncomingEdge[k].test(getColumnIndex(k, src))) {
                createNode = true;
                ++dummyOutgoingNodes;
                break;
              }
            }
          }
        }
        if (createNode) {
          localToGlobalVector.push_back(src);
          globalToLocalMap[src] = numNodes++;
          prefixSumOfEdges.push_back(numEdges);
        }
        ++src;
      }
    }
    assert(src == src_end);
    base_hGraph::numOwned = numNodes; // number of nodes for which there are outgoing edges
    src = base_hGraph::gid2host[leaderHostID].first;
    for (uint64_t dst = 0; dst < base_hGraph::totalNodes; ++dst) {
      if (dst == src) { // skip nodes which have been allocated above
        dst = src_end - 1;
        continue;
      }
      assert((dst < src) || (dst >= src_end));
      bool createNode = false;
      for (unsigned i = 0; i < numColumnHosts; ++i) {
        auto h = getColumnHostID(i, dst);
        if (h == gridColumnID()) {
          if (hasIncomingEdge[i].test(getColumnIndex(i, dst))) {
            createNode = true;
            break;
          }
        }
      }
      if (createNode) {
        localToGlobalVector.push_back(dst);
        globalToLocalMap[dst] = numNodes++;
        prefixSumOfEdges.push_back(numEdges);
      }
    }
  }

  template<typename GraphTy>
  void loadEdges(GraphTy& graph, 
                 Galois::Graph::OfflineGraph& g,
                 Galois::Graph::FileGraph& fileGraph) {
    if (base_hGraph::id == 0) {
      if (std::is_void<typename GraphTy::edge_data_type>::value) {
        fprintf(stderr, "Loading void edge-data while creating edges.\n");
      } else {
        fprintf(stderr, "Loading edge-data while creating edges.\n");
      }
    }

    Galois::Timer timer;
    timer.start();
    fileGraph.reset_byte_counters();

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
    fprintf(stderr, "[%u] Edge loading time : %f seconds to read %lu bytes (%f MBPS)\n", 
        base_hGraph::id, timer.get_usec()/1000000.0f, fileGraph.num_bytes_read(), fileGraph.num_bytes_read()/(float)timer.get_usec());
  }

  template<typename GraphTy, typename std::enable_if<!std::is_void<typename GraphTy::edge_data_type>::value>::type* = nullptr>
  void loadEdgesFromFile(GraphTy& graph, 
                         Galois::Graph::OfflineGraph& g,
                         Galois::Graph::FileGraph& fileGraph) {
    unsigned h_offset = gridRowID() * numColumnHosts;
    auto& net = Galois::Runtime::getSystemNetworkInterface();
    std::vector<std::vector<uint64_t>> gdst_vec(numColumnHosts);
    std::vector<std::vector<typename GraphTy::edge_data_type>> gdata_vec(numColumnHosts);

    auto ee = fileGraph.edge_begin(base_hGraph::gid2host[base_hGraph::id].first);
    for (auto n = base_hGraph::gid2host[base_hGraph::id].first; n < base_hGraph::gid2host[base_hGraph::id].second; ++n) {
      uint32_t lsrc = 0;
      uint64_t cur = 0;
      if (isLocal(n)) {
        lsrc = G2L(n);
        cur = *graph.edge_begin(lsrc, Galois::MethodFlag::UNPROTECTED);
      }
      auto ii = ee;
      ee = fileGraph.edge_end(n);
      for (unsigned i = 0; i < numColumnHosts; ++i) {
        gdst_vec[i].clear();
        gdata_vec[i].clear();
        gdst_vec[i].reserve(std::distance(ii, ee));
        gdata_vec[i].reserve(std::distance(ii, ee));
      }
      for (; ii < ee; ++ii) {
        uint64_t gdst = fileGraph.getEdgeDst(ii);
        auto gdata = fileGraph.getEdgeData<typename GraphTy::edge_data_type>(ii);
        int i = getColumnHostID(gridColumnID(), gdst);
        if ((h_offset + i) == base_hGraph::id) {
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

  template<typename GraphTy, typename std::enable_if<std::is_void<typename GraphTy::edge_data_type>::value>::type* = nullptr>
  void loadEdgesFromFile(GraphTy& graph, 
                         Galois::Graph::OfflineGraph& g,
                         Galois::Graph::FileGraph& fileGraph) {
    unsigned h_offset = gridRowID() * numColumnHosts;
    auto& net = Galois::Runtime::getSystemNetworkInterface();
    std::vector<std::vector<uint64_t>> gdst_vec(numColumnHosts);

    auto ee = fileGraph.edge_begin(base_hGraph::gid2host[base_hGraph::id].first);
    for (auto n = base_hGraph::gid2host[base_hGraph::id].first; n < base_hGraph::gid2host[base_hGraph::id].second; ++n) {
      uint32_t lsrc = 0;
      uint64_t cur = 0;
      if (isLocal(n)) {
        lsrc = G2L(n);
        cur = *graph.edge_begin(lsrc, Galois::MethodFlag::UNPROTECTED);
      }
      auto ii = ee;
      ee = fileGraph.edge_end(n);
      for (unsigned i = 0; i < numColumnHosts; ++i) {
        gdst_vec[i].clear();
        gdst_vec[i].reserve(std::distance(ii, ee));
      }
      for (; ii < ee; ++ii) {
        uint64_t gdst = fileGraph.getEdgeDst(ii);
        int i = getColumnHostID(gridColumnID(), gdst);
        if ((h_offset + i) == base_hGraph::id) {
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
          net.sendTagged(h_offset + i, Galois::Runtime::evilPhase, b);
        }
      }
      if (isLocal(n)) {
        assert(cur == (*graph.edge_end(lsrc)));
      }
    }
    net.flush();
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
    for (uint32_t i = 0; i < numNodes; ++i) {
      uint64_t gid = localToGlobalVector[i];
      unsigned hostID = getHostID(gid);
      if (hostID == base_hGraph::id) continue;
      mirrorNodes[hostID].push_back(gid);
    }
  }

public:

  std::string getPartitionFileName(const std::string& filename, const std::string & basename, unsigned hostID, unsigned num_hosts){
    return filename;
  }

  bool is_vertex_cut() const{
    if (moreColumnHosts) {
      // IEC and OEC will be reversed, so do not handle it as an edge-cut
      if ((numRowHosts == 1) && (numColumnHosts == 1)) return false;
    } else {
      if ((numRowHosts == 1) || (numColumnHosts == 1)) return false; // IEC or OEC
    }
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
    uint32_t numMasters = base_hGraph::endMaster - base_hGraph::beginMaster;

    assert(base_hGraph::beginMaster <= base_hGraph::endMaster);
    assert(numMasters == base_hGraph::totalOwnedNodes);

    if (numMasters != 0) {
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
  }
};
#endif
