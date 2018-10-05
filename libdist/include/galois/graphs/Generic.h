/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
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
 * Implements the custom edge cut partitioning scheme for DistGraph.
 */

#ifndef _GALOIS_DIST_GENERIC_H
#define _GALOIS_DIST_GENERIC_H

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
template <typename NodeTy, typename EdgeTy, typename Partitioner>
class DistGraphGeneric : public DistGraph<NodeTy, EdgeTy> {
  constexpr static const char* const GRNAME = "dGraph_Generic";
  Partitioner* graphPartitioner;

 public:
  //! typedef for base DistGraph class
  using base_DistGraph = DistGraph<NodeTy, EdgeTy>;

  //! GID = localToGlobalVector[LID]
  galois::gstl::Vector<uint64_t> localToGlobalVector;
  //! LID = globalToLocalMap[GID]
  std::unordered_map<uint64_t, uint32_t> globalToLocalMap;

  uint32_t numNodes;
  uint64_t numEdges;
  uint32_t nodesToReceive;

  unsigned getHostID(uint64_t gid) const {
    assert(gid < base_DistGraph::numGlobalNodes);
    return graphPartitioner->getMaster(gid);
  }

  bool isOwned(uint64_t gid) const {
    assert(gid < base_DistGraph::numGlobalNodes);
    return (graphPartitioner->getMaster(gid) == base_DistGraph::id);
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

  virtual bool nothingToSend(unsigned host,
                             typename base_DistGraph::SyncType syncType,
                             WriteLocation writeLocation,
                             ReadLocation readLocation) {
    auto& sharedNodes = (syncType == base_DistGraph::syncReduce) ?
                        base_DistGraph::mirrorNodes :
                        base_DistGraph::masterNodes;

    unsigned map = 2;
    if (syncType == base_DistGraph::syncReduce) {
      map = 0;
    } else {
      map = 1;
    }

    if (graphPartitioner->isCartCut()) {
      if (sharedNodes[host].size() > 0) {
        return graphPartitioner->isNotCommunicationPartner(host, map,
                                 writeLocation, readLocation,
                                 base_DistGraph::transposed);
      }
    } else {
      return (sharedNodes[host].size() == 0);
    }
    return true;
  }

  virtual bool nothingToRecv(unsigned host,
                             typename base_DistGraph::SyncType syncType,
                             WriteLocation writeLocation,
                             ReadLocation readLocation) {
    auto& sharedNodes = (syncType == base_DistGraph::syncReduce) ?
                        base_DistGraph::masterNodes :
                        base_DistGraph::mirrorNodes;

    unsigned map = 2;
    if (syncType == base_DistGraph::syncReduce) {
      map = 0;
    } else {
      map = 1;
    }

    if (graphPartitioner->isCartCut()) {
      if (sharedNodes[host].size() > 0) {
        return graphPartitioner->isNotCommunicationPartner(host, map,
                                 writeLocation, readLocation,
                                 base_DistGraph::transposed);
      }
    } else {
      return (sharedNodes[host].size() == 0);
    }
    return true;
  }

  /**
   * Constructor
   */
  DistGraphGeneric(const std::string& filename, unsigned host,
                   unsigned _numHosts, bool transpose = false)
      : base_DistGraph(host, _numHosts) {
    galois::runtime::reportParam("dGraph", "GenericPartitioner", "0");
    galois::CondStatTimer<MORE_DIST_STATS> Tgraph_construct(
        "GraphPartitioningTime", GRNAME);
    Tgraph_construct.start();

    galois::graphs::OfflineGraph g(filename);
    base_DistGraph::numGlobalNodes = g.size();
    base_DistGraph::numGlobalEdges = g.sizeEdges();
    std::vector<unsigned> dummy;
    // not actually getting masters, but getting assigned readers for nodes
    base_DistGraph::computeMasters(g, dummy);

    graphPartitioner = new Partitioner(host, _numHosts);
    // TODO abstract this away somehow
    graphPartitioner->saveGIDToHost(base_DistGraph::gid2host);

    uint64_t nodeBegin = base_DistGraph::gid2host[base_DistGraph::id].first;
    typename galois::graphs::OfflineGraph::edge_iterator edgeBegin =
        g.edge_begin(nodeBegin);
    uint64_t nodeEnd = base_DistGraph::gid2host[base_DistGraph::id].second;
    typename galois::graphs::OfflineGraph::edge_iterator edgeEnd =
        g.edge_begin(nodeEnd);

    // signifies how many outgoing edges a particular host should expect from
    // this host
    std::vector<std::vector<uint64_t>>
      numOutgoingEdges(base_DistGraph::numHosts);
    // signifies if a host should create a node because it has an incoming edge
    std::vector<galois::DynamicBitSet>
      hasIncomingEdge(base_DistGraph::numHosts);

    galois::StatTimer inspectionTimer("EdgeInspection", GRNAME);
    inspectionTimer.start();

    galois::graphs::BufferedGraph<EdgeTy> bufGraph;
    bufGraph.loadPartialGraph(filename, nodeBegin, nodeEnd, *edgeBegin,
                              *edgeEnd, base_DistGraph::numGlobalNodes,
                              base_DistGraph::numGlobalEdges);
    bufGraph.resetReadCounters();

    // assign edges to other nodes
    edgeInspection(bufGraph, numOutgoingEdges, hasIncomingEdge,
                   inspectionTimer);
    galois::DynamicBitSet& finalIncoming = hasIncomingEdge[base_DistGraph::id];

    galois::StatTimer mapTimer("NodeMapping", GRNAME);
    mapTimer.start();
    galois::gstl::Vector<uint64_t> prefixSumOfEdges =
      nodeMapping(numOutgoingEdges, finalIncoming);
    mapTimer.stop();


    // Graph construction related calls

    base_DistGraph::beginMaster = 0;
    // Allocate and construct the graph
    base_DistGraph::graph.allocateFrom(numNodes, numEdges);
    base_DistGraph::graph.constructNodes();

    // edge end fixing
    auto& base_graph = base_DistGraph::graph;
    galois::do_all(
      galois::iterate((uint32_t)0, numNodes),
      [&](auto n) { base_graph.fixEndEdge(n, prefixSumOfEdges[n]); },
#if MORE_DIST_STATS
      galois::loopname("FixEndEdgeLoop"),
#endif
      galois::no_stats()
    );
    fillMirrors();

    base_DistGraph::printStatistics();


    // Edge loading

    loadEdges(base_DistGraph::graph, bufGraph);
    bufGraph.resetAndFree();

    // Finalization

    if (transpose && (numNodes > 0)) {
      // consider all nodes to have outgoing edges (TODO better way to do this?)
      // for now it's fine I guess
      base_DistGraph::numNodesWithEdges = numNodes;
      base_DistGraph::graph.transpose(GRNAME);
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

    Tgraph_construct.stop();

    galois::CondStatTimer<MORE_DIST_STATS> Tgraph_construct_comm(
        "GraphCommSetupTime", GRNAME);

    Tgraph_construct_comm.start();
    base_DistGraph::setup_communication();
    Tgraph_construct_comm.stop();
  }

  /**
   * Free the graph partitioner
   */
  ~DistGraphGeneric() {
    delete graphPartitioner;
  }

 private:
  /**
   * Assign edges to hosts (but don't actually send), and send this information
   * out to all hosts
   * @param[in] bufGraph local graph to read
   * @param[in,out] numOutgoingEdges specifies which nodes on a host will have
   * outgoing edges
   * @param[in,out] hasIncomingEdge indicates which nodes (that need to be
   * created)on a host have incoming edges
   */
  void edgeInspection(galois::graphs::BufferedGraph<EdgeTy>& bufGraph,
                      std::vector<std::vector<uint64_t>>& numOutgoingEdges,
                      std::vector<galois::DynamicBitSet>& hasIncomingEdge,
                      galois::StatTimer& inspectionTimer) {
    // number of nodes that this host has read from disk
    uint32_t numRead = base_DistGraph::gid2host[base_DistGraph::id].second -
                       base_DistGraph::gid2host[base_DistGraph::id].first;

    // allocate space for outgoing edges
    for (uint32_t i = 0; i < base_DistGraph::numHosts; ++i) {
      numOutgoingEdges[i].assign(numRead, 0);
    }

    galois::DynamicBitSet hostHasOutgoing;
    hostHasOutgoing.resize(base_DistGraph::numHosts);
    hostHasOutgoing.reset();
    assignEdges(bufGraph, numOutgoingEdges, hasIncomingEdge, hostHasOutgoing);

    inspectionTimer.stop();
    // report edge inspection time
    uint64_t allBytesRead = bufGraph.getBytesRead();
    galois::gPrint(
        "[", base_DistGraph::id,
        "] Edge inspection time: ", inspectionTimer.get_usec() / 1000000.0f,
        " seconds to read ", allBytesRead, " bytes (",
        allBytesRead / (float)inspectionTimer.get_usec(), " MBPS)\n");

    sendInspectionData(numOutgoingEdges, hasIncomingEdge, hostHasOutgoing);

    // setup a single hasIncomingEdge bitvector

    uint32_t myHostID = base_DistGraph::id;
    if (hasIncomingEdge[myHostID].size() == 0) {
      hasIncomingEdge[myHostID].resize(base_DistGraph::numGlobalNodes);
      hasIncomingEdge[myHostID].reset();
    }
    recvInspectionData(numOutgoingEdges, hasIncomingEdge[myHostID]);
    base_DistGraph::increment_evilPhase();
  }

  /**
   * Inspect read edges and determine where to send them. Mark metadata as
   * necessary.
   *
   * @param[in] bufGraph local graph to read
   * @param[in,out] numOutgoingEdges specifies which nodes on a host will have
   * outgoing edges
   * @param[in,out] hasIncomingEdge indicates which nodes (that need to be
   * created)on a host have incoming edges
   * @param[in,out] hostHasOutgoing bitset tracking which hosts have outgoing
   * edges from this host
   */
  void assignEdges(galois::graphs::BufferedGraph<EdgeTy>& bufGraph,
                   std::vector<std::vector<uint64_t>>& numOutgoingEdges,
                   std::vector<galois::DynamicBitSet>& hasIncomingEdge,
                   galois::DynamicBitSet& hostHasOutgoing) {
    std::vector<galois::CopyableAtomic<char>>
      indicatorVars(base_DistGraph::numHosts);
    // initialize indicators of initialized bitsets to 0
    for (unsigned i = 0; i < base_DistGraph::numHosts; i++) {
      indicatorVars[i] = 0;
    }

    // global offset into my read nodes
    uint64_t globalOffset = base_DistGraph::gid2host[base_DistGraph::id].first;
    uint32_t globalNodes = base_DistGraph::numGlobalNodes;

    galois::do_all(
        // iterate over my read nodes
        galois::iterate(base_DistGraph::gid2host[base_DistGraph::id].first,
                        base_DistGraph::gid2host[base_DistGraph::id].second),
        [&](auto src) {
          auto ee        = bufGraph.edgeBegin(src);
          auto ee_end    = bufGraph.edgeEnd(src);
          uint64_t numEdges = std::distance(ee, ee_end);

          for (; ee != ee_end; ee++) {
            uint32_t dst = bufGraph.edgeDestination(*ee);
            uint32_t hostBelongs = -1;
            hostBelongs = graphPartitioner->getEdgeOwner(src, dst, numEdges);

            numOutgoingEdges[hostBelongs][src - globalOffset] += 1;
            hostHasOutgoing.set(hostBelongs);
            bool hostIsMasterOfDest = 
              (hostBelongs == graphPartitioner->getMaster(dst));

            // this means a mirror must be created for destination node on
            // that host since it will not be created otherwise
            if (!hostIsMasterOfDest) {
              auto& bitsetStatus = indicatorVars[hostBelongs];

              // initialize the bitset if necessary
              if (bitsetStatus == 0) {
                char expected = 0;
                bool result = bitsetStatus.compare_exchange_strong(expected,
                                                                   1);
                // i swapped successfully, therefore do allocation
                if (result) {
                  hasIncomingEdge[hostBelongs].resize(globalNodes);
                  hasIncomingEdge[hostBelongs].reset();
                  bitsetStatus = 2;
                }
              } 
              // until initialized, loop
              while (indicatorVars[hostBelongs] != 2);
              hasIncomingEdge[hostBelongs].set(dst);
            }
          }
        },
#if MORE_DIST_STATS
        galois::loopname("AssignEdges"),
#endif
        galois::steal(),
        galois::no_stats()
    );
  }

  /**
   * Send data out from inspection to other hosts.
   *
   * @param[in,out] numOutgoingEdges specifies which nodes on a host will have
   * outgoing edges
   * @param[in,out] hasIncomingEdge indicates which nodes (that need to be
   * created)on a host have incoming edges
   * @param[in] hostHasOutgoing bitset tracking which hosts have outgoing
   * edges from this host
   */
  void sendInspectionData(std::vector<std::vector<uint64_t>>& numOutgoingEdges,
                          std::vector<galois::DynamicBitSet>& hasIncomingEdge,
                          galois::DynamicBitSet& hostHasOutgoing) {
    auto& net = galois::runtime::getSystemNetworkInterface();

    galois::GAccumulator<uint64_t> bytesSent;
    bytesSent.reset();

    for (unsigned h = 0; h < net.Num; h++) {
      if (h == net.ID) {
        // i have no outgoing edges i will keep; go ahead and clear
        if (!hostHasOutgoing.test(h)) {
          numOutgoingEdges[h].clear();
        }
        continue;
      }
      // send outgoing edges data off to comm partner
      galois::runtime::SendBuffer b;

      // only send if non-zeros exist
      if (hostHasOutgoing.test(h)) {
        galois::runtime::gSerialize(b, 1); // token saying data exists
        galois::gPrint("data\n");
        auto before = b.size();
        galois::runtime::gSerialize(b, numOutgoingEdges[h]);
        galois::gPrint("[", base_DistGraph::id, "] bytes sent to ", h, " is ", b.size() - before, "\n");
      } else {
        galois::runtime::gSerialize(b, 0); // token saying no data exists
      }
      numOutgoingEdges[h].clear();

      // determine form to send bitset in
      auto& curBitset = hasIncomingEdge[h];
      uint64_t bitsetSize = curBitset.size(); // num bits
      uint64_t onlyOffsetsSize = curBitset.count() * 32;
      galois::gPrint("b size ", bitsetSize, " offset ", onlyOffsetsSize, "\n");
      if (bitsetSize == 0) {
        // there was nothing there to send in first place
        galois::runtime::gSerialize(b, 0);
      } else if (onlyOffsetsSize <= bitsetSize) {
        // send only offsets
        std::vector<uint32_t> offsets = curBitset.getOffsets();
        galois::gPrint("only offsets\n");
        galois::runtime::gSerialize(b, 2); // 2 = only offsets
        galois::runtime::gSerialize(b, offsets);
      } else {
        // send entire bitset
        galois::runtime::gSerialize(b, 1);
        galois::runtime::gSerialize(b, curBitset);
      }
      // get memory from bitset back
      curBitset.resize(0);

      bytesSent.update(b.size());

      // send buffer and free memory
      net.sendTagged(h, galois::runtime::evilPhase, b);
      b.getVec().clear();
    }

    galois::runtime::reportStat_Tsum(
      GRNAME, std::string("EdgeInspectionBytesSent"), bytesSent.reduce()
    );
  }

  /**
   * Receive data from inspection from other hosts. Processes the incoming
   * edge bitsets/offsets.
   *
   * @param[in,out] numOutgoingEdges specifies which nodes on a host will have
   * outgoing edges
   * @param[in,out] hasIncomingEdge indicates which nodes (that need to be
   * created) on this host have incoming edges
   */
  void recvInspectionData(std::vector<std::vector<uint64_t>>& numOutgoingEdges,
                          galois::DynamicBitSet& hasIncomingEdge) {
    auto& net = galois::runtime::getSystemNetworkInterface();

    for (unsigned h = 0; h < net.Num - 1; h++) {
      // expect data from comm partner back
      decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;
      do {
        p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
      } while (!p);

      uint32_t sendingHost = p->first;

      // get outgoing edges; first get status var
      uint32_t outgoingExists = 2;
      galois::runtime::gDeserialize(p->second, outgoingExists);

      if (outgoingExists == 1) {
        // actual data sent
        galois::runtime::gDeserialize(p->second, numOutgoingEdges[sendingHost]);
      } else if (outgoingExists == 0) {
        // no data sent; just clear again
        numOutgoingEdges[sendingHost].clear();
      } else {
        GALOIS_DIE("invalid recv inspection data metadata mode, outgoing");
      }

      uint32_t bitsetMetaMode = 3; // initialize to invalid mode
      galois::runtime::gDeserialize(p->second, bitsetMetaMode);
      if (bitsetMetaMode == 1) {
        // sent as bitset; deserialize then or with main bitset
        galois::DynamicBitSet recvSet;
        galois::runtime::gDeserialize(p->second, recvSet);
        hasIncomingEdge.bitwise_or(recvSet);
      } else if (bitsetMetaMode == 2) {
        // sent as vector of offsets
        std::vector<uint32_t> recvOffsets;
        galois::runtime::gDeserialize(p->second, recvOffsets);
        for (uint32_t offset : recvOffsets) {
          hasIncomingEdge.set(offset);
        }
      } else if (bitsetMetaMode == 0) {
        // do nothing; there was nothing to receive
      } else {
        GALOIS_DIE("invalid recv inspection data metadata mode");
      }
    }
  }

  /**
   * Take inspection metadata and being mapping nodes/creating prefix sums,
   * return the prefix sum.
   */
  galois::gstl::Vector<uint64_t> nodeMapping(
    std::vector<std::vector<uint64_t>>& numOutgoingEdges,
    galois::DynamicBitSet& hasIncomingEdge
  ) {
    numNodes = 0;
    numEdges = 0;
    nodesToReceive = 0;

    galois::gstl::Vector<uint64_t> prefixSumOfEdges;
    // reserve overestimation of nodes
    prefixSumOfEdges.reserve(base_DistGraph::numGlobalNodes /
                             base_DistGraph::numHosts * 1.15);
    localToGlobalVector.reserve(base_DistGraph::numGlobalNodes /
                                base_DistGraph::numHosts * 1.15);

    inspectMasterNodes(numOutgoingEdges, prefixSumOfEdges);
    inspectOutgoingNodes(numOutgoingEdges, prefixSumOfEdges);
    createIntermediateMetadata(prefixSumOfEdges, hasIncomingEdge.count());
    inspectIncomingNodes(hasIncomingEdge, prefixSumOfEdges);
    finalizeInspection(prefixSumOfEdges);

    galois::gDebug("[", base_DistGraph::id, "] To receive this many nodes: ",
                   nodesToReceive);

    return prefixSumOfEdges;
  }

  /**
   * Inspect master nodes; loop over all nodes, determine if master; if is,
   * create mapping + get num edges
   */
  void inspectMasterNodes(
    std::vector<std::vector<uint64_t>>& numOutgoingEdges,
    galois::gstl::Vector<uint64_t>& prefixSumOfEdges
  ) {
    uint32_t myHID = base_DistGraph::id;

    galois::GAccumulator<uint32_t> toReceive;
    toReceive.reset();

    for (unsigned h = 0; h < base_DistGraph::numHosts; ++h) {
      uint32_t activeThreads = galois::getActiveThreads();
      std::vector<uint64_t> threadPrefixSums(activeThreads);
      uint64_t startNode = base_DistGraph::gid2host[h].first;
      uint64_t lastNode  = base_DistGraph::gid2host[h].second;
      size_t hostSize = lastNode - startNode;

      if (numOutgoingEdges[h].size() != 0) {
        assert(hostSize == numOutgoingEdges[h].size());
      }

      // for each thread, figure out how many items it will work with (only
      // owned nodes)
      galois::on_each(
        [&](unsigned tid, unsigned nthreads) {
          size_t beginNode;
          size_t endNode;
          // loop over all nodes that host h has read
          std::tie(beginNode, endNode) = galois::block_range((size_t)0,
                                             hostSize, tid, nthreads);
          uint64_t count = 0;
          for (size_t i = beginNode; i < endNode; i++) {
            if (graphPartitioner->getMaster(i + startNode) == myHID) {
              count++;
            }
          }
          threadPrefixSums[tid] = count;
        }
      );

      // get prefix sums
      for (unsigned int i = 1; i < threadPrefixSums.size(); i++) {
        threadPrefixSums[i] += threadPrefixSums[i - 1];
      }

      assert(prefixSumOfEdges.size() == numNodes);
      assert(localToGlobalVector.size() == numNodes);

      uint32_t newMasterNodes = threadPrefixSums[activeThreads - 1];
      uint32_t startingNodeIndex = numNodes;
      // increase size of prefix sum + mapping vector
      prefixSumOfEdges.resize(numNodes + newMasterNodes);
      localToGlobalVector.resize(numNodes + newMasterNodes);

      if (newMasterNodes > 0) {
        // do actual work, second on_each
        galois::on_each(
          [&] (unsigned tid, unsigned nthreads) {
            size_t beginNode;
            size_t endNode;
            std::tie(beginNode, endNode) = galois::block_range((size_t)0,
                                             hostSize, tid, nthreads);

            // start location to start adding things into prefix sums/vectors
            uint32_t threadStartLocation = 0;
            if (tid != 0) {
              threadStartLocation = threadPrefixSums[tid - 1];
            }

            uint32_t handledNodes = 0;
            for (size_t i = beginNode; i < endNode; i++) {
              uint32_t globalID = startNode + i;
              // if this node is master, get outgoing edges + save mapping
              if (graphPartitioner->getMaster(globalID) == myHID) {
                // check size
                if (numOutgoingEdges[h].size() > 0) {
                  uint64_t myEdges = numOutgoingEdges[h][i];
                  numOutgoingEdges[h][i] = 0; // set to 0; does not need to be
                                              // handled later
                  prefixSumOfEdges[startingNodeIndex + threadStartLocation +
                                   handledNodes] = myEdges;
                  if (myEdges > 0 && h != myHID) {
                    toReceive += 1;
                  }
                } else {
                  prefixSumOfEdges[startingNodeIndex + threadStartLocation +
                                   handledNodes] = 0;
                }

                localToGlobalVector[startingNodeIndex + threadStartLocation +
                                    handledNodes] = globalID;
                handledNodes++;
              }
            }
          }
        );
        numNodes += newMasterNodes;
      }
    }

    nodesToReceive += toReceive.reduce();
    // masters have been handled
    base_DistGraph::numOwned = numNodes;
  }

  /**
   * Outgoing inspection: loop over all nodes, determnine if outgoing exists;
   * if does, create mapping, get edges
   */
  void inspectOutgoingNodes(
    std::vector<std::vector<uint64_t>>& numOutgoingEdges,
    galois::gstl::Vector<uint64_t>& prefixSumOfEdges
  ) {
    uint32_t myHID = base_DistGraph::id;

    galois::GAccumulator<uint32_t> toReceive;
    toReceive.reset();

    for (unsigned h = 0; h < base_DistGraph::numHosts; ++h) {
      size_t hostSize = numOutgoingEdges[h].size();
      // if i got no outgoing info from this host, safely continue to next one
      if (hostSize == 0) {
        continue;
      }

      uint32_t activeThreads = galois::getActiveThreads();
      std::vector<uint64_t> threadPrefixSums(activeThreads);

      // for each thread, figure out how many items it will work with (only
      // owned nodes)
      galois::on_each(
        [&](unsigned tid, unsigned nthreads) {
          size_t beginNode;
          size_t endNode;
          std::tie(beginNode, endNode) = galois::block_range((size_t)0,
                                             hostSize, tid, nthreads);
          uint64_t count = 0;
          for (size_t i = beginNode; i < endNode; i++) {
            if (numOutgoingEdges[h][i] > 0) {
              count++;
            }
          }
          threadPrefixSums[tid] = count;
        }
      );

      // get prefix sums
      for (unsigned int i = 1; i < threadPrefixSums.size(); i++) {
        threadPrefixSums[i] += threadPrefixSums[i - 1];
      }

      assert(prefixSumOfEdges.size() == numNodes);
      assert(localToGlobalVector.size() == numNodes);

      uint32_t newOutgoingNodes = threadPrefixSums[activeThreads - 1];
      // increase size of prefix sum + mapping vector
      prefixSumOfEdges.resize(numNodes + newOutgoingNodes);
      localToGlobalVector.resize(numNodes + newOutgoingNodes);

      uint64_t startNode = base_DistGraph::gid2host[h].first;
      uint32_t startingNodeIndex = numNodes;


      if (newOutgoingNodes > 0) {
        // do actual work, second on_each
        galois::on_each(
          [&] (unsigned tid, unsigned nthreads) {
            size_t beginNode;
            size_t endNode;
            std::tie(beginNode, endNode) = galois::block_range((size_t)0,
                                             hostSize, tid, nthreads);

            // start location to start adding things into prefix sums/vectors
            uint32_t threadStartLocation = 0;
            if (tid != 0) {
              threadStartLocation = threadPrefixSums[tid - 1];
            }

            uint32_t handledNodes = 0;

            for (size_t i = beginNode; i < endNode; i++) {
              uint64_t myEdges = numOutgoingEdges[h][i];
              if (myEdges > 0) {
                prefixSumOfEdges[startingNodeIndex + threadStartLocation +
                                 handledNodes] = myEdges;
                localToGlobalVector[startingNodeIndex + threadStartLocation +
                                    handledNodes] = startNode + i;
                handledNodes++;

                if (myEdges > 0 && h != myHID) {
                  toReceive += 1;
                }
              }
            }
          }
        );
        numNodes += newOutgoingNodes;
      }
      // don't need anymore after this point; get memory back
      numOutgoingEdges[h].clear();
    }

    nodesToReceive += toReceive.reduce();
    base_DistGraph::numNodesWithEdges = numNodes;
  }

  /**
   * Create a part of the global to local map (it's missing the incoming
   * mirrors with no edges) + part of prefix sum
   *
   * @param[in, out] prefixSumOfEdges edge prefix sum to build
   * @param[in] incomingEstimate estimate of number of incoming nodes to build
   */
  void createIntermediateMetadata(
    galois::gstl::Vector<uint64_t>& prefixSumOfEdges,
    const uint64_t incomingEstimate
  ) {
    if (numNodes == 0) {
      return;
    }
    globalToLocalMap.reserve(base_DistGraph::numNodesWithEdges + incomingEstimate);
    globalToLocalMap[localToGlobalVector[0]] = 0;
    // global to local map construction using num nodes with edges
    for (unsigned i = 1; i < base_DistGraph::numNodesWithEdges; i++) {
      prefixSumOfEdges[i] += prefixSumOfEdges[i - 1];
      globalToLocalMap[localToGlobalVector[i]] = i;
    }
  }

  /**
   * incoming node creation if is doesn't already exist + if actually amrked
   * as having incoming node
   */
  void inspectIncomingNodes(galois::DynamicBitSet& hasIncomingEdge,
                            galois::gstl::Vector<uint64_t>& prefixSumOfEdges) {
    uint32_t totalNumNodes = base_DistGraph::numGlobalNodes;

    uint32_t activeThreads = galois::getActiveThreads();
    std::vector<uint64_t> threadPrefixSums(activeThreads);

    galois::on_each(
      [&](unsigned tid, unsigned nthreads) {
        size_t beginNode;
        size_t endNode;
        std::tie(beginNode, endNode) = galois::block_range(0u,
                                         totalNumNodes, tid, nthreads);
        uint64_t count = 0;
        for (size_t i = beginNode; i < endNode; i++) {
          // only count if doesn't exist in global/local map + is incoming
          // edge
          if (hasIncomingEdge.test(i) && !globalToLocalMap.count(i)) ++count;
        }
        threadPrefixSums[tid] = count;
      }
    );
    // get prefix sums
    for (unsigned int i = 1; i < threadPrefixSums.size(); i++) {
      threadPrefixSums[i] += threadPrefixSums[i - 1];
    }

    assert(prefixSumOfEdges.size() == numNodes);
    assert(localToGlobalVector.size() == numNodes);

    uint32_t newIncomingNodes = threadPrefixSums[activeThreads - 1];
    // increase size of prefix sum + mapping vector
    prefixSumOfEdges.resize(numNodes + newIncomingNodes);
    localToGlobalVector.resize(numNodes + newIncomingNodes);

    uint32_t startingNodeIndex = numNodes;

    if (newIncomingNodes > 0) {
      // do actual work, second on_each
      galois::on_each(
        [&] (unsigned tid, unsigned nthreads) {
          size_t beginNode;
          size_t endNode;
          std::tie(beginNode, endNode) = galois::block_range(0u,
                                           totalNumNodes, tid, nthreads);

          // start location to start adding things into prefix sums/vectors
          uint32_t threadStartLocation = 0;
          if (tid != 0) {
            threadStartLocation = threadPrefixSums[tid - 1];
          }

          uint32_t handledNodes = 0;

          for (size_t i = beginNode; i < endNode; i++) {
            if (hasIncomingEdge.test(i) && !globalToLocalMap.count(i)) {
              prefixSumOfEdges[startingNodeIndex + threadStartLocation +
                               handledNodes] = 0;
              localToGlobalVector[startingNodeIndex + threadStartLocation +
                                  handledNodes] = i;
              handledNodes++;
            }
          }
        }
      );
      numNodes += newIncomingNodes;
    }
  }

  /**
   * finalize metadata maps
   */
  void finalizeInspection(galois::gstl::Vector<uint64_t>& prefixSumOfEdges) {
    // reserve rest of memory needed
    globalToLocalMap.reserve(numNodes);
    for (unsigned i = base_DistGraph::numNodesWithEdges; i < numNodes; i++) {
      // finalize prefix sum
      prefixSumOfEdges[i] += prefixSumOfEdges[i - 1];
      // global to local map construction
      globalToLocalMap[localToGlobalVector[i]] = i;
    }
    if (prefixSumOfEdges.size() != 0) {
      numEdges = prefixSumOfEdges.back();
    } else {
      numEdges = 0;
    }
  }

////////////////////////////////////////////////////////////////////////////////

  /**
   * Fill up mirror arrays.
   * TODO make parallel?
   */
  void fillMirrors() {
    base_DistGraph::mirrorNodes.reserve(numNodes - base_DistGraph::numOwned);
    for (uint32_t i = base_DistGraph::numOwned; i < numNodes; i++) {
      uint32_t globalID = localToGlobalVector[i];
      base_DistGraph::mirrorNodes[graphPartitioner->getMaster(globalID)].
              push_back(globalID);
    }
  }

////////////////////////////////////////////////////////////////////////////////

  template <typename GraphTy>
  void loadEdges(GraphTy& graph,
                 galois::graphs::BufferedGraph<EdgeTy>& bufGraph) {
    if (base_DistGraph::id == 0) {
      if (std::is_void<typename GraphTy::edge_data_type>::value) {
        fprintf(stderr, "Loading void edge-data while creating edges.\n");
      } else {
        fprintf(stderr, "Loading edge-data while creating edges.\n");
      }
    }

    bufGraph.resetReadCounters();

    std::atomic<uint32_t> receivedNodes;
    receivedNodes.store(0);

    galois::StatTimer loadEdgeTimer("EdgeLoading", GRNAME);
    loadEdgeTimer.start();

    // sends data
    sendEdges(graph, bufGraph, receivedNodes);
    // receives data
    galois::on_each([&](unsigned tid, unsigned nthreads) {
      receiveEdges(graph, receivedNodes);
    });
    base_DistGraph::increment_evilPhase();

    loadEdgeTimer.stop();

    galois::gPrint("[", base_DistGraph::id, "] Edge loading time: ",
                   loadEdgeTimer.get_usec() / 1000000.0f,
                   " seconds to read ", bufGraph.getBytesRead(), " bytes (",
                   bufGraph.getBytesRead() / (float)loadEdgeTimer.get_usec(),
                   " MBPS)\n");
  }

  // Edge type is not void. (i.e. edge data exists)
  template <typename GraphTy,
            typename std::enable_if<!std::is_void<
                typename GraphTy::edge_data_type>::value>::type* = nullptr>
  void sendEdges(GraphTy& graph,
                 galois::graphs::BufferedGraph<EdgeTy>& bufGraph,
                 std::atomic<uint32_t>& receivedNodes) {
    using DstVecType = std::vector<std::vector<uint64_t>>;
    using DataVecType =
        std::vector<std::vector<typename GraphTy::edge_data_type>>;
    using SendBufferVecTy = std::vector<galois::runtime::SendBuffer>;

    galois::substrate::PerThreadStorage<DstVecType> gdst_vecs(
        base_DistGraph::numHosts);
    galois::substrate::PerThreadStorage<DataVecType> gdata_vecs(
        base_DistGraph::numHosts);
    galois::substrate::PerThreadStorage<SendBufferVecTy> sendBuffers(
        base_DistGraph::numHosts);

    auto& net             = galois::runtime::getSystemNetworkInterface();
    const unsigned& id       = this->base_DistGraph::id;
    const unsigned& numHosts = this->base_DistGraph::numHosts;

    galois::GAccumulator<uint64_t> messagesSent;
    galois::GAccumulator<uint64_t> bytesSent;
    galois::GReduceMax<uint64_t> maxBytesSent;
    messagesSent.reset();
    bytesSent.reset();
    maxBytesSent.reset();

    // Go over assigned nodes and distribute edges.
    galois::do_all(
      galois::iterate(base_DistGraph::gid2host[base_DistGraph::id].first,
                      base_DistGraph::gid2host[base_DistGraph::id].second),
      [&](auto src) {
        uint32_t lsrc       = 0;
        uint64_t curEdge    = 0;
        if (this->isLocal(src)) {
          lsrc = this->G2L(src);
          curEdge = *graph.edge_begin(lsrc, galois::MethodFlag::UNPROTECTED);
        }

        auto ee     = bufGraph.edgeBegin(src);
        auto ee_end = bufGraph.edgeEnd(src);
        uint64_t numEdges = std::distance(ee, ee_end);
        auto& gdst_vec  = *gdst_vecs.getLocal();
        auto& gdata_vec = *gdata_vecs.getLocal();

        for (unsigned i = 0; i < numHosts; ++i) {
          gdst_vec[i].clear();
          gdata_vec[i].clear();
          gdst_vec[i].reserve(numEdges);
          gdata_vec[i].reserve(numEdges);
        }

        for (; ee != ee_end; ++ee) {
          uint32_t gdst = bufGraph.edgeDestination(*ee);
          auto gdata    = bufGraph.edgeData(*ee);

          uint32_t hostBelongs =
            graphPartitioner->getEdgeOwner(src, gdst, numEdges);
          if (hostBelongs == id) {
            // edge belongs here, construct on self
            assert(this->isLocal(src));
            uint32_t ldst = this->G2L(gdst);
            graph.constructEdge(curEdge++, ldst, gdata);
            // TODO
            // if ldst is an outgoing mirror, this is vertex cut
          } else {
            // add to host vector to send out later
            gdst_vec[hostBelongs].push_back(gdst);
            gdata_vec[hostBelongs].push_back(gdata);
          }
        }

        // make sure all edges accounted for if local
        if (this->isLocal(src)) {
          assert(curEdge == (*graph.edge_end(lsrc)));
        }

        // send
        for (uint32_t h = 0; h < numHosts; ++h) {
          if (h == id) continue;

          if (gdst_vec[h].size() > 0) {
            auto& b = (*sendBuffers.getLocal())[h];
            galois::runtime::gSerialize(b, src);
            galois::runtime::gSerialize(b, gdst_vec[h]);
            galois::runtime::gSerialize(b, gdata_vec[h]);

            // send if over limit
            if (b.size() > edgePartitionSendBufSize) {
              messagesSent += 1;
              bytesSent.update(b.size());
              maxBytesSent.update(b.size());

              net.sendTagged(h, galois::runtime::evilPhase, b);
              b.getVec().clear();
              b.getVec().reserve(edgePartitionSendBufSize * 1.25);
            }
          }
        }

        // overlap receives
        auto buffer = net.recieveTagged(galois::runtime::evilPhase, nullptr);
        this->processReceivedEdgeBuffer(buffer, graph, receivedNodes);
      },
      #if MORE_DIST_STATS
      galois::loopname("EdgeLoading"),
      #endif
      galois::steal(),
      galois::no_stats()
    );

    // flush buffers
    for (unsigned threadNum = 0; threadNum < sendBuffers.size(); ++threadNum) {
      auto& sbr = *sendBuffers.getRemote(threadNum);
      for (unsigned h = 0; h < this->base_DistGraph::numHosts; ++h) {
        if (h == this->base_DistGraph::id) continue;
        auto& sendBuffer = sbr[h];
        if (sendBuffer.size() > 0) {
          messagesSent += 1;
          bytesSent.update(sendBuffer.size());
          maxBytesSent.update(sendBuffer.size());

          net.sendTagged(h, galois::runtime::evilPhase, sendBuffer);
          sendBuffer.getVec().clear();
        }
      }
    }

    net.flush();

    galois::runtime::reportStat_Tsum(
      GRNAME, std::string("EdgeLoadingMessagesSent"), messagesSent.reduce()
    );
    galois::runtime::reportStat_Tsum(
      GRNAME, std::string("EdgeLoadingBytesSent"), bytesSent.reduce()
    );
    galois::runtime::reportStat_Tmax(
      GRNAME, std::string("EdgeLoadingMaxBytesSent"), maxBytesSent.reduce()
    );
  }

  // no edge data version
  template <typename GraphTy,
            typename std::enable_if<std::is_void<
                typename GraphTy::edge_data_type>::value>::type* = nullptr>
  void sendEdges(GraphTy& graph,
                 galois::graphs::BufferedGraph<EdgeTy>& bufGraph,
                 std::atomic<uint32_t>& receivedNodes) {
    using DstVecType = std::vector<std::vector<uint64_t>>;
    using SendBufferVecTy = std::vector<galois::runtime::SendBuffer>;

    galois::substrate::PerThreadStorage<DstVecType> gdst_vecs(
        base_DistGraph::numHosts);
    galois::substrate::PerThreadStorage<SendBufferVecTy> sendBuffers(
        base_DistGraph::numHosts);

    auto& net                = galois::runtime::getSystemNetworkInterface();
    const unsigned& id       = this->base_DistGraph::id;
    const unsigned& numHosts = this->base_DistGraph::numHosts;

    galois::GAccumulator<uint64_t> messagesSent;
    galois::GAccumulator<uint64_t> bytesSent;
    galois::GReduceMax<uint64_t> maxBytesSent;
    messagesSent.reset();
    bytesSent.reset();
    maxBytesSent.reset();

    // Go over assigned nodes and distribute edges.
    galois::do_all(
      galois::iterate(base_DistGraph::gid2host[base_DistGraph::id].first,
                      base_DistGraph::gid2host[base_DistGraph::id].second),
      [&](auto src) {
        uint32_t lsrc       = 0;
        uint64_t curEdge    = 0;
        if (this->isLocal(src)) {
          lsrc = this->G2L(src);
          curEdge = *graph.edge_begin(lsrc, galois::MethodFlag::UNPROTECTED);
        }

        auto ee     = bufGraph.edgeBegin(src);
        auto ee_end = bufGraph.edgeEnd(src);
        uint64_t numEdges = std::distance(ee, ee_end);
        auto& gdst_vec  = *gdst_vecs.getLocal();

        for (unsigned i = 0; i < numHosts; ++i) {
          gdst_vec[i].clear();
          gdst_vec[i].reserve(numEdges);
        }

        for (; ee != ee_end; ++ee) {
          uint32_t gdst = bufGraph.edgeDestination(*ee);
          uint32_t hostBelongs =
            graphPartitioner->getEdgeOwner(src, gdst, numEdges);

          if (hostBelongs == id) {
            // edge belongs here, construct on self
            assert(this->isLocal(src));
            uint32_t ldst = this->G2L(gdst);
            graph.constructEdge(curEdge++, ldst);
            // TODO
            // if ldst is an outgoing mirror, this is vertex cut
          } else {
            // add to host vector to send out later
            gdst_vec[hostBelongs].push_back(gdst);
          }
        }

        // make sure all edges accounted for if local
        if (this->isLocal(src)) {
          assert(curEdge == (*graph.edge_end(lsrc)));
        }

        // send
        for (uint32_t h = 0; h < numHosts; ++h) {
          if (h == id) continue;

          if (gdst_vec[h].size() > 0) {
            auto& b = (*sendBuffers.getLocal())[h];
            galois::runtime::gSerialize(b, src);
            galois::runtime::gSerialize(b, gdst_vec[h]);

            // send if over limit
            if (b.size() > edgePartitionSendBufSize) {
              messagesSent += 1;
              bytesSent.update(b.size());
              maxBytesSent.update(b.size());

              net.sendTagged(h, galois::runtime::evilPhase, b);
              b.getVec().clear();
              b.getVec().reserve(edgePartitionSendBufSize * 1.25);
            }
          }
        }

        // overlap receives
        auto buffer = net.recieveTagged(galois::runtime::evilPhase, nullptr);
        this->processReceivedEdgeBuffer(buffer, graph, receivedNodes);
      },
      #if MORE_DIST_STATS
      galois::loopname("EdgeLoading"),
      #endif
      galois::steal(),
      galois::no_stats()
    );

    // flush buffers
    for (unsigned threadNum = 0; threadNum < sendBuffers.size(); ++threadNum) {
      auto& sbr = *sendBuffers.getRemote(threadNum);
      for (unsigned h = 0; h < this->base_DistGraph::numHosts; ++h) {
        if (h == this->base_DistGraph::id) continue;
        auto& sendBuffer = sbr[h];
        if (sendBuffer.size() > 0) {
          messagesSent += 1;
          bytesSent.update(sendBuffer.size());
          maxBytesSent.update(sendBuffer.size());

          net.sendTagged(h, galois::runtime::evilPhase, sendBuffer);
          sendBuffer.getVec().clear();
        }
      }
    }

    net.flush();

    galois::runtime::reportStat_Tsum(
      GRNAME, std::string("EdgeLoadingMessagesSent"), messagesSent.reduce()
    );
    galois::runtime::reportStat_Tsum(
      GRNAME, std::string("EdgeLoadingBytesSent"), bytesSent.reduce()
    );
    galois::runtime::reportStat_Tmax(
      GRNAME, std::string("EdgeLoadingMaxBytesSent"), maxBytesSent.reduce()
    );
  }

  //! Optional type
  //! @tparam T type that the variable may possibly take
  template <typename T>
#if __GNUC__ > 5 || (__GNUC__ == 5 && __GNUC_MINOR__ > 1)
  using optional_t = std::experimental::optional<T>;
#else
  using optional_t = boost::optional<T>;
#endif
  //! @copydoc DistGraphHybridCut::processReceivedEdgeBuffer
  template <typename GraphTy>
  void processReceivedEdgeBuffer(
      optional_t<std::pair<uint32_t, galois::runtime::RecvBuffer>>& buffer,
      GraphTy& graph, std::atomic<uint32_t>& receivedNodes) {
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
        ++receivedNodes;
      }
    }
  }

  /**
   * Receive the edge dest/data assigned to this host from other hosts
   * that were responsible for reading them.
   */
  template <typename GraphTy>
  void receiveEdges(GraphTy& graph, std::atomic<uint32_t>& receivedNodes) {
    auto& net = galois::runtime::getSystemNetworkInterface();

    // receive edges for all mirror nodes
    while (receivedNodes < nodesToReceive) {
      decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;
      p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
      processReceivedEdgeBuffer(p, graph, receivedNodes);
    }
  }

  template <typename GraphTy,
            typename std::enable_if<!std::is_void<
                typename GraphTy::edge_data_type>::value>::type* = nullptr>
  void deserializeEdges(GraphTy& graph, galois::runtime::RecvBuffer& b,
                        std::vector<uint64_t>& gdst_vec, uint64_t& cur,
                        uint64_t& cur_end) {
    std::vector<typename GraphTy::edge_data_type> gdata_vec;
    galois::runtime::gDeserialize(b, gdata_vec);
    uint64_t i = 0;
    while (cur < cur_end) {
      auto gdata    = gdata_vec[i];
      uint64_t gdst = gdst_vec[i++];
      uint32_t ldst = G2L(gdst);
      graph.constructEdge(cur++, ldst, gdata);
      // TODO
      // if ldst is an outgoing mirror, this is vertex cut
    }
  }

  template <typename GraphTy,
            typename std::enable_if<std::is_void<
                typename GraphTy::edge_data_type>::value>::type* = nullptr>
  void deserializeEdges(GraphTy& graph, galois::runtime::RecvBuffer& b,
                        std::vector<uint64_t>& gdst_vec, uint64_t& cur,
                        uint64_t& cur_end) {
    uint64_t i = 0;
    while (cur < cur_end) {
      uint64_t gdst = gdst_vec[i++];
      uint32_t ldst = G2L(gdst);
      graph.constructEdge(cur++, ldst);
      // TODO
      // if ldst is an outgoing mirror, this is vertex cut
    }
  }

 public:
  /**
   * Reset bitset
   */
  void reset_bitset(typename base_DistGraph::SyncType syncType,
                    void (*bitset_reset_range)(size_t, size_t)) const {
    // layout: masters.... outgoing mirrors.... incoming mirrors
    // note the range for bitset reset range is inclusive
    if (base_DistGraph::numOwned > 0) {
      if (syncType == base_DistGraph::syncBroadcast) { // reset masters
        bitset_reset_range(0, base_DistGraph::numOwned - 1);
      } else {
        assert(syncType == base_DistGraph::syncReduce);
        // mirrors occur after masters
        if (base_DistGraph::numOwned < numNodes) {
          bitset_reset_range(base_DistGraph::numOwned, numNodes - 1);
        }
      }
    } else { // all things are mirrors
      // only need to reset if reduce
      if (syncType == base_DistGraph::syncReduce) {
        if (numNodes > 0) {
          bitset_reset_range(0, numNodes - 1);
        }
      }
    }
  }

  std::vector<std::pair<uint32_t, uint32_t>> getMirrorRanges() const {
    std::vector<std::pair<uint32_t, uint32_t>> mirrorRangesVector;
    // order of nodes locally is masters, outgoing mirrors, incoming mirrors,
    // so just get from numOwned to end
    mirrorRangesVector.push_back(std::make_pair(base_DistGraph::numOwned,
                                                numNodes));
    return mirrorRangesVector;
  }

  // TODO current uses graph partitioner
  // TODO make it so user doens't have to specify; can be done by tracking
  // if an outgoing mirror is marked as having an incoming edge on any
  // host
  bool is_vertex_cut() const {
    return graphPartitioner->isVertexCut();
  }
};

// make GRNAME visible to public
template <typename NodeTy, typename EdgeTy, typename Partitioner>
constexpr const char* const
    galois::graphs::DistGraphGeneric<NodeTy, EdgeTy, Partitioner>::GRNAME;

} // end namespace graphs
} // end namespace galois
#endif
