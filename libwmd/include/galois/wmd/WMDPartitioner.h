/**
 * @file WMDPartitioner.h
 *
 * Graph partitioning that duplicates edges for WMD dataset. Currently only
 * supports an outgoing edge cut.
 *
 */

#ifndef _WMD_PARTITIONER_H
#define _WMD_PARTITIONER_H

#include "galois/Galois.h"
#include "galois/graphs/DistributedGraph.h"
#include "galois/DReducible.h"

#include "WMDGraph.h"
#include "instrument.h"

#include <atomic>
#include <unistd.h>
#include <ios>
#include <iostream>
#include <fstream>
#include <string>

namespace galois {
namespace graphs {
/**
 * @tparam NodeTy type of node data for the graph
 * @tparam EdgeTy type of edge data for the graph
 *
 * @todo fully document and clean up code
 * @warning not meant for public use + not fully documented yet
 */
template <typename NodeTy, typename EdgeTy, typename Partitioner>
class WMDGraph : public DistGraph<NodeTy, EdgeTy> {

  //! size used to buffer edge sends during partitioning
  constexpr static unsigned edgePartitionSendBufSize = 8388608;
  constexpr static const char* const GRNAME          = "dGraph_WMD";
  std::unique_ptr<Partitioner> graphPartitioner;

  uint32_t G2LEdgeCut(uint64_t gid, uint32_t globalOffset) const {
    assert(base_DistGraph::isLocal(gid));
    // optimized for edge cuts
    if (gid >= globalOffset && gid < globalOffset + base_DistGraph::numOwned)
      return gid - globalOffset;

    return base_DistGraph::globalToLocalMap.at(gid);
  }

  /**
   * Free memory of a vector by swapping an empty vector with it
   */
  template <typename V>
  void freeVector(V& vectorToKill) {
    V dummyVector;
    vectorToKill.swap(dummyVector);
  }

  uint32_t nodesToReceive;

  uint64_t myKeptEdges;
  uint64_t globalKeptEdges;
  uint64_t totalEdgeProxies;

  std::vector<std::vector<size_t>> mirrorEdges;
  std::unordered_map<uint64_t, uint64_t> localEdgeGIDToLID;

  template <typename, typename, typename>
  friend class WMDGraph;

  virtual unsigned getHostIDImpl(uint64_t gid) const {
    assert(gid < base_DistGraph::numGlobalNodes);
    return graphPartitioner->retrieveMaster(gid);
  }

  virtual bool isOwnedImpl(uint64_t gid) const {
    assert(gid < base_DistGraph::numGlobalNodes);
    return (graphPartitioner->retrieveMaster(gid) == base_DistGraph::id);
  }

  virtual bool isLocalImpl(uint64_t gid) const {
    assert(gid < base_DistGraph::numGlobalNodes);
    return (base_DistGraph::globalToLocalMap.find(gid) !=
            base_DistGraph::globalToLocalMap.end());
  }

  virtual bool isVertexCutImpl() const { return false; }

public:
  //! typedef for base DistGraph class
  using base_DistGraph = DistGraph<NodeTy, EdgeTy>;

  /**
   * Returns edges owned by this graph (i.e. read).
   */
  uint64_t numOwnedEdges() const { return myKeptEdges; }

  /**
   * Returns # edges kept in all graphs.
   */
  uint64_t globalEdges() const { return globalKeptEdges; }

  std::vector<std::vector<size_t>>& getMirrorEdges() { return mirrorEdges; }

  /**
   * Return the reader of a particular node.
   * @param gid GID of node to get reader of
   * @return Host reader of node passed in as param
   */
  unsigned getHostReader(uint64_t gid) const {
    for (auto i = 0U; i < base_DistGraph::numHosts; ++i) {
      uint64_t start, end;
      std::tie(start, end) = base_DistGraph::gid2host[i];
      if (gid >= start && gid < end) {
        return i;
      }
    }
    return -1;
  }

  /**
   * Constructor
   */
  WMDGraph(
      std::vector<std::unique_ptr<galois::graphs::FileParser<NodeTy, EdgeTy>>>&
          parsers,
      unsigned host, unsigned _numHosts, bool setupGluon = true,
      bool doSort                             = false,
      galois::graphs::MASTERS_DISTRIBUTION md = BALANCED_EDGES_OF_MASTERS)
      : base_DistGraph(host, _numHosts) {
    galois::gInfo("[", base_DistGraph::id, "] Start DistGraph construction.");
    galois::runtime::reportParam(GRNAME, "WMDGraph", "0");
    // TODO: who is responsible for init and deinit?
    galois::StatTimer Tgraph_construct("GraphPartitioningTime", GRNAME);
    Tgraph_construct.start();

    ////////////////////////////////////////////////////////////////////////////
    galois::gInfo("[", base_DistGraph::id, "] Start reading graph.");
    galois::StatTimer graphReadTimer("GraphReading", GRNAME);
    graphReadTimer.start();

    galois::gDebug("[", base_DistGraph::id, "] WMDOfflineGraph End!");
    galois::graphs::WMDOfflineGraph<NodeTy, EdgeTy> g(parsers, md, 8);
    galois::gDebug("[", base_DistGraph::id, "] WMDOfflineGraph End!");
    std::vector<unsigned> dummy;

    // freeup memory that won't be used in the future
    g.clearEdgePrefixSumInfo();

    std::vector<uint64_t> ndegrees;

    graphPartitioner = std::make_unique<Partitioner>(
        host, _numHosts, base_DistGraph::numGlobalNodes,
        base_DistGraph::numGlobalEdges, ndegrees);

    graphReadTimer.stop();
    galois::gInfo("[", base_DistGraph::id, "] Reading graph complete in ",
                  graphReadTimer.get_usec() / 1000000.0, " sec.");
    ////////////////////////////////////////////////////////////////////////////
    galois::gInfo("[", base_DistGraph::id, "] Start exchanging edges.");
    galois::StatTimer edgesExchangeTimer("EdgesExchange", GRNAME);
    edgesExchangeTimer.start();

    // never read edge data from disk
    galois::graphs::WMDBufferedGraph<NodeTy, EdgeTy> bufGraph;
    bufGraph.loadPartialGraph(g, base_DistGraph::numGlobalEdges);

    edgesExchangeTimer.stop();
    galois::gInfo("[", base_DistGraph::id, "] Exchanging edges complete in ",
                  edgesExchangeTimer.get_usec() / 1000000.0, " sec.");
    ////////////////////////////////////////////////////////////////////////////
    galois::gInfo("[", base_DistGraph::id, "] Starting edge inspection.");
    galois::StatTimer inspectionTimer("EdgeInspection", GRNAME);
    inspectionTimer.start();
    base_DistGraph::numGlobalNodes = g.size();
    base_DistGraph::numGlobalEdges = g.sizeEdges();
    for (int k = 0; k < 2; k++)
      I_RR();
    I_WM(2);

    // galois::gstl::Vector<uint64_t> prefixSumOfEdges;
    // prefixSumOfEdges.resize(base_DistGraph::numOwned);

    // initial pass; set up lid-gid mappings, determine which proxies exist on
    // this host
    uint64_t nodeBegin = bufGraph.globalNodeOffset[base_DistGraph::id];
    uint64_t nodeEnd   = bufGraph.globalNodeOffset[base_DistGraph::id] +
                       bufGraph.localNodeSize[base_DistGraph::id];
    base_DistGraph::numOwned = bufGraph.localNodeSize[base_DistGraph::id];
    for (int k = 0; k < 4; k++)
      I_RR();
    I_WM(1);

    base_DistGraph::gid2host.resize(base_DistGraph::numHosts);
    for (uint64_t h = 0; h < base_DistGraph::numHosts - 1; h++) {
      base_DistGraph::gid2host[h] = std::pair<uint64_t, uint64_t>(
          bufGraph.globalNodeOffset[h], bufGraph.globalNodeOffset[h + 1]);
      for (int k = 0; k < 2; k++)
        I_RR();
      I_WM(1);
    }
    base_DistGraph::gid2host[base_DistGraph::numHosts - 1] =
        std::pair<uint64_t, uint64_t>(
            bufGraph.globalNodeOffset[base_DistGraph::numHosts - 1],
            base_DistGraph::numGlobalNodes);
    graphPartitioner->saveGIDToHost(bufGraph.virtualToPhyMapping);

    std::vector<std::vector<uint64_t>> presentProxies =
        edgeInspectionRound1(bufGraph);

    // vector to store bitsets received from other hosts
    std::vector<std::vector<uint64_t>> proxiesOnOtherHosts;
    proxiesOnOtherHosts.resize(_numHosts);

    // send off mirror proxies that exist on this host to other hosts
    communicateProxyInfo(presentProxies, proxiesOnOtherHosts);

    base_DistGraph::numEdges = bufGraph.sizeLocalEdges();
    // assumption: we keep all edges since mirror edges are not supported
    myKeptEdges     = base_DistGraph::numEdges;
    globalKeptEdges = base_DistGraph::numGlobalEdges;

    inspectionTimer.stop();
    galois::gInfo("[", base_DistGraph::id, "] Edge inspection complete in ",
                  inspectionTimer.get_usec() / 1000000.0, " sec.");
    ////////////////////////////////////////////////////////////////////////////
    galois::gInfo("[", base_DistGraph::id, "] Starting building LS_CSR.");
    galois::StatTimer buildingTimer("GraphBuilding", GRNAME);
    buildingTimer.start();

    // Graph construction related calls
    base_DistGraph::beginMaster = 0;
    // Allocate and construct the graph
    base_DistGraph::graph.allocateFrom(base_DistGraph::numNodes,
                                       base_DistGraph::numEdges);
    I_WM(base_DistGraph::numNodes);
    base_DistGraph::graph.constructNodes();

    // construct edges
    // not need to move edges from other host since all edges is already ready
    // when no edge mirror are used.
    galois::gDebug("[", base_DistGraph::id, "] add edges into graph.");
    galois::do_all(
        galois::iterate(nodeBegin, nodeEnd),
        [&](uint64_t globalID) {
          auto edgeDst = bufGraph.edgeLocalDst(globalID);
          I_RR();
          std::vector<uint64_t> dstData;
          for (auto dst : edgeDst) {
            dstData.emplace_back(base_DistGraph::globalToLocalMap[dst]);
            I_RR();
            I_WR();
          }
          auto edgeData = bufGraph.edgeDataPtr(globalID);
          I_RR();
          I_WM(bufGraph.edgeNum(globalID));
          base_DistGraph::graph.addEdgesUnSort(
              true, (globalID - bufGraph.globalNodeOffset[base_DistGraph::id]),
              dstData.data(), edgeData, bufGraph.edgeNum(globalID), false);
        },
        galois::steal());

    // move node data (include mirror nodes) from other hosts to graph in this
    // host
    galois::gDebug("[", base_DistGraph::id, "] add nodes data into graph.");
    bufGraph.gatherNodes(g, base_DistGraph::graph, proxiesOnOtherHosts,
                         base_DistGraph::numNodes,
                         base_DistGraph::globalToLocalMap);

    galois::gDebug("[", base_DistGraph::id, "] LS_CSR construction done.");
    galois::gInfo("[", base_DistGraph::id,
                  "] LS_CSR graph local nodes: ", base_DistGraph::numNodes);
    galois::gInfo("[", base_DistGraph::id,
                  "] LS_CSR graph master nodes: ", base_DistGraph::numOwned);
    galois::gInfo("[", base_DistGraph::id, "] LS_CSR graph local edges: ",
                  base_DistGraph::graph.sizeEdges());
    assert(base_DistGraph::graph.sizeEdges() == base_DistGraph::numEdges);
    assert(base_DistGraph::graph.size() == base_DistGraph::numNodes);

    bufGraph.resetAndFree();

    buildingTimer.stop();
    galois::gInfo("[", base_DistGraph::id, "] Building LS_CSR complete in ",
                  buildingTimer.get_usec() / 1000000.0, " sec.");
    ////////////////////////////////////////////////////////////////////////////

    if (setupGluon) {
      galois::CondStatTimer<MORE_DIST_STATS> TfillMirrors("FillMirrors",
                                                          GRNAME);

      TfillMirrors.start();
      fillMirrors();
      TfillMirrors.stop();
    }

    ////////////////////////////////////////////////////////////////////////////

    // TODO this might be useful to keep around
    proxiesOnOtherHosts.clear();
    proxiesOnOtherHosts.shrink_to_fit();
    ndegrees.clear();
    ndegrees.shrink_to_fit();

    // SORT EDGES
    if (doSort) {
      base_DistGraph::sortEdgesByDestination();
    }

    ////////////////////////////////////////////////////////////////////////////

    galois::CondStatTimer<MORE_DIST_STATS> Tthread_ranges("ThreadRangesTime",
                                                          GRNAME);

    galois::gInfo("[", base_DistGraph::id, "] Determining thread ranges");

    Tthread_ranges.start();
    base_DistGraph::determineThreadRanges();
    base_DistGraph::determineThreadRangesMaster();
    base_DistGraph::determineThreadRangesWithEdges();
    base_DistGraph::initializeSpecificRanges();
    Tthread_ranges.stop();

    Tgraph_construct.stop();
    galois::gInfo("[", base_DistGraph::id,
                  "] Total time of DistGraph construction is ",
                  Tgraph_construct.get_usec() / 1000000.0, " sec.");

    galois::DGAccumulator<uint64_t> accumer;
    accumer.reset();
    accumer += base_DistGraph::sizeEdges();
    totalEdgeProxies = accumer.reduce();

    uint64_t totalNodeProxies;
    accumer.reset();
    accumer += base_DistGraph::size();
    totalNodeProxies = accumer.reduce();

    // report some statistics
    if (base_DistGraph::id == 0) {
      galois::runtime::reportStat_Single(
          GRNAME, std::string("TotalNodeProxies"), totalNodeProxies);
      galois::runtime::reportStat_Single(
          GRNAME, std::string("TotalEdgeProxies"), totalEdgeProxies);
      galois::runtime::reportStat_Single(GRNAME,
                                         std::string("OriginalNumberEdges"),
                                         base_DistGraph::globalSizeEdges());
      galois::runtime::reportStat_Single(GRNAME, std::string("TotalKeptEdges"),
                                         globalKeptEdges);
      galois::runtime::reportStat_Single(
          GRNAME, std::string("ReplicationFactorNodes"),
          (totalNodeProxies) / (double)base_DistGraph::globalSize());
      galois::runtime::reportStat_Single(
          GRNAME, std::string("ReplicatonFactorEdges"),
          (totalEdgeProxies) / (double)globalKeptEdges);
    }
  }

  // this consumes the original graph
  // this does not support mirror edges
  template <class NewGraph, class Projection>
  std::unique_ptr<NewGraph> Project(Projection projection) {
    std::unique_ptr<NewGraph> newGraph = std::unique_ptr<NewGraph>(
        new NewGraph(base_DistGraph::id, base_DistGraph::numHosts));
    using NodeLID     = uint64_t;
    using NodeGID     = uint64_t;
    using NewEdgeType = typename NewGraph::EdgeType;

    galois::gInfo("[", base_DistGraph::id, "] Start projection.");

    newGraph->gid2host = base_DistGraph::gid2host;
    newGraph->localToGlobalVector.resize(base_DistGraph::numNodes);
    std::vector<bool> keepMirrors(base_DistGraph::numNodes -
                                  base_DistGraph::numOwned);
    // these 2 structures: newTopology and newEdgeData must mirror eachother
    std::vector<std::vector<NodeGID>> newTopology(base_DistGraph::numNodes);
    std::vector<std::vector<NewEdgeType>> newEdgeData(base_DistGraph::numNodes);

    std::atomic<uint64_t> masterNodes = 0;
    std::atomic<uint64_t> mirrorNodes = 0;
    galois::GAccumulator<uint64_t> nodesWithEdges;
    galois::DGAccumulator<uint64_t> globalNodes;
    galois::DGAccumulator<uint64_t> globalEdges;
    nodesWithEdges.reset();
    globalNodes.reset();
    globalEdges.reset();

    galois::do_all(
        galois::iterate(base_DistGraph::masterNodesRange().begin(),
                        base_DistGraph::masterNodesRange().end()),
        [&](auto& node) {
          I_RS();
          if (!projection.KeepNode(*this, node)) {
            return;
          }
          I_RR();
          NodeGID nodeGID = base_DistGraph::getGID(node);
          std::vector<NodeGID> edgeDsts;
          std::vector<NewEdgeType> keptEdgeData;

          uint64_t keptEdges = 0;
          for (const auto& edge : base_DistGraph::edges(node)) {
            I_RS();
            I_RR();
            EdgeTy edgeData = base_DistGraph::getEdgeData(edge);
            I_RR();
            NodeLID dstNode = base_DistGraph::getEdgeDst(edge);
            if (!projection.KeepEdge(*this, edgeData, node, dstNode)) {
              continue;
            }
            keptEdges++;
            I_RR();
            I_WR();
            edgeDsts.emplace_back(base_DistGraph::getGID(dstNode));
            I_WR();
            keptEdgeData.emplace_back(
                projection.ProjectEdge(*this, edgeData, node, dstNode));
            if (dstNode >= base_DistGraph::numOwned) {
              I_WR();
              keepMirrors[dstNode - base_DistGraph::numOwned] = true;
            }
          }
          if (projection.KeepEdgeLessMasters() || keptEdges > 0) {
            if (keptEdges > 0) {
              nodesWithEdges += 1;
            }
            globalNodes += 1;
            globalEdges += keptEdges;
            NodeLID nodeLID = masterNodes.fetch_add(1);
            I_WR();
            newGraph->localToGlobalVector[nodeLID] = nodeGID;
            newTopology[nodeLID]                   = std::move(edgeDsts);
            newEdgeData[nodeLID]                   = std::move(keptEdgeData);
          }
        });

    uint64_t numMasters = masterNodes;

    galois::do_all(galois::iterate(uint64_t(base_DistGraph::numOwned),
                                   uint64_t(base_DistGraph::numNodes)),
                   [&](auto& mirrorNode) {
                     I_RS();
                     I_RR();
                     if (!keepMirrors[mirrorNode - base_DistGraph::numOwned]) {
                       return;
                     }
                     I_RR();
                     NodeGID nodeGID = base_DistGraph::getGID(mirrorNode);
                     NodeLID nodeLID = numMasters + mirrorNodes.fetch_add(1);
                     I_WR();
                     newGraph->localToGlobalVector[nodeLID] = nodeGID;
                   });

    newGraph->numGlobalNodes = globalNodes.reduce();
    newGraph->numGlobalEdges = globalEdges.reduce();
    newGraph->numEdges       = globalEdges.read_local();

    newGraph->numOwned          = masterNodes;
    uint64_t numMirrors         = mirrorNodes;
    newGraph->numNodes          = newGraph->numOwned + numMirrors;
    newGraph->beginMaster       = 0;
    newGraph->numNodesWithEdges = nodesWithEdges.reduce();

    galois::gInfo("[", base_DistGraph::id,
                  "] Projected Global Nodes: ", newGraph->numGlobalNodes);
    galois::gInfo("[", base_DistGraph::id,
                  "] Projected Global Edges: ", newGraph->numGlobalEdges);
    galois::gInfo("[", base_DistGraph::id,
                  "] Projected Local Nodes: ", newGraph->numNodes);
    galois::gInfo("[", base_DistGraph::id,
                  "] Projected Local Edges: ", newGraph->numEdges);
    galois::gInfo("[", base_DistGraph::id,
                  "] Projected Master Nodes: ", newGraph->numOwned);
    galois::gInfo("[", base_DistGraph::id,
                  "] Projected Mirror Nodes: ", numMirrors);
    galois::gInfo("[", base_DistGraph::id,
                  "] Projected Edge Nodes: ", newGraph->numNodesWithEdges);

    newTopology.resize(newGraph->numNodes);
    newEdgeData.resize(newGraph->numNodes);
    newGraph->localToGlobalVector.resize(newGraph->numNodes);
    I_WM(newGraph->numNodes);
    newGraph->recalculateG2LMap();

    for (uint32_t i = newGraph->numOwned; i < newGraph->numNodes; i++) {
      I_RR();
      uint64_t globalID = newGraph->getGID(i);
      // deliberately use the old graph partitioner to get the owner of the GID
      I_RR();
      I_WR();
      newGraph->mirrorNodes[graphPartitioner->retrieveMaster(globalID)]
          .emplace_back(globalID);
    }

    galois::gInfo("[", base_DistGraph::id, "] Start building projected graph.");
    newGraph->graph.allocateFrom(newGraph->numNodes, newGraph->numEdges);

    galois::do_all(
        galois::iterate(uint64_t(0), uint64_t(newGraph->numNodes)),
        [&](auto& node) {
          I_RS();
          I_RR();
          NodeLID oldGraphLID =
              base_DistGraph::getLID(newGraph->localToGlobalVector[node]);
          I_WR(node >= newGraph->numOwned);
          newGraph->graph.getData(node) = projection.ProjectNode(
              *this, base_DistGraph::getData(oldGraphLID), oldGraphLID);

          I_RR();
          uint64_t numEdges = newTopology[node].size();
          if (node >= newGraph->numOwned) {
            return;
          }
          std::vector<NodeLID> localDsts;
          localDsts.reserve(numEdges);
          for (NodeGID gid : newTopology[node]) {
            I_RR();
            I_WR();
            localDsts.emplace_back(newGraph->getLID(gid));
          }
          I_RR();
          I_WM(numEdges);
          newGraph->graph.addEdgesUnSort(true, node, localDsts.data(),
                                         newEdgeData[node].data(), numEdges,
                                         false);

          newTopology[node].clear();
          newEdgeData[node].clear();
        });

    galois::gInfo("[", base_DistGraph::id,
                  "] Finished building projected graph.");

    newGraph->graphPartitioner = std::move(graphPartitioner);
    newGraph->determineThreadRanges();
    newGraph->determineThreadRangesMaster();
    newGraph->determineThreadRangesWithEdges();
    newGraph->initializeSpecificRanges();

    return newGraph;
  }

private:
  WMDGraph(unsigned host, unsigned _numHosts)
      : base_DistGraph(host, _numHosts) {}

  std::vector<std::vector<uint64_t>> edgeInspectionRound1(
      galois::graphs::WMDBufferedGraph<NodeTy, EdgeTy>& bufGraph) {
    std::vector<std::vector<uint64_t>> incomingMirrors(
        base_DistGraph::numHosts);
    uint32_t myID = base_DistGraph::id;
    base_DistGraph::localToGlobalVector.resize(base_DistGraph::numOwned);
    uint32_t activeThreads = galois::getActiveThreads();
    std::vector<std::vector<std::set<uint64_t>>> incomingMirrorsPerThread(
        base_DistGraph::numHosts);
    for (uint32_t h = 0; h < base_DistGraph::numHosts; h++) {
      incomingMirrorsPerThread[h].resize(activeThreads);
    }

    size_t start = bufGraph.globalNodeOffset[base_DistGraph::id];
    size_t end;
    if (base_DistGraph::id != base_DistGraph::numHosts - 1)
      end = bufGraph.globalNodeOffset[base_DistGraph::id + 1];
    else
      end = bufGraph.localNodeSize[base_DistGraph::numHosts - 1] +
            bufGraph.globalNodeOffset[base_DistGraph::id];

    galois::on_each([&](unsigned tid, unsigned nthreads) {
      uint64_t beginNode;
      uint64_t endNode;
      std::tie(beginNode, endNode) =
          galois::block_range(start, end, tid, nthreads);

      for (uint64_t i = beginNode; i < endNode; ++i) {
        auto ii = bufGraph.edgeBegin(i);
        auto ee = bufGraph.edgeEnd(i);
        for (; ii < ee; ++ii) {
          uint64_t dst = bufGraph.edgeDestination(*ii);
          uint64_t master_dst =
              bufGraph.virtualToPhyMapping[dst % (bufGraph.scaleFactor *
                                                  base_DistGraph::numHosts)];
          if (master_dst != myID) {
            assert(master_dst < base_DistGraph::numHosts);
            incomingMirrorsPerThread[master_dst][tid].insert(dst);
          }
        }
        base_DistGraph::localToGlobalVector[i - bufGraph.globalNodeOffset
                                                    [base_DistGraph::id]] =
            bufGraph
                .LIDtoGID[i - bufGraph.globalNodeOffset[base_DistGraph::id]];
      }
    });

    std::vector<std::set<uint64_t>> dest(base_DistGraph::numHosts);
    for (uint32_t h = 0; h < base_DistGraph::numHosts; h++) {
      for (uint32_t t = 0; t < activeThreads; t++) {
        std::set<uint64_t> tempUnion;
        std::set_union(dest[h].begin(), dest[h].end(),
                       incomingMirrorsPerThread[h][t].begin(),
                       incomingMirrorsPerThread[h][t].end(),
                       std::inserter(tempUnion, tempUnion.begin()));
        dest[h] = tempUnion;
      }
      std::copy(dest[h].begin(), dest[h].end(),
                std::back_inserter(incomingMirrors[h]));
    }
    incomingMirrorsPerThread.clear();
    uint64_t offset = base_DistGraph::localToGlobalVector.size();
    uint64_t count  = 0;
    for (uint64_t i = 0; i < incomingMirrors.size(); i++) {
      count += incomingMirrors[i].size();
    }
    uint32_t additionalMirrorCount = count;
    base_DistGraph::localToGlobalVector.resize(
        base_DistGraph::localToGlobalVector.size() + additionalMirrorCount);

    for (uint64_t i = 0; i < incomingMirrors.size(); i++) {
      for (uint64_t j = 0; j < incomingMirrors[i].size(); j++) {
        base_DistGraph::localToGlobalVector[offset] = incomingMirrors[i][j];
        offset++;
      }
    }

    base_DistGraph::numNodes = base_DistGraph::numOwned + additionalMirrorCount;
    // Creating Global to Local ID map
    base_DistGraph::globalToLocalMap.reserve(base_DistGraph::numNodes);
    for (unsigned i = 0; i < base_DistGraph::numNodes; i++) {
      base_DistGraph::globalToLocalMap[base_DistGraph::localToGlobalVector[i]] =
          i;
    }
    base_DistGraph::numNodesWithEdges = base_DistGraph::numNodes;
    return incomingMirrors;
  }

  /**
   * Communicate to other hosts which proxies exist on this host.
   *
   * @param presentProxies Bitset marking which proxies are present on this host
   * @param proxiesOnOtherHosts Vector to deserialize received bitsets into
   */
  void
  communicateProxyInfo(std::vector<std::vector<uint64_t>> presentProxies,
                       std::vector<std::vector<uint64_t>> proxiesOnOtherHosts) {
    auto& net = galois::runtime::getSystemNetworkInterface();
    // Send proxies on this host to other hosts
    for (unsigned h = 0; h < base_DistGraph::numHosts; ++h) {
      if (h != base_DistGraph::id) {
        galois::runtime::SendBuffer bitsetBuffer;
        galois::runtime::gSerialize(bitsetBuffer, presentProxies[h]);
        I_LC(h, bitsetBuffer.size());
        net.sendTagged(h, galois::runtime::evilPhase, bitsetBuffer);
      }
    }

    // receive loop
    for (unsigned h = 0; h < net.Num - 1; h++) {
      decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;
      do {
        p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
      } while (!p);
      uint32_t sendingHost = p->first;
      // deserialize proxiesOnOtherHosts
      I_LC(sendingHost, p->second.size());
      galois::runtime::gDeserialize(p->second,
                                    proxiesOnOtherHosts[sendingHost]);
    }

    base_DistGraph::increment_evilPhase();
  }

  ////////////////////////////////////////////////////////////////////////////////
public:
  galois::GAccumulator<uint64_t> lgMapAccesses;
  /**
   * Construct a map from local edge GIDs to LID
   */
  void constructLocalEdgeGIDMap() {
    lgMapAccesses.reset();
    galois::StatTimer mapConstructTimer("GID2LIDMapConstructTimer", GRNAME);
    mapConstructTimer.start();

    localEdgeGIDToLID.reserve(base_DistGraph::sizeEdges());

    uint64_t count = 0;
    for (unsigned src = 0; src < base_DistGraph::size(); src++) {
      for (auto edge = base_DistGraph::edge_begin(src);
           edge != base_DistGraph::edge_end(src); edge++) {
        assert((*edge) == count);
        unsigned dst      = base_DistGraph::getEdgeDst(edge);
        uint64_t localGID = getEdgeGIDFromSD(src, dst);
        // insert into map
        localEdgeGIDToLID.insert(std::make_pair(localGID, count));
        count++;
      }
    }

    GALOIS_ASSERT(localEdgeGIDToLID.size() == base_DistGraph::sizeEdges());
    GALOIS_ASSERT(count == base_DistGraph::sizeEdges());

    mapConstructTimer.stop();
  }

  void reportAccessBefore() {
    galois::runtime::reportStat_Single(GRNAME, std::string("MapAccessesBefore"),
                                       lgMapAccesses.reduce());
  }

  void reportAccess() {
    galois::runtime::reportStat_Single(GRNAME, std::string("MapAccesses"),
                                       lgMapAccesses.reduce());
  }

  /**
   * checks map constructed above to see which local id corresponds
   * to a node/edge (if it exists)
   *
   * assumes map is generated
   */
  std::pair<uint64_t, bool> getLIDFromMap(unsigned src, unsigned dst) {
    lgMapAccesses += 1;
    // try to find gid in map
    uint64_t localGID = getEdgeGIDFromSD(src, dst);
    auto findResult   = localEdgeGIDToLID.find(localGID);

    // return if found, else return a false
    if (findResult != localEdgeGIDToLID.end()) {
      return std::make_pair(findResult->second, true);
    } else {
      // not found
      return std::make_pair((uint64_t)-1, false);
    }
  }

  uint64_t getEdgeLID(uint64_t gid) {
    uint64_t sourceNodeGID = edgeGIDToSource(gid);
    uint64_t sourceNodeLID = base_DistGraph::getLID(sourceNodeGID);
    uint64_t destNodeLID   = base_DistGraph::getLID(edgeGIDToDest(gid));

    for (auto edge : base_DistGraph::edges(sourceNodeLID)) {
      uint64_t edgeDst = base_DistGraph::getEdgeDst(edge);
      if (edgeDst == destNodeLID) {
        return *edge;
      }
    }
    GALOIS_DIE("unreachable");
    return (uint64_t)-1;
  }

  uint32_t findSourceFromEdge(uint64_t lid) {
    // TODO binary search
    // uint32_t left = 0;
    // uint32_t right = base_DistGraph::numNodes;
    // uint32_t mid = (left + right) / 2;

    for (uint32_t mid = 0; mid < base_DistGraph::numNodes; mid++) {
      uint64_t edge_left  = *(base_DistGraph::edge_begin(mid));
      uint64_t edge_right = *(base_DistGraph::edge_begin(mid + 1));

      if (edge_left <= lid && lid < edge_right) {
        return mid;
      }
    }

    GALOIS_DIE("unreachable");
    return (uint32_t)-1;
  }

  uint64_t getEdgeGID(uint64_t lid) {
    uint64_t src = base_DistGraph::getGID(findSourceFromEdge(lid));
    uint64_t dst = base_DistGraph::getGID(base_DistGraph::getEdgeDst(lid));
    return getEdgeGIDFromSD(src, dst);
  }

private:
  // https://www.quora.com/
  // Is-there-a-mathematical-function-that-converts-two-numbers-into-one-so-
  // that-the-two-numbers-can-always-be-extracted-again
  // GLOBAL IDS ONLY
  uint64_t getEdgeGIDFromSD(uint64_t source, uint64_t dest) {
    return source + (dest % base_DistGraph::numGlobalNodes) *
                        base_DistGraph::numGlobalNodes;
  }

  uint64_t edgeGIDToSource(uint64_t gid) {
    return gid % base_DistGraph::numGlobalNodes;
  }

  uint64_t edgeGIDToDest(uint64_t gid) {
    // assuming this floors
    return gid / base_DistGraph::numGlobalNodes;
  }

  /**
   * Fill up mirror arrays.
   * TODO make parallel?
   */
  void fillMirrors() {
    base_DistGraph::mirrorNodes.reserve(base_DistGraph::numNodes -
                                        base_DistGraph::numOwned);
    for (uint32_t i = base_DistGraph::numOwned; i < base_DistGraph::numNodes;
         i++) {
      I_RR();
      uint64_t globalID = base_DistGraph::localToGlobalVector[i];
      I_RR();
      I_WR();
      assert(graphPartitioner->retrieveMaster(globalID) <
             base_DistGraph::numHosts);
      base_DistGraph::mirrorNodes[graphPartitioner->retrieveMaster(globalID)]
          .push_back(globalID);
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
};

// make GRNAME visible to public
template <typename NodeTy, typename EdgeTy, typename Partitioner>
constexpr const char* const
    galois::graphs::WMDGraph<NodeTy, EdgeTy, Partitioner>::GRNAME;

} // end namespace graphs
} // end namespace galois
#endif
