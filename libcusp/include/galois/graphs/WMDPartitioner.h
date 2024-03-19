/**
 * @file WMDPartitioner.h
 *
 * Graph partitioning that duplicates edges for WMD dataset. Currently only
 * supports an outgoing edge cut.
 *
 */

#include "galois/Galois.h"
#include "galois/graphs/DistributedGraph.h"
#include "galois/DReducible.h"

#include "WMDGraph.h"

#include <atomic>
#include <unistd.h>
#include <ios>
#include <iostream>
#include <fstream>
#include <string>
#include <stack>

namespace galois {
namespace graphs {
/**
 * @tparam NodeTy type of node data for the graph
 * @tparam EdgeTy type of edge data for the graph
 *
 * @todo fully document and clean up code
 * @warning not meant for public use + not fully documented yet
 */
template <typename NodeTy, typename EdgeTy, typename Partitioner = OECPolicy>
class WMDGraph : public DistGraph<NodeTy, EdgeTy> {

  //! size used to buffer edge sends during partitioning
  constexpr static unsigned edgePartitionSendBufSize = 8388608;
  constexpr static const char* const GRNAME          = "dGraph_WMD";
  std::unique_ptr<Partitioner> graphPartitioner;

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
      std::vector<std::unique_ptr<galois::graphs::FileParser<galois::Edge>>>&
          parsers,
      unsigned host, unsigned _numHosts,
      galois::graphs::MASTERS_DISTRIBUTION md = BALANCED_EDGES_OF_MASTERS)
      : base_DistGraph(host, _numHosts) {
    galois::gInfo("[", base_DistGraph::id, "] Start DistGraph construction.");
    galois::runtime::reportParam(GRNAME, "WMDGraph", "0");
    galois::StatTimer Tgraph_construct("GraphPartitioningTime", GRNAME);
    Tgraph_construct.start();

    ////////////////////////////////////////////////////////////////////////////
    galois::gInfo("[", base_DistGraph::id, "] Start reading graph.");
    galois::StatTimer graphReadTimer("GraphReading", GRNAME);
    graphReadTimer.start();
    galois::gInfo("[", base_DistGraph::id, "] EdgeListOfflineGraph End!");
    galois::graphs::EdgeListOfflineGraph<galois::Vertex, galois::Edge> g(parsers, md, 4);
    galois::gInfo("[", base_DistGraph::id, "] EdgeListOfflineGraph End!");
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
    galois::graphs::EdgeListBufferedGraph<galois::Vertex, galois::Edge> bufGraph;
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
    uint64_t nodeBegin = bufGraph.globalNodeOffset[base_DistGraph::id];
    uint64_t nodeEnd   = bufGraph.globalNodeOffset[base_DistGraph::id] +
                       bufGraph.localNodeSize[base_DistGraph::id];
    base_DistGraph::numOwned = bufGraph.localNodeSize[base_DistGraph::id];
    graphPartitioner->saveGIDToHost(bufGraph.virtualToPhyMapping);
    edgeInspectionRound(bufGraph);
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
    base_DistGraph::graph.constructNodes();

    // construct edges
    galois::gInfo("[", base_DistGraph::id, "] add edges into graph.");
    galois::do_all(
        galois::iterate(nodeBegin, nodeEnd),
        [&](uint64_t globalID) {
          auto edgeDst = bufGraph.edgeLocalDst(globalID);
          std::stack<uint64_t> dstData;
          for (auto dst : edgeDst) {
            dstData.push(base_DistGraph::globalToLocalMap[dst]);
          }
          base_DistGraph::graph.addEdges(
              (globalID - bufGraph.globalNodeOffset[base_DistGraph::id]), dstData);
        },
        galois::steal());
    // move node data (include mirror nodes) from other hosts to graph in this
    // host
    galois::gDebug("[", base_DistGraph::id, "] add nodes data into graph.");
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
    //if (setupGluon) {
      galois::CondStatTimer<MORE_DIST_STATS> TfillMirrors("FillMirrors",
                                                          GRNAME);
      TfillMirrors.start();
      fillMirrors();
      TfillMirrors.stop();
    //}
    ////////////////////////////////////////////////////////////////////////////
    ndegrees.clear();
    ndegrees.shrink_to_fit();
    // SORT EDGES
    //if (doSort) {
    //  base_DistGraph::sortEdgesByDestination();
    //}
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

private:
  WMDGraph(unsigned host, unsigned _numHosts)
      : base_DistGraph(host, _numHosts) {}

  void edgeInspectionRound(
      galois::graphs::EdgeListBufferedGraph<galois::Vertex, galois::Edge>& bufGraph) {
    std::vector<std::vector<uint64_t>> incomingMirrors(base_DistGraph::numHosts);
    uint32_t myID         = base_DistGraph::id;
    base_DistGraph::localToGlobalVector.resize(base_DistGraph::numOwned);
    uint32_t activeThreads = galois::getActiveThreads();
    std::vector<std::vector<std::set<uint64_t>>> incomingMirrorsPerThread(base_DistGraph::numHosts);
    for(uint32_t h=0; h<base_DistGraph::numHosts; h++) {
        incomingMirrorsPerThread[h].resize(activeThreads);
    }
    size_t start = bufGraph.globalNodeOffset[base_DistGraph::id];
    size_t end;
    if(base_DistGraph::id != base_DistGraph::numHosts - 1)
        end = bufGraph.globalNodeOffset[base_DistGraph::id + 1];
    else
	      end = bufGraph.localNodeSize[base_DistGraph::numHosts - 1] + bufGraph.globalNodeOffset[base_DistGraph::id];
    galois::on_each([&](unsigned tid, unsigned nthreads) {
      uint64_t beginNode;
      uint64_t endNode;
      std::tie(beginNode, endNode) = galois::block_range(start, end, tid, nthreads);
      for(uint64_t i = beginNode; i < endNode; ++i) {
        auto ii            = bufGraph.edgeBegin(i);
        auto ee            = bufGraph.edgeEnd(i);
        for (; ii < ee; ++ii) {
          uint64_t dst = bufGraph.edgeDestination(*ii);
          uint64_t master_dst = bufGraph.virtualToPhyMapping[dst%(bufGraph.scaleFactor*base_DistGraph::numHosts)];
            if (master_dst != myID) {
                  assert(master_dst < base_DistGraph::numHosts);
                  incomingMirrorsPerThread[master_dst][tid].insert(dst);
            }
        }
        base_DistGraph::localToGlobalVector[i - bufGraph.globalNodeOffset[base_DistGraph::id]] = bufGraph.LIDtoGID[i - bufGraph.globalNodeOffset[base_DistGraph::id]];
      }
      });
    std::vector<std::set<uint64_t>> dest(base_DistGraph::numHosts);
    for(uint32_t h=0; h<base_DistGraph::numHosts; h++) {
      for(uint32_t t=0; t<activeThreads; t++) {
        std::set<uint64_t> tempUnion;
        std::set_union(dest[h].begin(), dest[h].end(),
                   incomingMirrorsPerThread[h][t].begin(), incomingMirrorsPerThread[h][t].end(),
                   std::inserter(tempUnion, tempUnion.begin()));
        dest[h] = tempUnion;
      }
        std::copy(dest[h].begin(), dest[h].end(), std::back_inserter(incomingMirrors[h]));
    }
    incomingMirrorsPerThread.clear();
    uint64_t offset = base_DistGraph::localToGlobalVector.size();
    uint64_t count = 0;
    for(uint64_t i=0; i<incomingMirrors.size(); i++) {
        count += incomingMirrors[i].size();
    }
    uint32_t additionalMirrorCount = count;
    base_DistGraph::localToGlobalVector.resize(
        base_DistGraph::localToGlobalVector.size() + additionalMirrorCount);
    for (uint64_t i=0;i<incomingMirrors.size();i++) {
            for(uint64_t j=0; j <incomingMirrors[i].size(); j++) {
               base_DistGraph::localToGlobalVector[offset] = incomingMirrors[i][j];
               offset++;
            }
        }
    base_DistGraph::numNodes = base_DistGraph::numOwned + additionalMirrorCount;
    //Creating Global to Local ID map
    base_DistGraph::globalToLocalMap.reserve(base_DistGraph::numNodes);
    for (unsigned i = 0; i < base_DistGraph::numNodes; i++) {
      base_DistGraph::globalToLocalMap[base_DistGraph::localToGlobalVector[i]] =
          i;
    }
    base_DistGraph::numNodesWithEdges = base_DistGraph::numNodes;
  }

  ////////////////////////////////////////////////////////////////////////////////
public:
  galois::GAccumulator<uint64_t> lgMapAccesses;
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

private:
  /**
   * Fill up mirror arrays.
   * TODO make parallel?
   */
  void fillMirrors() {
    base_DistGraph::mirrorNodes.reserve(base_DistGraph::numNodes -
                                        base_DistGraph::numOwned);
    for (uint32_t i = base_DistGraph::numOwned; i < base_DistGraph::numNodes;
         i++) {
      uint64_t globalID = base_DistGraph::localToGlobalVector[i];
      assert(graphPartitioner->retrieveMaster(globalID) < base_DistGraph::numHosts);

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
