#ifndef __B_LC_CSR_GRAPH__
#define __B_LC_CSR_GRAPH__

#include "galois/graphs/LC_CSR_Graph.h"
#include "galois/Timer.h"

namespace galois {
namespace graphs {

template<typename NodeTy, typename EdgeTy,
         bool HasNoLockable=false,
         bool UseNumaAlloc=false,
         bool HasOutOfLineLockable=false,
         typename FileEdgeTy=EdgeTy>
class B_LC_CSR_Graph :
     public LC_CSR_Graph<NodeTy, EdgeTy, HasNoLockable, UseNumaAlloc, 
                  HasOutOfLineLockable, FileEdgeTy> {
  // typedef to make it easier to read
  using BaseGraph = LC_CSR_Graph<NodeTy, EdgeTy, HasNoLockable, UseNumaAlloc, 
                                 HasOutOfLineLockable, FileEdgeTy>;
  using ThisGraph = B_LC_CSR_Graph<NodeTy, EdgeTy, HasNoLockable, UseNumaAlloc, 
                                   HasOutOfLineLockable, FileEdgeTy>;

protected:
  // retypedefs of base class
  using EdgeData = LargeArray<EdgeTy>;
  using EdgeDst = LargeArray<uint32_t>;
  using EdgeIndData = LargeArray<uint64_t>;

  EdgeIndData inEdgeIndData;
  EdgeDst inEdgeDst;
  // TODO don't use a separate array for in edge data; link back to original
  // edge data array
  EdgeData inEdgeData; 

  /**
   * Wrapper of this class to make a prefix sum accessible through []
   */
  struct PrefixSumWrapper {
    ThisGraph* pointerToB_LC;

    PrefixSumWrapper() {pointerToB_LC = nullptr;}

    PrefixSumWrapper(ThisGraph* _thisGraph) 
        : pointerToB_LC(_thisGraph) {}

    /**
     * Set the pointer to self.
     */
    void setSelfPointer(ThisGraph* thisGraph) {
      pointerToB_LC = thisGraph;
    }

    uint64_t operator[](uint64_t n) {
      return *(pointerToB_LC->in_edge_end(n));
    }
  };
  // used to wrap this class to access prefix sum
  PrefixSumWrapper sumWrapper;

  // thread range vectors
  std::vector<uint32_t> allNodesInThreadRange;
  std::vector<uint32_t> masterNodesInThreadRange;

  using NodeRangeType = 
      galois::runtime::SpecificRange<boost::counting_iterator<size_t>>;
  std::vector<NodeRangeType> specificInRanges;

public:
  using GraphNode = uint32_t;
  using edge_iterator = 
    boost::counting_iterator<typename EdgeIndData::value_type>;
  using edge_data_reference = typename EdgeData::reference;

  B_LC_CSR_Graph() = default;
  B_LC_CSR_Graph(B_LC_CSR_Graph&& rhs) = default;
  B_LC_CSR_Graph& operator=(B_LC_CSR_Graph&&) = default;

  // sparse = push
  // dense = pull

  /*****************************************************************************
   * Construction functions
   ****************************************************************************/
  /**
   * Call only after the LC_CSR_Graph is fully constructed.
   * Creates the in edge data by reading from the out edge data.
   */
  void constructIncomingEdges() {
    galois::StatTimer incomingEdgeConstructTimer("IncomingEdgeConstruct");
    incomingEdgeConstructTimer.start();

    EdgeIndData dataBuffer;
    dataBuffer.allocateInterleaved(BaseGraph::numNodes);

    // initialize the temp array
    galois::do_all(
      galois::iterate(0ul, BaseGraph::numNodes), 
      [&](uint32_t n) {
        dataBuffer[n] = 0;
      }
    );
      
    // get destination of edge, copy to array, and
    galois::do_all(
      galois::iterate(0ul, BaseGraph::numEdges), 
      [&](uint64_t e) {
        auto dst = BaseGraph::edgeDst[e];
        // counting outgoing edges in the tranpose graph by
        // counting incoming edges in the original graph
        __sync_add_and_fetch(&(dataBuffer[dst]), 1);
      }
    );

    // prefix sum calculation of the edge index array
    for (uint32_t n = 1; n < BaseGraph::numNodes; ++n) {
      dataBuffer[n] += dataBuffer[n - 1];
    }

    // TODO thread ranges for incoming
    // recalculate thread ranges for nodes and edges
    //determineThreadRangesByNode(dataBuffer);

    // copy over the new tranposed edge index data
    inEdgeIndData.allocateInterleaved(BaseGraph::numNodes);
    galois::do_all(
      galois::iterate(0ul, BaseGraph::numNodes), 
      [&](uint32_t n) {
        inEdgeIndData[n] = dataBuffer[n];
      }
    );

    // after this block dataBuffer[i] will now hold number of edges that all 
    // nodes before the ith node have
    if (BaseGraph::numNodes >= 1) {
      dataBuffer[0] = 0;
      galois::do_all(
        galois::iterate(1ul, BaseGraph::numNodes), 
        [&](uint32_t n) {
          dataBuffer[n] = inEdgeIndData[n - 1];
        }
      );
    }

    // get edge dests and data
    inEdgeDst.allocateInterleaved(BaseGraph::numEdges);
    inEdgeData.allocateInterleaved(BaseGraph::numEdges);

    galois::do_all(
      galois::iterate(0ul, BaseGraph::numNodes), 
      [&](uint32_t src) {
        // e = start index into edge array for a particular node
        uint64_t e = (src == 0) ? 0 : BaseGraph::edgeIndData[src - 1];

        // get all outgoing edges of a particular node in the non-transpose and
        // convert to incoming
        while (e < BaseGraph::edgeIndData[src]) {
          // destination nodde
          auto dst = BaseGraph::edgeDst[e];
          // location to save edge
          auto e_new = __sync_fetch_and_add(&(dataBuffer[dst]), 1);
          // save src as destination
          inEdgeDst[e_new] = src;
          // copy edge data to "new" array
          BaseGraph::edgeDataCopy(inEdgeData, BaseGraph::edgeData, e_new, e);
          e++;
        }
      }
    );

    // set this for use when finding thread ranges later
    sumWrapper.setSelfPointer(this);

    incomingEdgeConstructTimer.stop();
  }

  /*****************************************************************************
   * Thread Ranges Finding Functions
   ****************************************************************************/

  /**
   * Returns 2 ranges (one for nodes, one for edges) for a particular division.
   * The ranges specify the nodes/edges that a division is responsible for. The
   * function attempts to split them evenly among threads given some kind of
   * weighting. USES THE IN-EDGES. 
   *
   * @param nodeWeight weight to give to a node in division
   * @param edgeWeight weight to give to an edge in division
   * @param id Division number you want the ranges for
   * @param total Total number of divisions
   * @param nodesInRange Number of nodes in the range you want to divide
   * up (specified by node offset)
   * @param edgeInRange Number of edges in range you want to divide up
   * (specified by node offset)
   * @param nodeOffset Offset to the first node in the range you want
   * to divide up
   * @param edgeOffset Offset to the first edge in the range you want
   * to divide up
   */
  auto divideByNodeInEdges(size_t nodeWeight, size_t edgeWeight, size_t id,
                    size_t total, uint32_t nodesInRange, uint64_t edgesInRange,
                    uint32_t nodeOffset, uint64_t edgeOffset) 
                    -> typename BaseGraph::GraphRange {
    std::vector<unsigned int> dummyScaleFactor;
    return
      galois::graphs::divideNodesBinarySearch<PrefixSumWrapper, uint32_t>(
        nodesInRange, edgesInRange, nodeWeight, edgeWeight, id, total, 
        sumWrapper, dummyScaleFactor, edgeOffset, nodeOffset);
  }


  /**
   * Helper function used by determineThreadRanges that consists of the main
   * loop over all threads and calls to divide by node to determine the
   * division of nodes to threads. USES IN EDGES.
   *
   * Saves the ranges to an argument vector provided by the caller.
   *
   * @param beginNode Beginning of range
   * @param endNode End of range, non-inclusive
   * @param returnRanges Vector to store thread offsets for ranges in
   * @param nodeAlpha The higher the number, the more weight nodes have in
   * determining division of nodes (edges have weight 1).
   */
  void determineThreadRangesThreadLoopInEdges(uint32_t beginNode,
                                          uint32_t endNode,
                                          std::vector<uint32_t>& returnRanges,
                                          uint32_t nodeAlpha) {
    uint32_t numNodesInRange = endNode - beginNode;
    uint64_t numEdgesInRange = in_raw_end(endNode - 1) -  
                               in_raw_begin(beginNode);
    uint32_t numThreads = galois::runtime::activeThreads;
    uint64_t edgeOffset = *in_raw_begin(beginNode);

    returnRanges[0] = beginNode;
    for (uint32_t i = 0; i < numThreads; i++) {
      // determine division for thread i
      auto nodeEdgeSplits = divideByNodeInEdges(nodeAlpha, 1, i, numThreads, 
                                                numNodesInRange,
                                                numEdgesInRange, beginNode, 
                                                edgeOffset);
      auto nodeSplits = nodeEdgeSplits.first;

      // i.e. if there are actually assigned nodes
      if (nodeSplits.first != nodeSplits.second) {
        if (i != 0) {
          assert(returnRanges[i] == *(nodeSplits.first));
        } else { // i == 0
          assert(returnRanges[i] == beginNode);
        }
        returnRanges[i + 1] = *(nodeSplits.second) + beginNode;
      } else {
        // thread assinged no nodes
        returnRanges[i + 1] = returnRanges[i];
      }

      galois::gDebug("SaveVector: Thread ", i, " gets nodes ", returnRanges[i],
                     " to ", returnRanges[i + 1], ", num in-edges is ",
                     in_raw_end(returnRanges[i + 1] - 1) - 
                     in_raw_begin(returnRanges[i]));
    }
  }


  /**
   * Determines thread ranges for a given range of nodes and returns it as
   * an offset vector in the passed in vector. (thread ranges = assigned
   * nodes that a thread should work on). USES IN EDGES.
   *
   * Checks for corner cases, then calls the main loop function.
   *
   * ONLY CALL AFTER GRAPH IS CONSTRUCTED as it uses functions that assume
   * the graph is already constructed.
   *
   * @param beginNode Beginning of range
   * @param endNode End of range, non-inclusive
   * @param returnRanges Vector to store thread offsets for ranges in
   * @param nodeAlpha The higher the number, the more weight nodes have in
   * determining division of nodes (edges have weight 1).
   */
  void determineThreadRangesInEdges(uint32_t beginNode, uint32_t endNode,
                             std::vector<uint32_t>& returnRanges,
                             uint32_t nodeAlpha=0) {
    uint32_t numThreads = galois::runtime::activeThreads;
    uint32_t total_nodes = endNode - beginNode;

    returnRanges.resize(numThreads + 1);

    // check corner cases
    // no nodes = assign nothing to all threads
    if (beginNode == endNode) {
      returnRanges[0] = beginNode;
      for (uint32_t i = 0; i < numThreads; i++) {
        returnRanges[i + 1] = beginNode;
      }
      return;
    }

    // single thread case; 1 thread gets all
    if (numThreads == 1) {
      returnRanges[0] = beginNode;
      returnRanges[1] = endNode;
      return;
    // more threads than nodes
    } else if (numThreads > total_nodes) {
      uint32_t current_node = beginNode;
      returnRanges[0] = current_node;
      // 1 node for threads until out of threads
      for (uint32_t i = 0; i < total_nodes; i++) {
        returnRanges[i + 1] = ++current_node;
      }
      // deal with remainder threads; they get nothing
      for (uint32_t i = total_nodes; i < numThreads; i++) {
        returnRanges[i + 1] = total_nodes;
      }
      return;
    }

    // no corner cases: onto main loop over nodes that determines
    // node ranges
    determineThreadRangesThreadLoopInEdges(beginNode, endNode, returnRanges, 
                                           nodeAlpha);
    #ifndef NDEBUG
    // sanity checks
    assert(returnRanges[0] == beginNode &&
           "return ranges begin not the begin node");
    assert(returnRanges[numThreads] == endNode &&
           "return ranges end not end node");

    for (uint32_t i = 1; i < numThreads; i++) {
      assert(returnRanges[i] >= beginNode && returnRanges[i] <= endNode);
      assert(returnRanges[i] >= returnRanges[i-1]);
    }
    #endif
  }

  /*****************************************************************************
   * Thread Ranges Saving Functions
   ****************************************************************************/

  /**
   * Find thread ranges for all nodes considering in-edges. Save to vector
   * in this instance of the class.
   */
  void findAllNodeThreadRangeIn() {
    assert(allNodesInThreadRange.size() == 0);
    determineThreadRangesInEdges(0, BaseGraph::numNodes, allNodesInThreadRange);
  }

  /**
   * Find thread ranges for master nodes considering in-edges. Save to vector
   * in this instance of the class.
   *
   * @param beginMaster first master node
   * @param numOwned number of nodes owned by this graph
   * @param numNodesWithEdges number of nodes with edges in this graph
   */
  void findMasterNodesThreadRangeIn(uint32_t beginMaster, uint32_t numOwned,
                                    uint32_t numNodesWithEdges) {
    assert(masterNodesInThreadRange.size() == 0);

    // determine if work needs to be done
    if (beginMaster == 0 && (beginMaster + numOwned) == BaseGraph::numNodes) {
      masterNodesInThreadRange = allNodesInThreadRange;
    } else {
      determineThreadRangesInEdges(beginMaster, beginMaster + numOwned, 
                                   masterNodesInThreadRange);
    }
  }

  /**
   * Construct 2 specific range objects using the 3 thread ranges vectors
   * that should have been calculated before this function is called.
   *
   * @param beginMaster first master node
   * @param numOwned number of nodes owned by this graph
   * @param numNodesWithEdges number of nodes with edges in this graph
   */
  void finalizeThreadRangesIn(uint32_t beginMaster, uint32_t numOwned, 
                              uint32_t numNodesWithEdges) {
    assert(specificInRanges.size() == 0);
    
    // 0 is all nodes
    specificInRanges.push_back(
      galois::runtime::makeSpecificRange(
        boost::counting_iterator<size_t>(0),
        boost::counting_iterator<size_t>(BaseGraph::size()),
        allNodesInThreadRange.data()
      )
    );

    // 1 is master nodes
    specificInRanges.push_back(
      galois::runtime::makeSpecificRange(
        boost::counting_iterator<size_t>(beginMaster),
        boost::counting_iterator<size_t>(beginMaster + numOwned),
        masterNodesInThreadRange.data()
      )
    );

    assert(specificInRanges.size() == 2);
  }

  /*****************************************************************************
   * Access functions
   ****************************************************************************/

  /**
   * Grabs in edge beginning without lock/safety.
   *
   * @param N node to get edge beginning of
   * @returns Iterator to first in edge of node N
   */
  edge_iterator in_raw_begin(GraphNode N) const {
    return edge_iterator((N == 0) ? 0 : inEdgeIndData[N - 1]);
  }

  /**
   * Grabs in edge end without lock/safety.
   *
   * @param N node to get edge end of
   * @returns Iterator to end of in edges of node N (i.e. first edge of 
   * node N+1)
   */
  edge_iterator in_raw_end(GraphNode N) const {
    return edge_iterator(inEdgeIndData[N]);
  }

  /**
   * Wrapper to get the in edge end of a node; lock if necessary.
   *
   * @param N node to get edge beginning of
   * @param mflag how safe the acquire should be
   * @returns Iterator to first in edge of node N
   */
  edge_iterator in_edge_begin(GraphNode N, 
                              MethodFlag mflag = MethodFlag::WRITE) {
    BaseGraph::acquireNode(N, mflag);
    if (galois::runtime::shouldLock(mflag)) {
      for (edge_iterator ii = in_raw_begin(N), ee = in_raw_end(N); 
           ii != ee; 
           ++ii) {
        BaseGraph::acquireNode(inEdgeDst[*ii], mflag);
      }
    }
    return in_raw_begin(N);
  }

  /**
   * Wrapper to get the in edge end of a node; lock if necessary.
   *
   * @param N node to get in edge end of
   * @param mflag how safe the acquire should be
   * @returns Iterator to end of in edges of node N (i.e. first in edge of N+1)
   */
  edge_iterator in_edge_end(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    BaseGraph::acquireNode(N, mflag);
    return in_raw_end(N);
  }

  // TODO don't use a separate array for in edge data; link back to original
  // edge data array
  /**
   * Given an edge id for in edges, get the destination of the edge
   *
   * @param ni edge id
   * @returns destination for that in edge
   */
  GraphNode getInEdgeDst(edge_iterator ni) const {
    return inEdgeDst[*ni];
  }

  /**
   * Given an edge id for in edge, get the data associated with that edge.
   *
   * @param ni in-edge id
   * @returns data of the edge
   */
  edge_data_reference getInEdgeData(edge_iterator ni, 
      MethodFlag = MethodFlag::UNPROTECTED) const {
    return inEdgeData[*ni];
  }

  /**
   * Returns specific range based on in-edges, all nodes.
   *
   * @returns SpecificRange with all nodes, split by in-edges
   */
  const NodeRangeType& allNodesRangeIn() const {
    return specificInRanges[0];
  }

  /**
   * Returns specific range based on in-edges, master nodes.
   *
   * @returns SpecificRange with master nodes, split by in-edges
   */
  const NodeRangeType& masterNodesRangeIn() const {
    return specificInRanges[1];
  }

  /**
   * Returns specific range based on in-edges, master nodes + nodes with 
   * in-edges (i.e. it's just all nodes anyways).
   *
   * @returns SpecificRange with all nodes, split by in-edges
   */
  const NodeRangeType& allNodesWithEdgesRangeIn() const {
    return specificInRanges[0];
  }
};

} // end graphs namespace
} // end galois namespace
#endif
