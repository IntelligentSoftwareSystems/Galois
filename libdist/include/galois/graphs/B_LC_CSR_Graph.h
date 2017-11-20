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

    incomingEdgeConstructTimer.stop();
  }

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
};

} // end graphs namespace
} // end galois namespace
#endif
