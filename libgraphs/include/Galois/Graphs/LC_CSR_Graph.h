/** Local Computation graphs -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2017, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 * @author Loc Hoang <l_hoang@utexas.edu>
 */

#ifndef GALOIS_GRAPH__LC_CSR_GRAPH_H
#define GALOIS_GRAPH__LC_CSR_GRAPH_H

#include "Galois/LargeArray.h"
#include "Galois/Graphs/FileGraph.h"
#include "Galois/Graphs/Details.h"
#include "Galois/Runtime/CompilerHelperFunctions.h"
#include "Galois/Galois.h"

#include <type_traits>

namespace Galois {
namespace Graph {

/**
 * Local computation graph (i.e., graph structure does not change). The data representation
 * is the traditional compressed-sparse-row (CSR) format.
 *
 * The position of template parameters may change between Galois releases; the
 * most robust way to specify them is through the with_XXX nested templates.
 *
 * An example of use:
 *
 * \snippet test/graph.cpp Using a graph
 *
 * And in C++11:
 *
 * \snippet test/graph.cpp Using a graph cxx11
 *
 * @tparam NodeTy data on nodes
 * @tparam EdgeTy data on out edges
 */
template<typename NodeTy, typename EdgeTy,
  bool HasNoLockable=false,
  bool UseNumaAlloc=false,
  bool HasOutOfLineLockable=false,
  typename FileEdgeTy=EdgeTy>
class LC_CSR_Graph:
 private boost::noncopyable,
 private detail::LocalIteratorFeature<UseNumaAlloc>,
 private detail::OutOfLineLockableFeature<HasOutOfLineLockable && !HasNoLockable> {
  template<typename Graph> friend class LC_InOut_Graph;

public:
  template<bool _has_id>
  struct with_id { typedef LC_CSR_Graph type; };

  template<typename _node_data>
  struct with_node_data { 
    typedef LC_CSR_Graph<_node_data, EdgeTy, HasNoLockable, UseNumaAlloc,HasOutOfLineLockable,FileEdgeTy> type;
  };

  template<typename _edge_data>
  struct with_edge_data { 
    typedef LC_CSR_Graph<NodeTy,_edge_data,HasNoLockable,UseNumaAlloc,HasOutOfLineLockable,FileEdgeTy> type; 
  };

  template<typename _file_edge_data>
  struct with_file_edge_data { typedef LC_CSR_Graph<NodeTy,EdgeTy,HasNoLockable,UseNumaAlloc,HasOutOfLineLockable,_file_edge_data> type; };

  //! If true, do not use abstract locks in graph
  template<bool _has_no_lockable>
  struct with_no_lockable { typedef LC_CSR_Graph<NodeTy,EdgeTy,_has_no_lockable,UseNumaAlloc,HasOutOfLineLockable,FileEdgeTy> type; };
  template<bool _has_no_lockable>
  using _with_no_lockable = LC_CSR_Graph<NodeTy,EdgeTy,_has_no_lockable,UseNumaAlloc,HasOutOfLineLockable,FileEdgeTy>;

  //! If true, use NUMA-aware graph allocation
  template<bool _use_numa_alloc>
  struct with_numa_alloc { typedef LC_CSR_Graph<NodeTy,EdgeTy,HasNoLockable,_use_numa_alloc,HasOutOfLineLockable,FileEdgeTy> type; };
  template<bool _use_numa_alloc>
  using _with_numa_alloc = LC_CSR_Graph<NodeTy,EdgeTy,HasNoLockable,_use_numa_alloc,HasOutOfLineLockable,FileEdgeTy>;

  //! If true, store abstract locks separate from nodes
  template<bool _has_out_of_line_lockable>
  struct with_out_of_line_lockable { typedef LC_CSR_Graph<NodeTy,EdgeTy,HasNoLockable,UseNumaAlloc,_has_out_of_line_lockable,FileEdgeTy> type; };

  typedef read_default_graph_tag read_tag;

protected:
  typedef LargeArray<EdgeTy> EdgeData;
  typedef LargeArray<uint32_t> EdgeDst;
  typedef detail::NodeInfoBaseTypes<NodeTy,!HasNoLockable && !HasOutOfLineLockable> NodeInfoTypes;
  typedef detail::NodeInfoBase<NodeTy,!HasNoLockable && !HasOutOfLineLockable> NodeInfo;
  typedef LargeArray<uint64_t> EdgeIndData;
  typedef LargeArray<NodeInfo> NodeData;

public:
  typedef uint32_t GraphNode;
  typedef EdgeTy edge_data_type;
  typedef FileEdgeTy file_edge_data_type;
  typedef NodeTy node_data_type;
  typedef typename EdgeData::reference edge_data_reference;
  typedef typename NodeInfoTypes::reference node_data_reference;
  typedef boost::counting_iterator<typename EdgeIndData::value_type> edge_iterator;
  typedef boost::counting_iterator<typename EdgeDst::value_type> iterator;
  typedef iterator const_iterator;
  typedef iterator local_iterator;
  typedef iterator const_local_iterator;

protected:
  NodeData nodeData;
  EdgeIndData edgeIndData;
  EdgeDst edgeDst;
  EdgeData edgeData;

  uint64_t numNodes;
  uint64_t numEdges;

  // used to track division of nodes among threads
  std::vector<uint32_t> threadRanges;
  // used to track division of edges among threads
  std::vector<uint64_t> threadRangesEdge;

  typedef detail::EdgeSortIterator<GraphNode,typename EdgeIndData::value_type,EdgeDst,EdgeData> edge_sort_iterator;
 
  edge_iterator raw_begin(GraphNode N) const {
    return edge_iterator((N == 0) ? 0 : edgeIndData[N-1]);
  }

  edge_iterator raw_end(GraphNode N) const {
    return edge_iterator(edgeIndData[N]);
  }

  edge_sort_iterator edge_sort_begin(GraphNode N) {
    return edge_sort_iterator(*raw_begin(N), &edgeDst, &edgeData);
  }

  edge_sort_iterator edge_sort_end(GraphNode N) {
    return edge_sort_iterator(*raw_end(N), &edgeDst, &edgeData);
  }

  template<bool _A1 = HasNoLockable, bool _A2 = HasOutOfLineLockable>
  void acquireNode(GraphNode N, MethodFlag mflag, typename std::enable_if<!_A1 && !_A2>::type* = 0) {
    Galois::Runtime::acquire(&nodeData[N], mflag);
  }

  template<bool _A1 = HasOutOfLineLockable, bool _A2 = HasNoLockable>
  void acquireNode(GraphNode N, MethodFlag mflag, typename std::enable_if<_A1 && !_A2>::type* = 0) {
    this->outOfLineAcquire(getId(N), mflag);
  }

  template<bool _A1 = HasOutOfLineLockable, bool _A2 = HasNoLockable>
  void acquireNode(GraphNode N, MethodFlag mflag, typename std::enable_if<_A2>::type* = 0) { }

  template<bool _A1 = EdgeData::has_value, bool _A2 = LargeArray<FileEdgeTy>::has_value>
  void constructEdgeValue(FileGraph& graph, typename FileGraph::edge_iterator nn, 
      typename std::enable_if<!_A1 || _A2>::type* = 0) {
    typedef LargeArray<FileEdgeTy> FED;
    if (EdgeData::has_value)
      edgeData.set(*nn, graph.getEdgeData<typename FED::value_type>(nn));
  }

  template<bool _A1 = EdgeData::has_value, bool _A2 = LargeArray<FileEdgeTy>::has_value>
  void constructEdgeValue(FileGraph& graph, typename FileGraph::edge_iterator nn,
      typename std::enable_if<_A1 && !_A2>::type* = 0) {
    edgeData.set(*nn, {});
  }

  size_t getId(GraphNode N) {
    return N;
  }

  GraphNode getNode(size_t n) {
    return n;
  }

public:
  LC_CSR_Graph(LC_CSR_Graph&& rhs) = default;
  LC_CSR_Graph() = default;
  LC_CSR_Graph& operator=(LC_CSR_Graph&&) = default;

  /**
   * Initialize the thread ranges pointers to null pointer. This is EXTREMELY
   * important for correctness and should be called if the initializer below
   * isn't used.
   */
  void clearRanges() {
    threadRanges.clear();
    threadRangesEdge.clear();
  }

  template<typename EdgeNumFnTy, typename EdgeDstFnTy, typename EdgeDataFnTy>
  LC_CSR_Graph(uint32_t _numNodes, uint64_t _numEdges,
               EdgeNumFnTy edgeNum, EdgeDstFnTy _edgeDst, EdgeDataFnTy _edgeData)
   : numNodes(_numNodes), numEdges(_numEdges) {
    //std::cerr << "\n**" << numNodes << " " << numEdges << "\n\n";
    if (UseNumaAlloc) {
      nodeData.allocateLocal(numNodes);
      edgeIndData.allocateLocal(numNodes);
      edgeDst.allocateLocal(numEdges);
      edgeData.allocateLocal(numEdges);
      this->outOfLineAllocateLocal(numNodes, false);
    } else {
      nodeData.allocateInterleaved(numNodes);
      edgeIndData.allocateInterleaved(numNodes);
      edgeDst.allocateInterleaved(numEdges);
      edgeData.allocateInterleaved(numEdges);
      this->outOfLineAllocateInterleaved(numNodes);
    }
    //std::cerr << "Done Alloc\n";
    for (size_t n = 0; n < numNodes; ++n) {
      nodeData.constructAt(n);
    }
    //std::cerr << "Done Node Construct\n";
    uint64_t cur = 0;
    for (size_t n = 0; n < numNodes; ++n) {
      cur += edgeNum(n);
      edgeIndData[n] = cur;
    }
    //std::cerr << "Done Edge Reserve\n";
    cur = 0;
    for (size_t n = 0; n < numNodes; ++n) {
      //if (n % (1024*128) == 0)
      //  std::cout << n << " " << cur << "\n";
      for (uint64_t e = 0, ee = edgeNum(n); e < ee; ++e) {
        if (EdgeData::has_value)
          edgeData.set(cur, _edgeData(n, e));
        edgeDst[cur] = _edgeDst(n, e);
        ++cur;
      }
    }

    //std::cerr << "Done Construct\n";
  }

  friend void swap(LC_CSR_Graph& lhs, LC_CSR_Graph& rhs) {
    swap(lhs.nodeData, rhs.nodeData);
    swap(lhs.edgeIndData, rhs.edgeIndData);
    swap(lhs.edgeDst, rhs.edgeDst);
    swap(lhs.edgeData, rhs.edgeData);
    std::swap(lhs.numNodes, rhs.numNodes);
    std::swap(lhs.numEdges, rhs.numEdges);
  }
  
  node_data_reference getData(GraphNode N, 
                              MethodFlag mflag = MethodFlag::WRITE) {
    // Galois::Runtime::checkWrite(mflag, false);
    NodeInfo& NI = nodeData[N];
    acquireNode(N, mflag);
    return NI.getData();
  }

  edge_data_reference getEdgeData(edge_iterator ni, 
                                  MethodFlag mflag = MethodFlag::UNPROTECTED) {
    // Galois::Runtime::checkWrite(mflag, false);
    return edgeData[*ni];
  }

  GraphNode getEdgeDst(edge_iterator ni) {
    return edgeDst[*ni];
  }

  size_t size() const { return numNodes; }
  size_t sizeEdges() const { return numEdges; }

  iterator begin() const { return iterator(0); }
  iterator end() const { return iterator(numNodes); }

  const_local_iterator local_begin() const { 
    return const_local_iterator(this->localBegin(numNodes));
  }

  const_local_iterator local_end() const { 
    return const_local_iterator(this->localEnd(numNodes));
  }

  local_iterator local_begin() { 
    return local_iterator(this->localBegin(numNodes));
  }

  local_iterator local_end() { 
    return local_iterator(this->localEnd(numNodes)); 
  }

  edge_iterator edge_begin(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    acquireNode(N, mflag);
    if (Galois::Runtime::shouldLock(mflag)) {
      for (edge_iterator ii = raw_begin(N), ee = raw_end(N); ii != ee; ++ii) {
        acquireNode(edgeDst[*ii], mflag);
      }
    }
    return raw_begin(N);
  }

  edge_iterator edge_end(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    acquireNode(N, mflag);
    return raw_end(N);
  }

  edge_iterator findEdge(GraphNode N1, GraphNode N2) {
    return std::find_if(edge_begin(N1), edge_end(N1), [=] (edge_iterator e) { return getEdgeDst(e) == N2; });
  }

  edge_iterator findEdgeSortedByDst(GraphNode N1, GraphNode N2) {
    auto e = std::lower_bound(edge_begin(N1), edge_end(N1), N2, [=] (edge_iterator e, GraphNode N) { return getEdgeDst(e) < N; });
    return (getEdgeDst(e) == N2) ? e : edge_end(N1);
  }

  Runtime::iterable<NoDerefIterator<edge_iterator>> edges(GraphNode N, 
      MethodFlag mflag = MethodFlag::WRITE) {
    return detail::make_no_deref_range(edge_begin(N, mflag), edge_end(N, mflag));
  }

  Runtime::iterable<NoDerefIterator<edge_iterator>> out_edges(GraphNode N, 
      MethodFlag mflag = MethodFlag::WRITE) {
    return edges(N, mflag);
  }

  /**
   * Sorts outgoing edges of a node. Comparison function is over EdgeTy.
   */
  template<typename CompTy>
  void sortEdgesByEdgeData(GraphNode N, 
                           const CompTy& comp = std::less<EdgeTy>(), 
                           MethodFlag mflag = MethodFlag::WRITE) {
    acquireNode(N, mflag);
    std::sort(edge_sort_begin(N), edge_sort_end(N), 
              detail::EdgeSortCompWrapper<EdgeSortValue<GraphNode,EdgeTy>,CompTy>(comp));
  }

  /**
   * Sorts outgoing edges of a node. 
   * Comparison function is over <code>EdgeSortValue<EdgeTy></code>.
   */
  template<typename CompTy>
  void sortEdges(GraphNode N, const CompTy& comp, 
                 MethodFlag mflag = MethodFlag::WRITE) {
    acquireNode(N, mflag);
    std::sort(edge_sort_begin(N), edge_sort_end(N), comp);
  }

  /**
   * Sorts outgoing edges of a node. Comparison is over getEdgeDst(e).
   */
  void sortEdgesByDst(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    acquireNode(N, mflag);
    typedef EdgeSortValue<GraphNode,EdgeTy> EdgeSortVal;
    std::sort(edge_sort_begin(N), edge_sort_end(N), [=] (const EdgeSortVal& e1, const EdgeSortVal& e2) { return e1.dst < e2.dst; });
  }

  /**
   * Sorts all outgoing edges of all nodes in parallel. Comparison is over getEdgeDst(e).
   */
  void sortAllEdgesByDst(MethodFlag mflag = MethodFlag::WRITE) {
    Galois::do_all_local(*this, [=] (GraphNode N) {this->sortEdgesByDst(N, mflag);}, Galois::do_all_steal<true>());
  }

  // find index + divide by node swiped from FileGraph.cpp with some edits

  /**
   * Return a suitable index between an upper bound and a lower bound that
   * attempts to get close to the target size (i.e. find a good chunk that
   * corresponds to some size).
   *
   * @param TODO
   */
  size_t findIndex(size_t nodeSize, size_t edgeSize, size_t targetSize, 
                   size_t lb, size_t ub, std::vector<uint64_t>edgePrefixSum) {
    while (lb < ub) {
      size_t mid = lb + (ub - lb) / 2;
      size_t num_edges;

      if (mid != 0) {
        num_edges = edgePrefixSum[mid - 1];
      } else {
        num_edges = 0;
      }
      //size_t num_edges = *edge_begin(mid);

      size_t size = num_edges * edgeSize + (mid) * nodeSize;

      if (size < targetSize)
        lb = mid + 1;
      else
        ub = mid;
    }
    return lb;
  }

  typedef std::pair<iterator, iterator> NodeRange;
  typedef std::pair<edge_iterator, edge_iterator> EdgeRange;
  typedef std::pair<NodeRange, EdgeRange> GraphRange;

  /** 
   * Returns 2 ranges (one for nodes, one for edges) for a particular division.
   * The ranges specify the memory that a division is responsible for. The
   * function attempts to split memory evenly among threads.
   *
   * @param TODO
   */
  auto divideByNode(size_t nodeSize, size_t edgeSize, size_t id, size_t total,
                    std::vector<uint64_t>edgePrefixSum)
      -> GraphRange {
    // size of all data
    size_t size = numNodes * nodeSize + numEdges * edgeSize;
    // size of a block (one block for each division)
    size_t block = (size + total - 1) / total;
  
    size_t aa = numEdges;
    size_t ea = numEdges;
  
    size_t bb = findIndex(nodeSize, edgeSize, block * id, 0, numNodes, 
                          edgePrefixSum);
    size_t eb = findIndex(nodeSize, edgeSize, block * (id + 1), bb, numNodes, 
                          edgePrefixSum);
  
    if (bb != eb) {
      if (bb != 0) {
        aa = edgePrefixSum[bb - 1];
      } else {
        aa = 0;
      }
      ea = edgePrefixSum[eb - 1];
    }
    return GraphRange(NodeRange(iterator(bb), iterator(eb)), 
                      EdgeRange(edge_iterator(aa), edge_iterator(ea)));
  }


  /**
   * Returns the thread ranges array that specifies division of nodes among
   * threads 
   * 
   * @returns An array of uint32_t that specifies which thread gets which nodes.
   */
  const uint32_t* getThreadRanges() const {
    if (threadRanges.size() == 0) return nullptr;
    return threadRanges.data();
  }

  // DEPRECATED: do not use, only use 1 that takes prefix sum + nodes
  ///** 
  // * Call ONLY after graph is completely constructed. Attempts to more evenly 
  // * distribute nodes among threads by checking the number of edges per
  // * node and determining where each thread should start. 
  // * This should only be done once after graph construction to prevent
  // * too much overhead.
  // **/
  //void determineThreadRanges() {
  //  if (threadRanges.size() != 0) {
  //    // other version already called or this is a second call to this;
  //    // return 
  //    return;
  //  }

  //  uint32_t num_threads = Galois::Runtime::activeThreads;
  //  uint32_t total_nodes = end() - begin();
  //  //printf("nodes is %u\n", total_nodes);

  //  threadRanges.resize(num_threads + 1);

  //  //printf("num owned edges is %lu\n", sizeEdges());

  //  // theoretically how many edges we want to distributed to each thread
  //  uint64_t edges_per_thread = sizeEdges() / num_threads;

  //  //printf("want %lu edges per thread\n", edges_per_thread);

  //  // Case where there are less nodes than threads
  //  if (num_threads > end() - begin()) {
  //    iterator current_node = begin();
  //    uint64_t num_nodes = end() - current_node;
  //    
  //    // assign one node per thread (note not all nodes will get a thread in
  //    // this case
  //    threadRanges[0] = *current_node;
  //    for (uint32_t i = 0; i < num_nodes; i++) {
  //      threadRanges[i+1] = *(++current_node);
  //    }

  //    // deal with remainder threads
  //    for (uint32_t i = num_nodes; i < num_threads; i++) {
  //      threadRanges[i+1] = total_nodes;
  //    }

  //    return;
  //  }

  //  // Single node case
  //  if (num_threads == 1) {
  //    threadRanges[0] = *(begin());
  //    threadRanges[1] = total_nodes;

  //    return;
  //  }


  //  uint32_t current_thread = 0;
  //  uint64_t current_edge_count = 0;
  //  iterator current_local_node = begin();

  //  threadRanges[current_thread] = *(begin());

  //  //printf("going to determine thread ranges\n");

  //  while (current_local_node != end() && current_thread != num_threads) {
  //    uint32_t nodes_remaining = end() - current_local_node;
  //    uint32_t threads_remaining = num_threads - current_thread;
  //   
  //    assert(nodes_remaining >= threads_remaining);

  //    if (threads_remaining == 1) {
  //      // give the rest of the nodes to this thread and get out
  //      printf("Thread %u has %lu edges and %u nodes (only 1 thread)\n",
  //             current_thread, 
  //             edge_end(*(end() - 1)) -
  //             edge_begin(threadRanges[current_thread]),
  //             total_nodes - threadRanges[current_thread]);

  //      threadRanges[current_thread + 1] = total_nodes;

  //      break;
  //    } else if ((end() - current_local_node) == threads_remaining) {
  //      // Out of nodes to assign: finish up assignments (at this point,
  //      // each remaining thread gets 1 node) and break
  //      printf("Out of nodes: assigning the rest of the nodes to remaining "
  //             "threads\n");
  //      for (uint32_t i = 0; i < threads_remaining; i++) {
  //        printf("Thread %u has %lu edges and %u nodes (oon)\n",
  //               current_thread, 
  //               edge_end(*current_local_node) -
  //               edge_begin(threadRanges[current_thread]),
  //               (*current_local_node) + 1 - threadRanges[current_thread]);

  //        threadRanges[++current_thread] = *(++current_local_node);
  //      }

  //      assert(current_local_node == end());
  //      assert(current_thread == num_threads);
  //      break;
  //    }

  //    uint64_t num_edges = std::distance(edge_begin(*current_local_node), 
  //                                       edge_end(*current_local_node));


  //    current_edge_count += num_edges;
  //    //printf("%u has %lu\n", *current_local_node, num_edges);
  //    //printf("[%u] cur edge count is %lu\n", id, current_edge_count);

  //    if (num_edges > (3 * edges_per_thread / 4)) {
  //      if (current_edge_count - num_edges > (edges_per_thread / 2)) {
  //        printf("Thread %u has %lu edges and %u nodes (big)\n",
  //               current_thread, current_edge_count - num_edges,
  //               (*current_local_node) - threadRanges[current_thread]);

  //        // else, assign to the NEXT thread (i.e. end this thread and move
  //        // on to the next)
  //        // beginning of next thread is current local node (the big one)
  //        threadRanges[current_thread + 1] = *current_local_node;
  //        current_thread++;

  //        current_edge_count = 0;
  //        // go back to beginning of loop without incrementing
  //        // current_local_node so the loop can handle this next node
  //        continue;
  //      }
  //    }

  //    if (current_edge_count >= edges_per_thread) {
  //      printf("Thread %u has %lu edges and %u nodes (over)\n",
  //             current_thread, current_edge_count,
  //             (*current_local_node) + 1 - threadRanges[current_thread]);

  //      // This thread has enough edges; save end of this node and move on
  //      // mark beginning of next thread as the next node
  //      threadRanges[current_thread + 1] = (*current_local_node) + 1;
  //      current_edge_count = 0;
  //      current_thread++;
  //    } 
  //    
  //    current_local_node++;
  //  }

  //  // sanity checks
  //  assert(threadRanges[0] == 0);
  //  assert(threadRanges[num_threads] == total_nodes);
  //  printf("ranges found\n");
  //}


  /**
   * Uses a pre-computed prefix sum of edges to determine division of nodes 
   * among threads.
   *
   * @param totalNodes The total number of nodes (masters + mirrors) on this
   * partition.
   * @param edgePrefixSum The edge prefix sum of the nodes on this partition.
   * @param nodeAlpha The higher the number, the more weight nodes have in
   * determining division of nodes.
   */
  template <typename VectorTy>
  void determineThreadRanges(uint32_t totalNodes, VectorTy& edgePrefixSum,
                             uint32_t nodeAlpha=0) {
    // if we already have thread ranges calculated, do nothing and return
    if (threadRanges.size() != 0) {
      return;
    }

    assert(edgePrefixSum.size() == totalNodes);
    uint32_t num_threads = Galois::Runtime::activeThreads;

    threadRanges.resize(num_threads + 1);

    // Single thread case
    if (num_threads == 1) {
      threadRanges[0] = 0;
      threadRanges[1] = totalNodes;

      return;
    } else if (num_threads > totalNodes) {
      // more threads than nodes
      uint32_t current_node = 0;

      threadRanges[0] = current_node;
      for (uint32_t i = 0; i < totalNodes; i++) {
        threadRanges[i+1] = ++current_node;
      }

      // deal with remainder threads
      for (uint32_t i = totalNodes; i < num_threads; i++) {
        threadRanges[i+1] = totalNodes;
      }

      return;
    }

    uint64_t node_weight = (uint64_t)totalNodes * (uint64_t)nodeAlpha;

    // theoretically how many units we want to distributed to each thread
    uint64_t units_per_thread = edgePrefixSum[totalNodes - 1] / num_threads +
                                node_weight;

    printf("Optimally want %lu units per thread\n", units_per_thread);

    uint32_t current_thread = 0;
    uint32_t current_element = 0;

    uint64_t accounted_edges = 0;
    uint32_t accounted_elements = 0;

    threadRanges[0] = 0;

    while (current_element < totalNodes && current_thread < num_threads) {
      uint64_t elements_remaining = totalNodes - current_element;
      uint32_t threads_remaining = num_threads - current_thread;

      assert(elements_remaining >= threads_remaining);

      if (threads_remaining == 1) {
        // assign remaining elements to last thread
        assert(current_thread == num_threads - 1); 
        threadRanges[current_thread + 1] = totalNodes;
        printf("Thread %u begin %u end %u (1 thread)\n", current_thread,
               threadRanges[current_thread], threadRanges[current_thread+1]);

        break;
      } else if ((totalNodes - current_element) == threads_remaining) {
        // Out of elements to assign: finish up assignments (at this point,
        // each remaining thread gets 1 element except for the current
        // thread which may have some already)
        for (uint32_t i = 0; i < threads_remaining; i++) {
          threadRanges[++current_thread] = (++current_element);
          printf("Thread %u begin %u end %u (out of element)\n", 
                 current_thread-1, threadRanges[current_thread-1], 
                 threadRanges[current_thread]);
        }

        assert(current_element == totalNodes);
        assert(current_thread == num_threads);
        break;
      }

      //
      // Determine various count numbers from prefix sum
      //

      // num edges this element has
      uint64_t element_edges; 
      if (current_element > 0) {
        element_edges = edgePrefixSum[current_element] - 
                        edgePrefixSum[current_element - 1];
      } else {
        element_edges = edgePrefixSum[0];
      }
  
      // num edges this division currently has not taking into account 
      // this element we are currently on
      uint64_t edge_count_without_current;
      if (current_element > 0) {
        edge_count_without_current = edgePrefixSum[current_element] -
                                     accounted_edges - element_edges;
      } else {
        edge_count_without_current = 0;
      }

      // determine current unit count by taking into account nodes already 
      // handled
      uint64_t unit_count_without_current = 
         edge_count_without_current + 
         ((current_element - accounted_elements) * nodeAlpha); 

      // include node into weight of this element
      uint64_t element_units = element_edges + nodeAlpha;

      if (element_units > (3 * units_per_thread / 4)) {
        // if this current thread + units of this element is too much,
        // then do not add to this thread but rather the next one
        if (unit_count_without_current > (units_per_thread / 2)) {
          assert(current_element != 0);

          // assign to the NEXT thread (i.e. end this thread and move on to the 
          // next)
          // beginning of next thread is current local element (i.e. the big 
          // one we are giving up to the next)
          threadRanges[current_thread + 1] = current_element;
          printf("Thread %u begin %u end %u (big)\n", 
                 current_thread, threadRanges[current_thread], 
                 threadRanges[current_thread+1]);

          accounted_edges = edgePrefixSum[current_element - 1];
          accounted_elements = current_element;

          current_thread++;

          // go back to beginning of loop 
          continue;
        }
      }

      // handle this element by adding edges to running sums
      uint64_t unit_count_with_current = unit_count_without_current + 
                                         element_units;

      if (unit_count_with_current >= units_per_thread) {
        threadRanges[++current_thread] = current_element + 1;
        printf("Thread %u begin %u end %u (over)\n", current_thread-1,
               threadRanges[current_thread - 1], 
               threadRanges[current_thread]);
        //printf("sum is %lu\n", edgePrefixSum[current_thread-1]);

        accounted_edges = edgePrefixSum[current_element];
        accounted_elements = current_element + 1;
      }

      current_element++;
    }

    assert(threadRanges[0] == 0);
    assert(threadRanges[num_threads] == totalNodes);
  }

  /**
   * Uses the divideByNode function from FileGraph to do partitioning
   * of nodes/edges among threads.
   *
   * @param edgePrefixSum A prefix sum of edges
   */
  void determineThreadRangesByNode(std::vector<uint64_t>& edgePrefixSum) {
    uint32_t numThreads = Galois::Runtime::activeThreads;
    assert(numThreads > 0);

    if (threadRanges.size() != 0) {
      printf("Warning: Thread ranges already specified\n");
    }

    if (threadRangesEdge.size() != 0) {
      printf("Warning: Thread ranges edge already specified\n");
    }

    clearRanges();
    threadRanges.resize(numThreads + 1);
    threadRangesEdge.resize(numThreads + 1);

    threadRanges[0] = 0;
    threadRangesEdge[0] = 0;

    for (uint32_t i = 0; i < numThreads; i++) {
      auto nodeEdgeSplits = divideByNode(
          NodeData::size_of::value + EdgeIndData::size_of::value + 
          LC_CSR_Graph::size_of_out_of_line::value,
          EdgeDst::size_of::value + EdgeData::size_of::value,
          i, numThreads, edgePrefixSum);

      auto nodeSplits = nodeEdgeSplits.first;
      auto edgeSplits = nodeEdgeSplits.second;

      if (nodeSplits.first != nodeSplits.second) {
        if (i != 0) {
          //printf("thread ranges edge i is %lu\n", threadRangesEdge[i]);
          //printf("edgeSplits first is %lu\n", *edgeSplits.first);
          assert(threadRanges[i] == *(nodeSplits.first));
          assert(threadRangesEdge[i] == *(edgeSplits.first));
        } else {
          // = 0
          assert(threadRanges[i] == 0);
          assert(threadRangesEdge[i] == 0);
        }

        threadRanges[i + 1] = *(nodeSplits.second);
        threadRangesEdge[i + 1] = *(edgeSplits.second);
      } else {
        // thread assinged no nodes
        assert(edgeSplits.first == edgeSplits.second);

        threadRanges[i + 1] = threadRanges[i];
        threadRangesEdge[i + 1] = threadRangesEdge[i];
      }
      printf("thread %u gets nodes %u to %u\n", i, threadRanges[i], 
              threadRanges[i+1]);
      printf("thread %u gets edges %lu to %lu\n", i, threadRangesEdge[i], 
              threadRangesEdge[i+1]);
    }
  }

  /**
   * Determine the ranges for the edges of a graph for each thread given the
   * prefix sum. threadRanges must already be calculated.
   *
   * @param edgePrefixSum Prefix sum of edges 
   */
  template <typename VectorTy>
  void determineThreadRangesEdge(VectorTy& edgePrefixSum) {
    if (threadRanges.size() > 0) {
      uint32_t totalThreads = Galois::Runtime::activeThreads;

      threadRangesEdge.resize(totalThreads + 1);
      threadRangesEdge[0] = 0;

      // determine edge range for each active thread
      for (uint32_t i = 0; i < totalThreads; i++) {
        uint32_t beginNode = threadRanges[i];
        uint32_t endNode = threadRanges[i + 1];

        uint64_t beginEdge;
        if (beginNode > 0) {
          beginEdge = edgePrefixSum[beginNode - 1];
        } else {
          beginEdge = 0;
        }

        uint64_t endEdge = edgePrefixSum[endNode - 1];

        assert(beginEdge <= endEdge);
        assert(endEdge <= numEdges);

        assert(threadRangesEdge[i] == beginEdge);
        //printf("[%u] begin edge is %lu, end edge is %lu\n", i, beginEdge, 
        //       endEdge);
        threadRangesEdge[i + 1] = endEdge;
      }
    } else {
      printf("WARNING: threadRangesEdge not calculated because threadRanges "
             "isn't calculated.\n");
    }
  }

  template <typename F>
  ptrdiff_t partition_neighbors(GraphNode N, const F& func) {
    auto beg = &edgeDst[*raw_begin (N)];
    auto end = &edgeDst[*raw_end (N)];
    auto mid = std::partition (beg, end, func);
    return (mid - beg);
  }


  void allocateFrom(FileGraph& graph) {
    numNodes = graph.size();
    numEdges = graph.sizeEdges();
    if (UseNumaAlloc) {
      //nodeData.allocateInterleaved(numNodes);
      //edgeIndData.allocateInterleaved(numNodes);
      //edgeDst.allocateInterleaved(numEdges);
      //edgeData.allocateInterleaved(numEdges);
      //this->outOfLineAllocateInterleaved(numNodes);

      nodeData.allocateBlocked(numNodes);
      edgeIndData.allocateBlocked(numNodes);
      edgeDst.allocateBlocked(numEdges);
      edgeData.allocateBlocked(numEdges);
      this->outOfLineAllocateBlocked(numNodes);

      //nodeData.allocateLocal(numNodes);
      //edgeIndData.allocateLocal(numNodes);
      //edgeDst.allocateLocal(numEdges);
      //edgeData.allocateLocal(numEdges);
      //this->outOfLineAllocateLocal(numNodes);

      //nodeData.allocateFloating(numNodes);
      //edgeIndData.allocateFloating(numNodes);
      //edgeDst.allocateFloating(numEdges);
      //edgeData.allocateFloating(numEdges);
      //this->outOfLineAllocateFloating(numNodes);
    } else {
      //nodeData.allocateLocal(numNodes);
      //edgeIndData.allocateLocal(numNodes);
      //edgeDst.allocateLocal(numEdges);
      //edgeData.allocateLocal(numEdges);
      //this->outOfLineAllocateLocal(numNodes);
      //nodeData.allocateBlocked(numNodes);
      //edgeIndData.allocateBlocked(numNodes);
      //edgeDst.allocateBlocked(numEdges);
      //edgeData.allocateBlocked(numEdges);
      //this->outOfLineAllocateBlocked(numNodes);
      nodeData.allocateInterleaved(numNodes);
      edgeIndData.allocateInterleaved(numNodes);
      edgeDst.allocateInterleaved(numEdges);
      edgeData.allocateInterleaved(numEdges);
      this->outOfLineAllocateInterleaved(numNodes);
    }
  }

  void allocateFrom(uint32_t nNodes, uint64_t nEdges) {
    numNodes = nNodes;
    numEdges = nEdges;
    if (UseNumaAlloc) {
      //nodeData.allocateLocal(numNodes);
      //edgeIndData.allocateLocal(numNodes);
      //edgeDst.allocateLocal(numEdges);
      //edgeData.allocateLocal(numEdges);
      //this->outOfLineAllocateLocal(numNodes);
      nodeData.allocateInterleaved(numNodes);
      edgeIndData.allocateInterleaved(numNodes);
      edgeDst.allocateInterleaved(numEdges);
      edgeData.allocateInterleaved(numEdges);
      this->outOfLineAllocateInterleaved(numNodes);
    } else {
      nodeData.allocateInterleaved(numNodes);
      edgeIndData.allocateInterleaved(numNodes);
      edgeDst.allocateInterleaved(numEdges);
      edgeData.allocateInterleaved(numEdges);
      this->outOfLineAllocateInterleaved(numNodes);

      //nodeData.allocateLocal(numNodes);
      //edgeIndData.allocateLocal(numNodes);
      //edgeDst.allocateLocal(numEdges);
      //edgeData.allocateLocal(numEdges);
      //this->outOfLineAllocateLocal(numNodes);

      //nodeData.allocateBlocked(numNodes);
      //edgeIndData.allocateBlocked(numNodes);
      //edgeDst.allocateBlocked(numEdges);
      //edgeData.allocateBlocked(numEdges);
      //this->outOfLineAllocateBlocked(numNodes);

      //nodeData.allocateFloating(numNodes);
      //edgeIndData.allocateFloating(numNodes);
      //edgeDst.allocateFloating(numEdges);
      //edgeData.allocateFloating(numEdges);
      //this->outOfLineAllocateFloating(numNodes);
    }
  }

  /**
   * A version of allocateFrom that takes into account edge distribution across
   * nodes.
   *
   * @param nNodes Number to allocate for node data
   * @param nEdges Number to allocate for edge data
   * @param edgePrefixSum Vector with prefix sum of edges.
   */
  void allocateFrom(uint32_t nNodes, uint64_t nEdges, 
                    std::vector<uint64_t> edgePrefixSum) {
    //printf("My split\n");
    // update graph values
    numNodes = nNodes;
    numEdges = nEdges;

    // calculate thread ranges for nodes and edges
    determineThreadRanges(numNodes, edgePrefixSum);
    determineThreadRangesEdge(edgePrefixSum);

    // node based alloc
    nodeData.allocateSpecified(numNodes, threadRanges);
    edgeIndData.allocateSpecified(numNodes, threadRanges);
    //nodeData.allocateLocal(numNodes);
    //edgeIndData.allocateLocal(numNodes);

    // edge based alloc
    edgeDst.allocateSpecified(numEdges, threadRangesEdge);
    edgeData.allocateSpecified(numEdges, threadRangesEdge);
    //edgeDst.allocateLocal(numEdges);
    //edgeData.allocateLocal(numEdges);

    this->outOfLineAllocateSpecified(numNodes, threadRanges);
    //this->outOfLineAllocateLocal(numNodes);
  }


  /**
   * A version of allocateFrom that takes into account node/edge distribution
   * and tries to make the allocation of memory more. 
   * Based on divideByNode in Lonestar.
   *
   * @param graph A graph in the FileGraph class.
   */
  void allocateFromByNode(FileGraph& graph) {
    //printf("alloc by node\n");
    numNodes = graph.size();
    numEdges = graph.sizeEdges();

    uint32_t numThreads = Galois::Runtime::activeThreads;
    assert(numThreads > 0);

    threadRanges.resize(numThreads + 1);
    threadRangesEdge.resize(numThreads + 1);

    threadRanges[0] = 0;
    threadRangesEdge[0] = 0;

    for (uint32_t i = 0; i < numThreads; i++) {
      auto nodeEdgeSplits = graph.divideByNode(
          NodeData::size_of::value + EdgeIndData::size_of::value + 
          LC_CSR_Graph::size_of_out_of_line::value,
          EdgeDst::size_of::value + EdgeData::size_of::value,
          i, numThreads);

      auto nodeSplits = nodeEdgeSplits.first;
      auto edgeSplits = nodeEdgeSplits.second;

      if (nodeSplits.first != nodeSplits.second) {
        if (i != 0) {
          //printf("thread ranges edge i is %lu\n", threadRangesEdge[i]);
          //printf("edgeSplits first is %lu\n", *edgeSplits.first);
          assert(threadRanges[i] == *(nodeSplits.first));
          assert(threadRangesEdge[i] == *(edgeSplits.first));
        } else {
          // = 0
          assert(threadRanges[i] == 0);
          assert(threadRangesEdge[i] == 0);
        }

        threadRanges[i + 1] = *(nodeSplits.second);
        threadRangesEdge[i + 1] = *(edgeSplits.second);
      } else {
        // thread assinged no nodes
        assert(edgeSplits.first == edgeSplits.second);

        threadRanges[i + 1] = threadRanges[i];
        threadRangesEdge[i + 1] = threadRangesEdge[i];
      }
      printf("thread %u gets nodes %u to %u\n", i, threadRanges[i], 
              threadRanges[i+1]);
      printf("thread %u gets edges %lu to %lu\n", i, threadRangesEdge[i], 
              threadRangesEdge[i+1]);
    }

    // node based alloc
    nodeData.allocateSpecified(numNodes, threadRanges);
    edgeIndData.allocateSpecified(numNodes, threadRanges);
    //nodeData.allocateLocal(numNodes);
    //edgeIndData.allocateLocal(numNodes);

    // edge based alloc
    edgeDst.allocateSpecified(numEdges, threadRangesEdge);
    edgeData.allocateSpecified(numEdges, threadRangesEdge);
    //edgeDst.allocateLocal(numEdges);
    //edgeData.allocateLocal(numEdges);

    this->outOfLineAllocateSpecified(numNodes, threadRanges);
    //this->outOfLineAllocateLocal(numNodes);
  }


  /**
   * A version of allocateFrom that takes into account node/edge distribution
   * and tries to make the allocation of memory more. 
   * Based on divideByNode in Lonestar.
   *
   * @param nNodes Number to allocate for node data
   * @param nEdges Number to allocate for edge data
   * @param edgePrefixSum Vector with prefix sum of edges.
   */
  void allocateFromByNode(uint32_t nNodes, uint64_t nEdges, 
                          std::vector<uint64_t> edgePrefixSum) {
    printf("by node allocate specified\n");
    numNodes = nNodes;
    numEdges = nEdges;

    // gets both threadRanges + threadRangesEdge
    determineThreadRangesByNode(edgePrefixSum);

    // node based alloc
    nodeData.allocateSpecified(numNodes, threadRanges);
    edgeIndData.allocateSpecified(numNodes, threadRanges);
    //nodeData.allocateLocal(numNodes);
    //edgeIndData.allocateLocal(numNodes);

    // edge based alloc
    edgeDst.allocateSpecified(numEdges, threadRangesEdge);
    edgeData.allocateSpecified(numEdges, threadRangesEdge);
    //edgeDst.allocateLocal(numEdges);
    //edgeData.allocateLocal(numEdges);

    this->outOfLineAllocateSpecified(numNodes, threadRanges);
  }

  void constructNodes() {
#ifndef GALOIS_GRAPH_CONSTRUCT_SERIAL
    for (uint32_t x = 0; x < numNodes; ++x) {
      nodeData.constructAt(x);
      this->outOfLineConstructAt(x);
    }
#else
    Galois::do_all(boost::counting_iterator<uint32_t>(0), 
                   boost::counting_iterator<uint32_t>(numNodes),
                   [&](uint32_t x) {
                     nodeData.constructAt(x);
                     this->outOfLineConstructAt(x);
                   }, 
                   Galois::loopname("CONSTRUCT_NODES"), 
                   Galois::numrun("0"));
#endif
  }

  void constructEdge(uint64_t e, uint32_t dst, const typename EdgeData::value_type& val) {
    edgeData.set(e, val);
    edgeDst[e] = dst;
  }

  void constructEdge(uint64_t e, uint32_t dst) {
    edgeDst[e] = dst;
  }

  void fixEndEdge(uint32_t n, uint64_t e) {
    edgeIndData[n] = e;
  }

  /** 
   * Perform an in-memory transpose of the graph, replacing the original
   * CSR to CSC
   */
  void transpose(bool reallocate = false) {
    Galois::StatTimer timer("TIME_GRAPH_TRANSPOSE");
    timer.start();

    EdgeDst edgeDst_old;
    EdgeData edgeData_new;
    EdgeIndData edgeIndData_old;
    EdgeIndData edgeIndData_temp;

    edgeIndData_old.allocateInterleaved(numNodes);
    edgeIndData_temp.allocateInterleaved(numNodes);
    edgeDst_old.allocateInterleaved(numEdges);
    edgeData_new.allocateInterleaved(numEdges);

    // Copy old node->index location + initialize the temp array
    Galois::do_all(boost::counting_iterator<uint32_t>(0), 
                   boost::counting_iterator<uint32_t>(numNodes),
                   [&](uint32_t n) {
                     edgeIndData_old[n] = edgeIndData[n];
                     edgeIndData_temp[n] = 0;
                   }, 
                   Galois::loopname("TRANSPOSE_EDGEINTDATA_COPY"), 
                   Galois::numrun("0"));

    // parallelization makes this slower
    // get destination of edge, copy to array, and  
    for (uint64_t e = 0; e < numEdges; ++e) { 
        auto dst = edgeDst[e];
        edgeDst_old[e] = dst;
        // counting outgoing edges in the tranpose graph by
        // counting incoming edges in the original graph
        ++edgeIndData_temp[dst];
    }

    // prefix sum calculation of the edge index array
    for (uint32_t n = 1; n < numNodes; ++n) {
      edgeIndData_temp[n] += edgeIndData_temp[n-1];
    }

    // recalculate thread ranges for nodes and edges
    clearRanges();
    determineThreadRanges(numNodes, edgeIndData_temp);
    determineThreadRangesEdge(edgeIndData_temp);

    // reallocate nodeData
    if (reallocate) {
      nodeData.deallocate();
      nodeData.allocateSpecified(numNodes, threadRanges);
    }

    // reallocate edgeIndData
    if (reallocate) {
      edgeIndData.deallocate();
      edgeIndData.allocateSpecified(numNodes, threadRanges);
    }

    // copy over the new tranposed edge index data
    Galois::do_all(boost::counting_iterator<uint32_t>(0), 
                   boost::counting_iterator<uint32_t>(numNodes),
                   [&](uint32_t n) {
                     edgeIndData[n] = edgeIndData_temp[n];
                   }, 
                   Galois::loopname("TRANSPOSE_EDGEINTDATA_SET"), 
                   Galois::numrun("0"));

    // edgeIndData_temp[i] will now hold number of edges that all nodes
    // before the ith node have
    edgeIndData_temp[0] = 0;
    Galois::do_all(boost::counting_iterator<uint32_t>(1), 
                   boost::counting_iterator<uint32_t>(numNodes),
                   [&](uint32_t n){
                     edgeIndData_temp[n] = edgeIndData[n-1];
                   }, 
                   Galois::loopname("TRANSPOSE_EDGEINTDATA_TEMP"), 
                   Galois::numrun("0"));

    // reallocate edgeDst
    if (reallocate) {
      edgeDst.deallocate();
      edgeDst.allocateSpecified(numEdges, threadRangesEdge);
    }

    // parallelization makes this slower
    for (uint32_t src = 0; src < numNodes; ++src) { 
      // e = start index into edge array for a particular node
      uint64_t e;
      if (src == 0) 
        e = 0;
      else 
        e = edgeIndData_old[src - 1];

      // get all outgoing edges of a particular node in the non-tranpose and
      // convert to incoming
      while (e < edgeIndData_old[src]) {
        // destination nodde
        auto dst = edgeDst_old[e];
        // location to save edge
        auto e_new = edgeIndData_temp[dst]++;
        // save src as destination
        edgeDst[e_new] = src;
        // copy edge data to "new" array
        edgeDataCopy(edgeData_new, edgeData, e_new, e);
        e++;
      }
    }

    // reallocate edgeData
    if (reallocate) {
      edgeData.deallocate();
      edgeData.allocateSpecified(numEdges, threadRangesEdge);
    }

    // if edge weights, then overwrite edgeData with new edge data
    if (EdgeData::has_value) {
      Galois::do_all(boost::counting_iterator<uint32_t>(0), 
                     boost::counting_iterator<uint32_t>(numEdges),
                     [&](uint32_t e){
                       edgeDataCopy(edgeData, edgeData_new, e, e);
                     }, 
                     Galois::loopname("TRANSPOSE_EDGEDATA_SET"), 
                     Galois::numrun("0"));
    }

    timer.stop();
  }

  template<bool is_non_void = EdgeData::has_value>
  void edgeDataCopy(EdgeData &edgeData_new, EdgeData &edgeData, uint64_t e_new, uint64_t e, typename std::enable_if<is_non_void>::type* = 0) {
    edgeData_new[e_new] = edgeData[e];
  }

  template<bool is_non_void = EdgeData::has_value>
  void edgeDataCopy(EdgeData &edgeData_new, EdgeData &edgeData, uint64_t e_new, uint64_t e, typename std::enable_if<!is_non_void>::type* = 0) {
  }


  void constructFrom(FileGraph& graph, unsigned tid, unsigned total) {
    // at this point memory is already allocated
    //auto both = graph.divideByNode(
    //    NodeData::size_of::value + EdgeIndData::size_of::value + 
    //    LC_CSR_Graph::size_of_out_of_line::value,
    //    EdgeDst::size_of::value + EdgeData::size_of::value,
    //    tid, total);
    //auto r = both.first;
    //auto edgeSplit = both.second;

    auto r = graph.divideByNode(
        NodeData::size_of::value + EdgeIndData::size_of::value + 
        LC_CSR_Graph::size_of_out_of_line::value,
        EdgeDst::size_of::value + EdgeData::size_of::value,
        tid, total).first;

    this->setLocalRange(*r.first, *r.second);

    for (FileGraph::iterator ii = r.first, ei = r.second; ii != ei; ++ii) {
      nodeData.constructAt(*ii);
      edgeIndData[*ii] = *graph.edge_end(*ii);

      this->outOfLineConstructAt(*ii);

      for (FileGraph::edge_iterator nn = graph.edge_begin(*ii), 
                                    en = graph.edge_end(*ii); 
           nn != en;
           ++nn) {
        constructEdgeValue(graph, nn);
        edgeDst[*nn] = graph.getEdgeDst(nn);
      }
    }
  }
};

} // end namespace
} // end namespace

#endif
