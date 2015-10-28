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
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#ifndef GALOIS_GRAPH_LC_CSR_MIC_GRAPH_H
#define GALOIS_GRAPH_LC_CSR_MIC_GRAPH_H

#include "Galois/LargeArray.h"
#include "Galois/Graphs/FileGraph.h"
#include "Galois/Graphs/Details.h"

#include <xmmintrin.h>

#include <type_traits>

// #define _DO_INNER_PREFETCH 1


// #define _DO_OUTER_PREFETCH 1


#if defined(__INTEL_COMPILER)
#define _DO_UNROLL
#endif

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
 * \code
 * typedef Galois::Graph::LC_CSR_Graph<int,int> Graph;
 * 
 * // Create graph
 * Graph g;
 * Galois::Graph::readGraph(g, inputfile);
 *
 * // Traverse graph
 * for (Graph::iterator ii = g.begin(), ei = g.end(); ii != ei; ++ii) {
 *   Graph::GraphNode src = *ii;
 *   for (Graph::edge_iterator jj = g.edge_begin(src), ej = g.edge_end(src); jj != ej; ++jj) {
 *     Graph::GraphNode dst = g.getEdgeDst(jj);
 *     int edgeData = g.getEdgeData(jj);
 *     int nodeData = g.getData(dst);
 *   }
 * }
 * \endcode
 *
 * And in C++11:
 *
 * \code
 * typedef Galois::Graph::LC_CSR_Graph<int,int> Graph;
 * // or typedef Galois::Graph::LC_CSR_Graph<int,int>::with_no_lockable<true>::with_numa_alloc<true>
 *
 * // Create graph
 * Graph g;
 * Galois::Graph::readGraph(g, inputfile);
 *
 * // Traverse graph
 * for (Graph::GraphNode src : g) {
 *   for (Graph::edge_iterator edge : g.out_edges(src)) {
 *     Graph::GraphNode dst = g.getEdgeDst(edge);
 *     int edgeData = g.getEdgeData(edge);
 *     int nodeData = g.getData(dst);
 *   }
 * }
 * \endcode
 *
 * @tparam NodeTy data on nodes
 * @tparam EdgeTy data on out edges
 */
template<typename NodeTy, typename EdgeTy,
  bool HasNoLockable=false,
  bool UseNumaAlloc=false,
  bool HasOutOfLineLockable=false,
  typename FileEdgeTy=EdgeTy>
class LC_CSR_MIC_Graph:
    private boost::noncopyable,
    private detail::LocalIteratorFeature<UseNumaAlloc>,
    private detail::OutOfLineLockableFeature<HasOutOfLineLockable && !HasNoLockable> {
  template<typename Graph> friend class LC_InOut_Graph;

public:
  template<bool _has_id>
  struct with_id { typedef LC_CSR_MIC_Graph type; };

  template<typename _node_data>
  struct with_node_data { typedef LC_CSR_MIC_Graph<_node_data,EdgeTy,HasNoLockable,UseNumaAlloc,HasOutOfLineLockable,FileEdgeTy> type; };

  template<typename _edge_data>
  struct with_edge_data { typedef LC_CSR_MIC_Graph<NodeTy,_edge_data,HasNoLockable,UseNumaAlloc,HasOutOfLineLockable,FileEdgeTy> type; };

  template<typename _file_edge_data>
  struct with_file_edge_data { typedef LC_CSR_MIC_Graph<NodeTy,EdgeTy,HasNoLockable,UseNumaAlloc,HasOutOfLineLockable,_file_edge_data> type; };

  //! If true, do not use abstract locks in graph
  template<bool _has_no_lockable>
  struct with_no_lockable { typedef LC_CSR_MIC_Graph<NodeTy,EdgeTy,_has_no_lockable,UseNumaAlloc,HasOutOfLineLockable,FileEdgeTy> type; };

  //! If true, use NUMA-aware graph allocation
  template<bool _use_numa_alloc>
  struct with_numa_alloc { typedef LC_CSR_MIC_Graph<NodeTy,EdgeTy,HasNoLockable,_use_numa_alloc,HasOutOfLineLockable,FileEdgeTy> type; };

  //! If true, store abstract locks separate from nodes
  template<bool _has_out_of_line_lockable>
  struct with_out_of_line_lockable { typedef LC_CSR_MIC_Graph<NodeTy,EdgeTy,HasNoLockable,UseNumaAlloc,_has_out_of_line_lockable,FileEdgeTy> type; };

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

  typedef typename EdgeIndData::value_type EdgeIndex;

protected:
  NodeData nodeData;
  EdgeIndData edgeIndData;
  EdgeDst edgeDst;
  EdgeData edgeData;

  uint64_t numNodes;
  uint64_t numEdges;

  typedef detail::EdgeSortIterator<GraphNode,typename EdgeIndData::value_type,EdgeDst,EdgeData> edge_sort_iterator;

  EdgeIndex raw_begin(GraphNode N) const {
    assert (N < edgeIndData.size ());
    return edgeIndData[N];
  }

  EdgeIndex raw_end(GraphNode N) const {
    assert ((N + 1) < edgeIndData.size ());
    return edgeIndData[N + 1];
  }

  edge_sort_iterator edge_sort_begin(GraphNode N) {
    return edge_sort_iterator(*edge_iterator (raw_begin(N)), &edgeDst, &edgeData);
  }

  edge_sort_iterator edge_sort_end(GraphNode N) {
    return edge_sort_iterator(*edge_iterator (raw_end(N)), &edgeDst, &edgeData);
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
  node_data_reference getData(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    // Galois::Runtime::checkWrite(mflag, false);
    NodeInfo& NI = nodeData[N];
    acquireNode(N, mflag);
    return NI.getData();
  }

  void prefetchData (GraphNode N, const unsigned pftch_kind) const {
#ifdef _DO_OUTER_PREFETCH
    if (pftch_kind == _MM_HINT_T1) {
      _mm_prefetch ((const char*) &nodeData[N], _MM_HINT_T1);
    } else {
      _mm_prefetch ((const char*) &nodeData[N], _MM_HINT_T0);
    }
#endif
  }

  void prefetchOutEdges (GraphNode N, const unsigned pftch_kind) const {
  }

  void prefetchOutNeighbors (GraphNode N, const unsigned pftch_kind) const {
#ifdef _DO_OUTER_PREFETCH
    const EdgeIndex beg = raw_begin (N);
    const EdgeIndex end = raw_end (N);

    if (pftch_kind == _MM_HINT_T1) {
      const unsigned l1_pftch_count = 4;
      const unsigned prev_pftch_count = l1_pftch_count;
      const unsigned l2_pftch_count = 16;
#ifdef _DO_UNROLL
#pragma unroll (l2_pftch_count)
#endif
      for (unsigned j = 0; j < l2_pftch_count; ++j) {
        if ((beg + prev_pftch_count + j) < end) {
          GraphNode next = edgeDst[beg + prev_pftch_count + j];
          _mm_prefetch ((const char*) &nodeData[next], _MM_HINT_T1);
        }
      }
#ifdef _DO_UNROLL
#pragma unroll (l1_pftch_count)
#endif
      for (unsigned j = 0; j < l1_pftch_count; ++j) {
        if ((beg + j) < end) {
          GraphNode next = edgeDst[beg + j];
          _mm_prefetch ((const char*) &nodeData[next], _MM_HINT_T0);
        }
      }
    } else { // pftch_kind == _MM_HINT_T0
      
      const unsigned l1_pftch_count = 4;
      // const unsigned prev_pftch_count = l1_pftch_count;
      // const unsigned l2_pftch_count = 4;
// #pragma unroll (l2_pftch_count)
      // for (unsigned j = 0; j < l2_pftch_count; ++j) {
        // if ((beg + prev_pftch_count + j) < end) {
          // GraphNode next = edgeDst[beg + prev_pftch_count + j];
          // _mm_prefetch ((const char*) &nodeData[next], _MM_HINT_T1);
        // }
      // }
#pragma unroll (l1_pftch_count)
      for (unsigned j = 0; j < l1_pftch_count; ++j) {
        if ((beg + j) < end) {
          GraphNode next = edgeDst[beg + j];
          _mm_prefetch ((const char*) &nodeData[next], _MM_HINT_T0);
        }
      }
    }
#endif
  }

  template <typename F>
  void mapOutEdges (GraphNode N, const F& func, MethodFlag flag = MethodFlag::WRITE) {
    const EdgeIndex beg = raw_begin (N);
    const EdgeIndex end = raw_end (N);

    for (EdgeIndex i = beg; i < end; ++i) {
      GraphNode dst = edgeDst[i];
      func (dst);
    }
  }

  template <typename F>
  void mapOutNeighbors (GraphNode N, const F& func, MethodFlag mflag = MethodFlag::WRITE) {
    const EdgeIndex beg = raw_begin (N);
    const EdgeIndex end = raw_end (N);


#if 0
    const int pftch_dist = 1;
    for (EdgeIndex i = beg, p = beg + pftch_dist; i < end; ++i) {

#pragma unroll (pftch_count)
      for (unsigned j = 0; j < pftch_count; ++j) {
      // for(EdgeIndex end_p = std::min(p + pftch_count, end); p < end_p; ++p) {
        if (p < end) {
          GraphNode next = edgeDst[p];
          _mm_prefetch ((const char*) &nodeData[next], _MM_HINT_T0);
          ++p;

        } 
      }

      GraphNode dst = edgeDst[i];
      func (dst);
    }
#endif

    const unsigned pftch_count = 4;

#ifdef _DO_INNER_PREFETCH
#ifdef _DO_UNROLL
#pragma unroll (pftch_count)
#endif
    for (unsigned j = 0; j < pftch_count; ++j) {
      if ((beg + j) < end) {
        GraphNode next = edgeDst[beg + j];
        _mm_prefetch ((const char*) &nodeData[next], _MM_HINT_T0);
      }
    }

// #pragma unroll (pftch_count)
    // for (unsigned j = 0; j < pftch_count; ++j) {
      // if ((beg + pftch_count + j) < end) {
        // GraphNode next = edgeDst[beg + pftch_count + j];
        // _mm_prefetch ((const char*) &nodeData[next], _MM_HINT_T1);
      // }
    // }
#endif

// #pragma unroll_and_jam (pftch_count)
    for (EdgeIndex i = beg; i < end; i += pftch_count) {


#ifdef _DO_INNER_PREFETCH
#ifdef _DO_UNROLL
#pragma unroll (pftch_count)
#endif
      for (unsigned j = 0; j < pftch_count; ++j) {
        if ((i + j + pftch_count) < end) {
          GraphNode next = edgeDst[i + j + pftch_count];
          _mm_prefetch ((const char*) &nodeData[next], _MM_HINT_T0);
        }
      }

// #pragma unroll (pftch_count)
      // for (unsigned j = 0; j < pftch_count; ++j) {
        // if ((i + j + pftch_count + pftch_count) < end) {
          // GraphNode next = edgeDst[i + j + pftch_count + pftch_count];
          // _mm_prefetch ((const char*) &nodeData[next], _MM_HINT_T1);
        // }
      // }
#endif

#ifdef _DO_UNROLL
#pragma unroll (pftch_count)
#endif
      for (unsigned j = 0; j < pftch_count; ++j) {
        if ((i + j) < end) {
          GraphNode dst = edgeDst[i + j];
          func (dst);
        }
      }

    }



#if 0
    for (EdgeIndex i = beg; i < end; ++i) {
#ifdef _DO_INNER_PREFETCH
      const int pftch_dist = 3;
      if (i + pftch_dist < end) {
        GraphNode next = edgeDst[i + pftch_dist];
        _mm_prefetch ((const char*) &nodeData[next], _MM_HINT_T0);
      }
#endif

      GraphNode dst = edgeDst[i];
      func (dst);
    }
#endif // 


  }

  edge_data_reference getEdgeData(edge_iterator ni, MethodFlag mflag = MethodFlag::UNPROTECTED) {
    // Galois::Runtime::checkWrite(mflag, false);
    return edgeData[*ni];
  }

  GraphNode getEdgeDst(edge_iterator ni) {
    return edgeDst[*ni];
  }

  // For Prefetching hack (Amber);
  const GraphNode& getEdgeDst (edge_iterator ni) const {
    return edgeDst[*ni];
  }

  uint64_t size() const { return numNodes; }
  uint64_t sizeEdges() const { return numEdges; }

  iterator begin() const { return iterator(0); }
  iterator end() const { return iterator(numNodes); }

  const_local_iterator local_begin() const { return const_local_iterator(this->localBegin(numNodes)); }
  const_local_iterator local_end() const { return const_local_iterator(this->localEnd(numNodes)); }
  local_iterator local_begin() { return local_iterator(this->localBegin(numNodes)); }
  local_iterator local_end() { return local_iterator(this->localEnd(numNodes)); }

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

  detail::EdgesIterator<LC_CSR_MIC_Graph> out_edges(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    return detail::EdgesIterator<LC_CSR_MIC_Graph>(*this, N, mflag);
  }

  /**
   * Sorts outgoing edges of a node. Comparison function is over EdgeTy.
   */
  template<typename CompTy>
  void sortEdgesByEdgeData(GraphNode N, const CompTy& comp = std::less<EdgeTy>(), MethodFlag mflag = MethodFlag::WRITE) {
    acquireNode(N, mflag);
    std::sort(edge_sort_begin(N), edge_sort_end(N), detail::EdgeSortCompWrapper<EdgeSortValue<GraphNode,EdgeTy>,CompTy>(comp));
  }

  /**
   * Sorts outgoing edges of a node. Comparison function is over <code>EdgeSortValue<EdgeTy></code>.
   */
  template<typename CompTy>
  void sortEdges(GraphNode N, const CompTy& comp, MethodFlag mflag = MethodFlag::WRITE) {
    acquireNode(N, mflag);
    std::sort(edge_sort_begin(N), edge_sort_end(N), comp);
  }

  void allocateFrom(FileGraph& graph) {
    numNodes = graph.size();
    numEdges = graph.sizeEdges();
    if (UseNumaAlloc) {
      nodeData.allocateLocal(numNodes);
      edgeIndData.allocateLocal(numNodes + 1);
      edgeDst.allocateLocal(numEdges);
      edgeData.allocateLocal(numEdges);
      this->outOfLineAllocateLocal(numNodes);
    } else {
      nodeData.allocateInterleaved(numNodes);
      edgeIndData.allocateInterleaved(numNodes + 1);
      edgeDst.allocateInterleaved(numEdges);
      edgeData.allocateInterleaved(numEdges);
      this->outOfLineAllocateInterleaved(numNodes);
    }
  }

  void constructFrom(FileGraph& graph, unsigned tid, unsigned total) {
    auto r = graph.divideByNode(
        NodeData::size_of::value + EdgeIndData::size_of::value + LC_CSR_MIC_Graph::size_of_out_of_line::value,
        EdgeDst::size_of::value + EdgeData::size_of::value,
        tid, total).first;
    this->setLocalRange(*r.first, *r.second);
    edgeIndData[0] = 0;
    for (FileGraph::iterator ii = r.first, ei = r.second; ii != ei; ++ii) {
      nodeData.constructAt(*ii);
      assert (((*ii) + 1) < edgeIndData.size ());
      edgeIndData[(*ii) + 1] = *graph.edge_end(*ii);
      this->outOfLineConstructAt(*ii);
      for (FileGraph::edge_iterator nn = graph.edge_begin(*ii), en = graph.edge_end(*ii); nn != en; ++nn) {
        constructEdgeValue(graph, nn);
        edgeDst[*nn] = graph.getEdgeDst(nn);
      }
    }
  }
};

} // end namespace
} // end namespace

#undef _DO_INNER_PREFETCH
#undef _DO_OUTER_PREFETCH
#undef _DO_UNROLL

#endif // GALOIS_GRAPH_LC_CSR_MIC_GRAPH_H
