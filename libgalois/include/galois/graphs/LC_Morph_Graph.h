/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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
 * @file LC_Morph_Graph.h
 *
 * Contains the LC_Morph_Graph class.
 */

#ifndef GALOIS_GRAPHS_LC_MORPH_GRAPH_H
#define GALOIS_GRAPHS_LC_MORPH_GRAPH_H

#include <type_traits>

#include <boost/mpl/if.hpp>

#include "galois/Bag.h"
#include "galois/config.h"
#include "galois/LargeArray.h"
#include "galois/graphs/FileGraph.h"
#include "galois/graphs/Details.h"

namespace galois {
namespace graphs {

/**
 * Local computation graph that allows addition of nodes (but not removals)
 * if the maximum degree of a node is known at the time it is added.
 */
template <typename NodeTy, typename EdgeTy, bool HasNoLockable = false,
          bool UseNumaAlloc = false, bool HasOutOfLineLockable = false,
          bool HasId = false, typename FileEdgeTy = EdgeTy>
class LC_Morph_Graph
    : private boost::noncopyable,
      private internal::OutOfLineLockableFeature<HasOutOfLineLockable &&
                                                 !HasNoLockable> {
  //! Friend of LC_InOut_Graph (makes a graph have both in and out edges)
  template <typename Graph>
  friend class LC_InOut_Graph;

public:
  /**
   * Struct that allows activation of the HasId template parameter
   * Example: using Graph = LC_Morph_Graph::with_id<true> defines
   * LC_Morph_Graph with HasId = true
   */
  template <bool _has_id>
  struct with_id {
    using type = LC_Morph_Graph<NodeTy, EdgeTy, HasNoLockable, UseNumaAlloc,
                                HasOutOfLineLockable, _has_id, FileEdgeTy>;
  };

  /**
   * Struct used to define the type of node data through the
   * template parameter. See with_id doxygen for example.
   */
  template <typename _node_data>
  struct with_node_data {
    using type = LC_Morph_Graph<_node_data, EdgeTy, HasNoLockable, UseNumaAlloc,
                                HasOutOfLineLockable, HasId, FileEdgeTy>;
  };

  /**
   * Struct used to define the type of edge data through the
   * template parameter. See with_id doxygen for example.
   */
  template <typename _edge_data>
  struct with_edge_data {
    using type = LC_Morph_Graph<NodeTy, _edge_data, HasNoLockable, UseNumaAlloc,
                                HasOutOfLineLockable, HasId, FileEdgeTy>;
  };

  /**
   * Struct used to define the type of file edge data through the
   * template parameter. See with_id doxygen for example.
   */
  template <typename _file_edge_data>
  struct with_file_edge_data {
    using type = LC_Morph_Graph<NodeTy, EdgeTy, HasNoLockable, UseNumaAlloc,
                                HasOutOfLineLockable, HasId, _file_edge_data>;
  };

  /**
   * Struct used to define the HasNoLockable template parameter.
   * See with_id doxygen for example.
   */
  template <bool _has_no_lockable>
  struct with_no_lockable {
    using type = LC_Morph_Graph<NodeTy, EdgeTy, _has_no_lockable, UseNumaAlloc,
                                HasOutOfLineLockable, HasId, FileEdgeTy>;
  };

  /**
   * Struct used to define the UseNumaAlloc template parameter.
   * See with_id doxygen for example.
   */
  template <bool _use_numa_alloc>
  struct with_numa_alloc {
    using type = LC_Morph_Graph<NodeTy, EdgeTy, HasNoLockable, _use_numa_alloc,
                                HasOutOfLineLockable, HasId, FileEdgeTy>;
  };

  /**
   * Struct used to define the HasOutOfLineLockable template parameter.
   * See with_id doxygen for example.
   */
  template <bool _has_out_of_line_lockable>
  struct with_out_of_line_lockable {
    using type = LC_Morph_Graph<NodeTy, EdgeTy, HasNoLockable, UseNumaAlloc,
                                _has_out_of_line_lockable,
                                _has_out_of_line_lockable || HasId, FileEdgeTy>;
  };

  //! type that tells graph reader how to read a file for this graph
  using read_tag = read_with_aux_graph_tag;

protected:
  // Forward declaration of class (defined below)
  class NodeInfo;

  //! EdgeInfo keeps destination of edges
  using EdgeInfo = internal::EdgeInfoBase<NodeInfo*, EdgeTy>;
  //! Nodes are stored in an insert bag
  using Nodes = galois::InsertBag<NodeInfo>;
  //! Type of nodes
  using NodeInfoTypes =
      internal::NodeInfoBaseTypes<NodeTy,
                                  !HasNoLockable && !HasOutOfLineLockable>;

  //! Linked list structure holding together blocks of memory that stores
  //! edges.
  struct EdgeHolder {
    //! Beginning of memory for this block.
    EdgeInfo* begin;
    //! End of memory for this block.
    EdgeInfo* end;
    //! Pointer to another block of memory for edges (if it exists).
    EdgeHolder* next;
  };

  /**
   * Class that stores node info (e.g. where its edges begin and end, its data,
   * etc.).
   */
  class NodeInfo
      : public internal::NodeInfoBase<NodeTy,
                                      !HasNoLockable && !HasOutOfLineLockable> {
    using Super =
        internal::NodeInfoBase<NodeTy, !HasNoLockable && !HasOutOfLineLockable>;
    friend class LC_Morph_Graph;

    EdgeInfo* edgeBegin;
    EdgeInfo* edgeEnd;
#ifndef NDEBUG
    EdgeInfo* trueEdgeEnd;
#endif

  public:
    //! Calls NodeInfoBase constructor
    template <typename... Args>
    NodeInfo(Args&&... args) : Super(std::forward<Args>(args)...) {}
  }; // end NodeInfo

  //! Functor that returns pointers to NodeInfo objects given references
  struct makeGraphNode {
    //! Returns a pointer to the NodeInfo reference passed into this functor
    NodeInfo* operator()(NodeInfo& data) const { return &data; }
  };

  /**
   * Functor: contains an operator to compare the destination of an edge with
   * a particular node.
   */
  struct dst_equals {
    //! Destination to compare with
    NodeInfo* dst;
    //! Constructor: takes a node to compare edge destinations with
    dst_equals(NodeInfo* d) : dst(d) {}
    //! Given an edge, check if the edge destination matches the node that
    //! this functor was constructed with
    bool operator()(const EdgeInfo& edge) { return edge.dst == dst; }
  };

public:
  //! A graph node is a NodeInfo object.
  using GraphNode = NodeInfo*;
  //! Type of edge data in file
  using file_edge_data_type = FileEdgeTy;
  //! Type of edge data
  using edge_data_type = EdgeTy;
  //! Type of node data
  using node_data_type = NodeTy;
  //! Reference type to node data
  using node_data_reference = typename NodeInfoTypes::reference;
  //! Reference type to edge data
  using edge_data_reference = typename EdgeInfo::reference;
  //! Iterator over EdgeInfo objects (edges)
  using edge_iterator = EdgeInfo*;
  //! Iterator over nodes
  using iterator =
      boost::transform_iterator<makeGraphNode, typename Nodes::iterator>;
  //! Constant iterator over nodes
  using const_iterator =
      boost::transform_iterator<makeGraphNode, typename Nodes::const_iterator>;
  //! Local iterator is just an iterator
  using local_iterator = iterator;
  //! Const local iterator is just an const_iterator
  using const_local_iterator = const_iterator;
  //! @todo doxygen this
  using ReadGraphAuxData = LargeArray<GraphNode>;

protected:
  //! Nodes in this graph
  Nodes nodes;
  //! Memory for edges in this graph (memory held in EdgeHolders)
  galois::substrate::PerThreadStorage<EdgeHolder*> edgesL;

  /**
   * Acquire a node for the scope in which the function is called.
   *
   * @param N node to acquire
   * @param mflag Method flag specifying type of acquire (e.g. read, write,
   * etc.)
   */
  template <bool _A1 = HasNoLockable, bool _A2 = HasOutOfLineLockable>
  void acquireNode(GraphNode N, MethodFlag mflag,
                   typename std::enable_if<!_A1 && !_A2>::type* = 0) {
    galois::runtime::acquire(N, mflag);
  }

  /**
   * Acquire a node for the scope in which the function is called. The
   * lock is out of line (not local to the node).
   *
   * @param N node to acquire
   * @param mflag Method flag specifying type of acquire (e.g. read, write,
   * etc.)
   */
  template <bool _A1 = HasOutOfLineLockable, bool _A2 = HasNoLockable>
  void acquireNode(GraphNode N, MethodFlag mflag,
                   typename std::enable_if<_A1 && !_A2>::type* = 0) {
    this->outOfLineAcquire(getId(N), mflag);
  }

  /**
   * Given a FileGraph and an edge in it, add it to the LCMorphGraph.
   * Can handle edge weights.
   */
  template <bool _A1 = EdgeInfo::has_value,
            bool _A2 = LargeArray<FileEdgeTy>::has_value>
  void constructEdgeValue(FileGraph& graph,
                          typename FileGraph::edge_iterator nn, GraphNode src,
                          GraphNode dst,
                          typename std::enable_if<!_A1 || _A2>::type* = 0) {
    if (EdgeInfo::has_value) {
      // type of edge data in file graph
      using FEDV = typename LargeArray<FileEdgeTy>::value_type;
      // add an edge with edge data
      addMultiEdge(src, dst, galois::MethodFlag::UNPROTECTED,
                   graph.getEdgeData<FEDV>(nn));
    } else {
      // add an edge without edge data
      addMultiEdge(src, dst, galois::MethodFlag::UNPROTECTED);
    }
  }

  /**
   * Given a FileGraph and an edge in it, add it to the LCMorphGraph.
   * Does not handle edge weights.
   */
  template <bool _A1 = EdgeInfo::has_value,
            bool _A2 = LargeArray<FileEdgeTy>::has_value>
  void constructEdgeValue(FileGraph&, typename FileGraph::edge_iterator,
                          GraphNode src, GraphNode dst,
                          typename std::enable_if<_A1 && !_A2>::type* = 0) {
    addMultiEdge(src, dst, galois::MethodFlag::UNPROTECTED);
  }

  /**
   * No-op acquire node when HasNoLockable is true.
   */
  template <bool _A1 = HasOutOfLineLockable, bool _A2 = HasNoLockable>
  void acquireNode(GraphNode, MethodFlag,
                   typename std::enable_if<_A2>::type* = 0) {}

  /**
   * Get the ID of a graph node if they're enabled in the class.
   */
  template <bool _Enable = HasId>
  size_t getId(GraphNode N, typename std::enable_if<_Enable>::type* = 0) {
    return N->getId();
  }

public:
  /**
   * Destructor. If edges have some value, destory all of it (i.e. free up
   * memory).
   */
  ~LC_Morph_Graph() {
    for (typename Nodes::iterator ii = nodes.begin(), ei = nodes.end();
         ii != ei; ++ii) {
      NodeInfo& n         = *ii;
      EdgeInfo* edgeBegin = n.edgeBegin;
      EdgeInfo* edgeEnd   = n.edgeEnd;

      if (EdgeInfo::has_value) {
        while (edgeBegin != edgeEnd) {
          edgeBegin->destroy();
          ++edgeBegin;
        }
      }
    }
  }

  /**
   * Get the data of a node N.
   */
  node_data_reference getData(const GraphNode& N,
                              MethodFlag mflag = MethodFlag::WRITE) {
    // galois::runtime::checkWrite(mflag, false);
    acquireNode(N, mflag);
    return N->getData();
  }

  /**
   * Get edge data of an edge given an iterator to the edge.
   */
  edge_data_reference getEdgeData(edge_iterator ni,
                                  MethodFlag mflag = MethodFlag::UNPROTECTED) {
    // galois::runtime::checkWrite(mflag, false);
    acquireNode(ni->dst, mflag);
    return ni->get();
  }

  /**
   * Get the destination of an edge given an iterator to the edge.
   */
  GraphNode getEdgeDst(edge_iterator ni) {
    // galois::runtime::checkWrite(mflag, false);
    // acquireNode(ni->dst, mflag);
    return GraphNode(ni->dst);
  }

  /**
   * Returns an iterator to all the nodes in the graph. Not thread-safe.
   */
  iterator begin() {
    return boost::make_transform_iterator(nodes.begin(), makeGraphNode());
  }

  //! Returns the end of the node iterator. Not thread-safe.
  iterator end() {
    return boost::make_transform_iterator(nodes.end(), makeGraphNode());
  }

  //! Return an iterator to the beginning of the local nodes of the graph.
  local_iterator local_begin() {
    return boost::make_transform_iterator(nodes.local_begin(), makeGraphNode());
  }

  //! Return an iterator to the end of the local nodes of the graph.
  local_iterator local_end() {
    return boost::make_transform_iterator(nodes.local_end(), makeGraphNode());
  }

  /**
   * Return an iterator to the first edge of a particular node.
   */
  edge_iterator edge_begin(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    acquireNode(N, mflag);
    // Locks all destinations before returning edge iterator.
    if (galois::runtime::shouldLock(mflag)) {
      for (edge_iterator ii = N->edgeBegin, ee = N->edgeEnd; ii != ee; ++ii) {
        acquireNode(ii->dst, mflag);
      }
    }
    return N->edgeBegin;
  }

  /**
   * Return an iterator to the end of edges of a particular node.
   */
  edge_iterator edge_end(GraphNode N,
                         MethodFlag GALOIS_UNUSED(mflag) = MethodFlag::WRITE) {
    return N->edgeEnd;
  }

  /**
   * Return a range for edges of a node for use by C++ for_each loops.
   */
  runtime::iterable<NoDerefIterator<edge_iterator>>
  edges(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    return internal::make_no_deref_range(edge_begin(N, mflag),
                                         edge_end(N, mflag));
  }

  /**
   * Returns an object with begin() and end() methods to iterate over the
   * outgoing edges of N.
   */
  internal::EdgesIterator<LC_Morph_Graph>
  out_edges(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    return internal::EdgesIterator<LC_Morph_Graph>(*this, N, mflag);
  }

  /**
   * Creates a new node with a cap on the number of edges.
   *
   * @param nedges Number of edges reserved for this node.
   * @param args Arguments required to construct a new node
   * @returns Newly created node
   */
  template <typename... Args>
  GraphNode createNode(int nedges, Args&&... args) {
    NodeInfo* N = &nodes.emplace(std::forward<Args>(args)...);
    acquireNode(N, MethodFlag::WRITE);
    EdgeHolder*& local_edges = *edgesL.getLocal();

    // Allocate space for a new EdgeHolder object if necessary
    if (!local_edges ||
        std::distance(local_edges->begin, local_edges->end) < nedges) {
      EdgeHolder* old = local_edges;
      // FIXME: this seems to leak
      size_t size       = runtime::pagePoolSize();
      void* block       = runtime::pagePoolAlloc();
      local_edges       = reinterpret_cast<EdgeHolder*>(block);
      local_edges->next = old;

      size -= sizeof(EdgeHolder);
      block = reinterpret_cast<char*>(block) + sizeof(EdgeHolder);

      if (!std::align(std::alignment_of_v<EdgeInfo>, sizeof(EdgeInfo), block,
                      size)) {
        GALOIS_DIE("not enough space for EdgeInfo");
      }

      local_edges->begin = reinterpret_cast<EdgeInfo*>(block);
      local_edges->end   = local_edges->begin;
      local_edges->end += size / sizeof(EdgeInfo);
      if (std::distance(local_edges->begin, local_edges->end) < nedges) {
        GALOIS_DIE("not enough space for EdgeInfo");
      }
    }

    // Set the memory aside for the new node in the edge holder object
    N->edgeBegin = N->edgeEnd = local_edges->begin;
    local_edges->begin += nedges;
#ifndef NDEBUG
    N->trueEdgeEnd = local_edges->begin;
#endif
    return GraphNode(N);
  }

  /**
   * Adds an edge if it doesn't already exist.
   *
   * @param src Source to add edge to
   * @param dst Destination to add edge to
   * @param mflag Method flag specifying type of acquire (e.g. read, write)
   * @param args Arguments needed to construct an edge
   */
  template <typename... Args>
  edge_iterator addEdge(GraphNode src, GraphNode dst, galois::MethodFlag mflag,
                        Args&&... args) {
    // galois::runtime::checkWrite(mflag, true);
    acquireNode(src, mflag);
    auto it = std::find_if(src->edgeBegin, src->edgeEnd, dst_equals(dst));
    if (it == src->edgeEnd) {
      it->dst = dst;
      it->construct(std::forward<Args>(args)...);
      src->edgeEnd++;
      assert(src->edgeEnd <= src->trueEdgeEnd);
    }
    return it;
  }

  /**
   * Construct a new edge for a node. Can add duplicate edges.
   *
   * @param src Source node to add edge to
   * @param dst Destination node of new edge
   * @param mflag Method flag specifying type of acquire (e.g. read, write)
   * @param args Other arguments that need to be passed in to construct
   * a new edge
   * @returns Iterator to newly added edge
   */
  template <typename... Args>
  edge_iterator addMultiEdge(GraphNode src, GraphNode dst,
                             galois::MethodFlag mflag, Args&&... args) {
    acquireNode(src, mflag);
    auto it = src->edgeEnd;
    it->dst = dst;
    it->construct(std::forward<Args>(args)...);
    src->edgeEnd++;
    assert(src->edgeEnd <= src->trueEdgeEnd);
    return it;
  }

  /**
   * Remove an edge from the graph.
   *
   * Invalidates edge iterator.
   */
  void removeEdge(GraphNode src, edge_iterator dst,
                  galois::MethodFlag mflag = MethodFlag::WRITE) {
    // galois::runtime::checkWrite(mflag, true);
    acquireNode(src, mflag);
    src->edgeEnd--;
    assert(src->edgeBegin <= src->edgeEnd);
    std::swap(*dst, *src->edgeEnd);
    src->edgeEnd->destroy();
  }

  /**
   * Finds an edge between 2 nodes and returns the iterator to it if it exists.
   */
  edge_iterator findEdge(GraphNode src, GraphNode dst,
                         galois::MethodFlag mflag = MethodFlag::WRITE) {
    // galois::runtime::checkWrite(mflag, true); // TODO: double check 'true'
    // here
    acquireNode(src, mflag);
    return std::find_if(src->edgeBegin, src->edgeEnd, dst_equals(dst));
  }

  /**
   * Allocate memory for nodes given a file graph with a particular number of
   * nodes. This graph will allocate out of line space for that number of
   * nodes as well.
   *
   * @param graph FileGraph with a number of nodes to allocate
   * @param aux Data structure in which to allocate space for nodes.
   */
  void allocateFrom(FileGraph& graph, ReadGraphAuxData& aux) {
    size_t numNodes = graph.size();

    if (UseNumaAlloc) {
      aux.allocateLocal(numNodes);
      this->outOfLineAllocateLocal(numNodes);
    } else {
      aux.allocateInterleaved(numNodes);
      this->outOfLineAllocateInterleaved(numNodes);
    }
  }

  /**
   * Constructs the LCMorphGraph nodes given a FileGraph to construct it from.
   * Meant to be called by multiple threads.
   *
   * @param[in] graph FileGraph to construct a morph graph from
   * @param[in] tid Thread id of thread calling this function
   * @param[in] total Total number of threads in current execution
   * @param[in,out] aux Allocated memory to store pointers to the created nodes
   */
  void constructNodesFrom(FileGraph& graph, unsigned tid, unsigned total,
                          ReadGraphAuxData& aux) {
    // get the portion of the graph that this thread is responsible for
    // creating
    auto r = graph
                 .divideByNode(sizeof(NodeInfo) +
                                   LC_Morph_Graph::size_of_out_of_line::value,
                               sizeof(EdgeInfo), tid, total)
                 .first;

    // create nodes of portion we are responsible for only
    for (FileGraph::iterator ii = r.first, ei = r.second; ii != ei; ++ii) {
      aux[*ii] =
          createNode(std::distance(graph.edge_begin(*ii), graph.edge_end(*ii)));
    }
  }

  /**
   * Constructs the LCMorphGraph edges given a FileGraph to construct it from
   * and pointers to already created nodes. Meant to be called by multiple
   * threads.
   *
   * @param[in] graph FileGraph to construct a morph graph from
   * @param[in] tid Thread id of thread calling this function
   * @param[in] total Total number of threads in current execution
   * @param[in] aux Contains pointers to already created nodes to
   * create edges for.
   */
  void constructEdgesFrom(FileGraph& graph, unsigned tid, unsigned total,
                          const ReadGraphAuxData& aux) {
    auto r = graph
                 .divideByNode(sizeof(NodeInfo) +
                                   LC_Morph_Graph::size_of_out_of_line::value,
                               sizeof(EdgeInfo), tid, total)
                 .first;

    for (FileGraph::iterator ii = r.first, ei = r.second; ii != ei; ++ii) {
      for (FileGraph::edge_iterator nn = graph.edge_begin(*ii),
                                    en = graph.edge_end(*ii);
           nn != en; ++nn) {
        constructEdgeValue(graph, nn, aux[*ii], aux[graph.getEdgeDst(nn)]);
      }
    }
  }
};

} // namespace graphs
} // namespace galois

#endif /* LC_MORPH_GRAPH_H_ */
