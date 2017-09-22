#ifndef GALOIS_GRAPHCHIEXECUTOR_H
#define GALOIS_GRAPHCHIEXECUTOR_H

#include "galois/graphs/OCGraph.h"
#include "galois/graphs/GraphNodeBag.h"

#include <boost/iterator/filter_iterator.hpp>
#include <boost/utility.hpp>

namespace galois {
//! Implementation of GraphChi DSL in Galois
namespace graphChi {

namespace internal {

template<bool PassWrappedGraph>
struct DispatchOperator {
  template<typename O,typename G,typename N>
  void run(O&& o, G&& g, N&& n) {
    std::forward<O>(o)(std::forward<G>(g), std::forward<N>(n));
  }
};

template<>
struct DispatchOperator<false> {
  template<typename O,typename G,typename N>
  void run(O&& o, G&& g, N&& n) {
    std::forward<O>(o)(std::forward<N>(n));
  }
};

template<bool PassWrappedGraph,typename Graph,typename WrappedGraph,typename VertexOperator>
class SparseVertexMap: public DispatchOperator<PassWrappedGraph> {
  typedef typename Graph::segment_type segment_type;
  typedef typename Graph::GraphNode GNode;

  Graph& graph;
  WrappedGraph& wrappedGraph;
  VertexOperator op;
  int& first;
  segment_type& prev;
  segment_type& cur;
  segment_type& next;
  bool updated;

public:
  typedef int tt_does_not_need_push;
  typedef int tt_does_not_need_aborts;

  SparseVertexMap(Graph& g, WrappedGraph& w, VertexOperator op, int& f,
      segment_type& p, segment_type& c, segment_type& n):
    graph(g), wrappedGraph(w), op(op), first(f), prev(p), cur(c), next(n), updated(false) { }

  void operator()(size_t n, galois::UserContext<size_t>&) {
    (*this)(n);
  }

  void operator()(size_t n) {
    if (!updated) {
      if (first == 0 && __sync_bool_compare_and_swap(&first, 0, 1)) {
        if (prev.loaded()) {
          graph.unload(prev);
        }
        if (next) {
          graph.load(next);
        }
      }
      updated = true;
    }
    // Check if range
    if (!cur.containsNode(n)) {
      return;
    }

    this->run(op, wrappedGraph, graph.nodeFromId(n));
  }
};

template<bool CheckInput,bool PassWrappedGraph,typename Graph,typename WrappedGraph,typename VertexOperator,typename Bag>
class DenseVertexMap: public DispatchOperator<PassWrappedGraph> {
  typedef typename Graph::segment_type segment_type;
  typedef typename Graph::GraphNode GNode;

  Graph& graph;
  WrappedGraph& wrappedGraph;
  VertexOperator op;
  Bag* bag;
  int& first;
  segment_type& prev;
  segment_type& cur;
  segment_type& next;
  bool updated;

public:
  typedef int tt_does_not_need_push;
  typedef int tt_does_not_need_aborts;

  DenseVertexMap(Graph& g, WrappedGraph& w, VertexOperator op, Bag* b, int& f,
      segment_type& p, segment_type& c, segment_type& n):
    graph(g), wrappedGraph(w), op(op), bag(b), first(f), prev(p), cur(c), next(n), updated(false) { }

  void operator()(GNode n, galois::UserContext<GNode>&) {
    (*this)(n);
  }

  void operator()(GNode n) {
    if (!updated) {
      if (first == 0 && __sync_bool_compare_and_swap(&first, 0, 1)) {
        if (prev.loaded()) {
          graph.unload(prev);
        }
        if (next) {
          graph.load(next);
        }
      }
      updated = true;
    }
    if (CheckInput && !bag->contains(graph.idFromNode(n)))
      return;

    this->run(op, wrappedGraph, n);
  }
};

template<typename Graph,typename Bag>
struct contains_node {
  Graph* graph;
  Bag* bag;
  contains_node(Graph* g, Bag* b): graph(g), bag(b) { }
  bool operator()(typename Graph::GraphNode n) {
    return bag->contains(graph->idFromNode(n));
  }
};

template<typename EdgeTy>
struct sizeof_edge {
  static const unsigned int value = sizeof(EdgeTy);
};

template<>
struct sizeof_edge<void> {
  static const unsigned int value = 0;
};

struct logical_or {
  bool operator()(bool a, bool b) const { return a || b; }
};

template<typename Graph,typename Seg,typename Bag>
bool any_in_range(Graph& graph, const Seg& cur, Bag* input) {
  return std::find_if(graph.begin(cur), graph.end(cur), contains_node<Graph,Bag>(&graph, input)) != graph.end(cur);
  // TODO(ddn): Figure out the memory leak in ParallelSTL::find_if
  //return galois::ParallelSTL::find_if(graph.begin(cur), graph.end(cur), contains_node<Graph>(&graph, input)) != graph.end(cur);
  //return galois::ParallelSTL::map_reduce(graph.begin(cur), graph.end(cur), contains_node<Graph,Bag>(&graph, input), false, logical_or());
}

template<typename Graph>
size_t computeEdgeLimit(Graph& graph, size_t memoryLimit) {
  // Convert memoryLimit which is in MB into edges
  size_t bytes = memoryLimit;
  bytes *= 1024 * 1024;
  size_t sizeNodes = graph.size() * sizeof(uint64_t);
  if (bytes < sizeNodes) {
    GALOIS_DIE("Cannot limit graph in memory allotted");
  }
  bytes -= sizeNodes;
  // double-buffering (2), in and out edges (2)
  size_t edgeBytes = 2 * 2 * (sizeof(uint64_t) + sizeof_edge<typename Graph::edge_data_type>::value);
  size_t edges = bytes / edgeBytes;

  return edges;
}

template<typename Graph>
bool fitsInMemory(Graph& graph, size_t memoryLimit) {
  size_t bytes = memoryLimit;
  bytes *= 1024 * 1024;
  size_t nodeBytes = graph.size() * sizeof(uint64_t);
  size_t edgeBytes = graph.sizeEdges() * 2 * (sizeof(uint64_t) + sizeof_edge<typename Graph::edge_data_type>::value);

  return nodeBytes + edgeBytes < bytes;
}

template<bool CheckInput, bool PassWrappedGraph, typename Graph, typename WrappedGraph, typename VertexOperator, typename Bag>
void vertexMap(Graph& graph, WrappedGraph& wgraph, VertexOperator op, Bag* input, size_t memoryLimit) {
  typedef typename Graph::segment_type segment_type;
  galois::Statistic rounds("GraphChiRounds");
  
  size_t edges = computeEdgeLimit(graph, memoryLimit);
  segment_type prev;
  segment_type cur = graph.nextSegment(edges);

  bool useDense;
  if (!CheckInput) {
    useDense = true;
  } else {
    // TODO improve this heuristic
    bool useSparse = (cur.size() > graph.size() / 2) && (input->getSize() < graph.size() / 4);
    useDense = !useSparse;
  }

  if (useDense && CheckInput) {
    input->densify();
  }

  while (cur) {
    if (!CheckInput || !useDense || any_in_range(graph, cur, input)) {
      if (!cur.loaded()) {
        graph.load(cur);
      }

      segment_type next = graph.nextSegment(cur, edges);

      int first = 0;
      wgraph.setSegment(cur);

      if (useDense) {
        DenseVertexMap<CheckInput,PassWrappedGraph,Graph,WrappedGraph,VertexOperator,Bag> vop(graph, wgraph, op, input, first, prev, cur, next);
        galois::for_each(graph.begin(cur), graph.end(cur), vop);
      } else {
        SparseVertexMap<PassWrappedGraph,Graph,WrappedGraph,VertexOperator> vop(graph, wgraph, op, first, prev, cur, next);
        galois::for_each_local(*input, vop);
      }

      // XXX Shouldn't be necessary
      if (prev.loaded()) {
        abort();
        graph.unload(prev);
      }
      
      rounds += 1;

      prev = cur;
      cur = next;
    } else {
      segment_type next = graph.nextSegment(cur, edges);
      if (prev.loaded())
        graph.unload(prev);
      if (cur.loaded())
        graph.unload(cur);
      cur = next;
    }
  }

  if (prev.loaded())
    graph.unload(prev);
}
} // end namespace


template<typename Graph, typename VertexOperator>
void vertexMap(Graph& graph, VertexOperator op, size_t size) {
  galois::graphs::BindSegmentGraph<Graph> wgraph(graph);
  
  internal::vertexMap<false,true>(graph, wgraph, op, static_cast<GraphNodeBag<>*>(0), size);
}

template<typename Graph, typename VertexOperator, typename Bag>
void vertexMap(Graph& graph, VertexOperator op, Bag& input, size_t size) {
  galois::graphs::BindSegmentGraph<Graph> wgraph(graph);
  
  internal::vertexMap<true,true>(graph, wgraph, op, &input, size);
}

} // end namespace
} // end namespace

#endif
