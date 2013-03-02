#ifndef GALOIS_DOMAINSPECIFICEXECUTORS_H
#define GALOIS_DOMAINSPECIFICEXECUTORS_H

#include "Galois/Galois.h"
#include "Galois/Accumulator.h"
#include "Galois/Bag.h"

namespace Galois {

template<typename Graph>
class GraphNodeBag {
  typedef typename Graph::GraphNode GNode;
  typedef Galois::InsertBag<GNode> Bag;

  Graph& graph;
  Bag bag;
  Galois::GAccumulator<size_t> counts;
  Galois::LargeArray<bool,false> bitmask;

  struct Densify {
    GraphNodeBag<Graph>* self;
    Densify(GraphNodeBag<Graph>* s): self(s) { }
    void operator()(GNode n) {
      self->bitmask[n] = true;
    }
  };

public:
  GraphNodeBag(Graph& g): graph(g) { 
    bitmask.allocate(g.size());
  }

  typedef typename Bag::iterator iterator;
  typedef typename Bag::local_iterator local_iterator;

  iterator begin() { return bag.begin(); }
  iterator end() { return bag.end(); }
  local_iterator local_begin() { return bag.local_begin(); }
  local_iterator local_end() { return bag.local_end(); }

  void push(GNode n) {
    bag.push(n);
    counts += 1 + std::distance(graph.edge_begin(n, Galois::MethodFlag::NONE), graph.edge_end(n, Galois::MethodFlag::NONE));
  }

  size_t getCount() { return counts.reduce(); }

  void clear() { 
    bag.clear();
    counts.reset(); 
    memset(bitmask.data(), 0, sizeof(bitmask[0]) * graph.size());
  }

  bool contains(GNode n) {
    return bitmask[n];
  }

  bool empty() const { return bag.empty(); }

  void densify() {
    Galois::do_all_local(bag, Densify(this));
  }
};

template<typename Graph>
class GraphNodeBagPair {
  GraphNodeBag<Graph> bag1;
  GraphNodeBag<Graph> bag2;
  int curp;
public:
  GraphNodeBagPair(Graph& g): bag1(g), bag2(g), curp(0) { }

  GraphNodeBag<Graph>& cur() { return (*this)[curp]; }
  GraphNodeBag<Graph>& next() { return (*this)[(curp+1) & 1]; }
  void swap() { curp = (curp + 1) & 1; }
  GraphNodeBag<Graph>& operator[](int i) {
    if (i == 0)
      return bag1;
    else
      return bag2;
  }
};

template<typename Graph, typename EdgeOperator>
struct DenseOperator {
  typedef int tt_does_not_need_aborts;
  typedef int tt_does_not_need_push;
  typedef typename Graph::GraphNode GNode;

  Graph& graph;
  GraphNodeBag<Graph>& input;
  GraphNodeBag<Graph>& output;
  EdgeOperator op;
  
  DenseOperator(Graph& g, GraphNodeBag<Graph>& i, GraphNodeBag<Graph>& o, EdgeOperator op): 
    graph(g), input(i), output(o), op(op) { }

  void operator()(GNode n, Galois::UserContext<GNode>&) {
    (*this)(n);
  }

  void operator()(GNode n) {
    if (!op.cond(graph, n))
      return;

    for (typename Graph::in_edge_iterator ii = graph.in_edge_begin(n, Galois::MethodFlag::NONE),
          ei = graph.in_edge_end(n, Galois::MethodFlag::NONE); ii != ei; ++ii) {
      GNode src = graph.getInEdgeDst(ii);
        
      if (input.contains(src) && op(graph, src, n, graph.getInEdgeData(ii))) {
        output.push(n);
      }
      if (!op.cond(graph, n))
        return;
    }
  }
};

template<typename Graph, typename EdgeOperator>
struct SparseOperator { 
  typedef int tt_does_not_need_aborts;
  typedef int tt_does_not_need_push;
  typedef typename Graph::GraphNode GNode;

  Graph& graph;
  GraphNodeBag<Graph>& output;
  EdgeOperator op;
  GNode source;
  
  SparseOperator(Graph& g, GraphNodeBag<Graph>& o, EdgeOperator op, GNode s = GNode()): graph(g), output(o), op(op), source(s) { }

  void operator()(GNode n, Galois::UserContext<GNode>&) {
    (*this)(n);
  }

  void operator()(GNode n) {
    for (typename Graph::edge_iterator ii = graph.edge_begin(n, Galois::MethodFlag::NONE),
          ei = graph.edge_end(n, Galois::MethodFlag::NONE); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);

      if (op.cond(graph, dst) && op(graph, n, dst, graph.getEdgeData(ii))) {
        output.push(dst);
      }
    }
  }

  void operator()(typename Graph::edge_iterator ii, Galois::UserContext<typename Graph::edge_iterator>&) {
    (*this)(ii);
  }

  void operator()(typename Graph::edge_iterator ii) {
    GNode dst = graph.getEdgeDst(ii);

    if (op.cond(graph, dst) && op(graph, source, dst, graph.getEdgeData(ii))) {
      output.push(dst);
    }
  }
};

template<typename Graph, typename EdgeOperator>
void edgeMap(Graph& graph, EdgeOperator op, typename Graph::GraphNode single, GraphNodeBag<Graph>& output) {
  Galois::for_each(graph.out_edges(single, Galois::MethodFlag::NONE).begin(),
      graph.out_edges(single, Galois::MethodFlag::NONE).end(),
      SparseOperator<Graph,EdgeOperator>(graph, output, op, single));
}

template<typename Graph, typename EdgeOperator>
void edgeMap(Graph& graph, EdgeOperator op, GraphNodeBag<Graph>& input, GraphNodeBag<Graph>& output) {
  using namespace Galois::WorkList;
  typedef dChunkedLIFO<256> WL;
  size_t count = input.getCount();

  if (count > graph.sizeEdges() / 20) {
    std::cout << "Dense " << count << "\n";
    input.densify();
    Galois::do_all_local(graph, DenseOperator<Graph,EdgeOperator>(graph, input, output, op));
  } else {
    std::cout << "Sparse " << count << "\n";
    Galois::for_each_local<WL>(input, SparseOperator<Graph,EdgeOperator>(graph, output, op));
  }

  input.clear();
}

}
#endif
