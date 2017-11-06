#ifndef GALOIS_LIGRAEXECUTOR_H
#define GALOIS_LIGRAEXECUTOR_H

#include "galois/Galois.h"

namespace galois {
//! Implementation of Ligra DSL in Galois
namespace Ligra {

namespace internal {
template<typename Graph,bool Forward>
struct Transposer {
  typedef typename Graph::GraphNode GNode;
  typedef typename Graph::in_edge_iterator in_edge_iterator;
  typedef typename Graph::edge_iterator edge_iterator;
  typedef typename Graph::edge_data_reference edge_data_reference;

  GNode getInEdgeDst(Graph& g, in_edge_iterator ii) {
    return g.getInEdgeDst(ii);
  }

  in_edge_iterator in_edge_begin(Graph& g, GNode n) {
    return g.in_edge_begin(n, galois::MethodFlag::UNPROTECTED);
  }

  in_edge_iterator in_edge_end(Graph& g, GNode n) {
    return g.in_edge_end(n, galois::MethodFlag::UNPROTECTED);
  }

  edge_data_reference getInEdgeData(Graph& g, in_edge_iterator ii) {
    return g.getInEdgeData(ii);
  }

  GNode getEdgeDst(Graph& g, edge_iterator ii) {
    return g.getEdgeDst(ii);
  }

  edge_iterator edge_begin(Graph& g, GNode n) {
    return g.edge_begin(n, galois::MethodFlag::UNPROTECTED);
  }

  edge_iterator edge_end(Graph& g, GNode n) {
    return g.edge_end(n, galois::MethodFlag::UNPROTECTED);
  }

  edge_data_reference getEdgeData(Graph& g, edge_iterator ii) {
    return g.getEdgeData(ii);
  }
};

template<typename Graph>
struct Transposer<Graph,false> {
  typedef typename Graph::GraphNode GNode;
  typedef typename Graph::edge_iterator in_edge_iterator;
  typedef typename Graph::in_edge_iterator edge_iterator;
  typedef typename Graph::edge_data_reference edge_data_reference;

  GNode getInEdgeDst(Graph& g, in_edge_iterator ii) {
    return g.getEdgeDst(ii);
  }

  in_edge_iterator in_edge_begin(Graph& g, GNode n) {
    return g.edge_begin(n, galois::MethodFlag::UNPROTECTED);
  }

  in_edge_iterator in_edge_end(Graph& g, GNode n) {
    return g.edge_end(n, galois::MethodFlag::UNPROTECTED);
  }

  edge_data_reference getInEdgeData(Graph& g, in_edge_iterator ii) {
    return g.getEdgeData(ii);
  }

  GNode getEdgeDst(Graph& g, edge_iterator ii) {
    return g.getInEdgeDst(ii);
  }

  edge_iterator edge_begin(Graph& g, GNode n) {
    return g.in_edge_begin(n, galois::MethodFlag::UNPROTECTED);
  }

  edge_iterator edge_end(Graph& g, GNode n) {
    return g.in_edge_end(n, galois::MethodFlag::UNPROTECTED);
  }

  edge_data_reference getEdgeData(Graph& g, edge_iterator ii) {
    return g.getInEdgeData(ii);
  }
};

template<typename Graph,typename Bag,typename EdgeOperator,bool Forward>
struct DenseOperator: public Transposer<Graph,Forward> {
  typedef Transposer<Graph,Forward> Super;
  typedef typename Super::GNode GNode;
  typedef typename Super::in_edge_iterator in_edge_iterator;
  typedef typename Super::edge_iterator edge_iterator;

  typedef int tt_does_not_need_aborts;
  typedef int tt_does_not_need_push;

  Graph& graph;
  Bag& input;
  Bag& output;
  EdgeOperator op;
  
  DenseOperator(Graph& g, Bag& i, Bag& o, EdgeOperator op): 
    graph(g), input(i), output(o), op(op) { }

  void operator()(GNode n, galois::UserContext<GNode>&) {
    (*this)(n);
  }

  void operator()(GNode n) {
    if (!op.cond(graph, n))
      return;

    for (in_edge_iterator ii = this->in_edge_begin(graph, n), ei = this->in_edge_end(graph, n); ii != ei; ++ii) {
      GNode src = this->getInEdgeDst(graph, ii);
        
      if (input.contains(graph.idFromNode(src)) && op(graph, src, n, this->getInEdgeData(graph, ii))) {
        output.push(graph.idFromNode(n), std::distance(this->edge_begin(graph, n), this->edge_end(graph, n)));
      }
      if (!op.cond(graph, n))
        return;
    }
  }
};

template<typename Graph,typename Bag,typename EdgeOperator,bool Forward,bool IgnoreInput>
struct DenseForwardOperator: public Transposer<Graph,Forward> {
  typedef Transposer<Graph,Forward> Super;
  typedef typename Super::GNode GNode;
  typedef typename Super::in_edge_iterator in_edge_iterator;
  typedef typename Super::edge_iterator edge_iterator;

  typedef int tt_does_not_need_aborts;
  typedef int tt_does_not_need_push;

  Graph& graph;
  Bag& input;
  Bag& output;
  EdgeOperator op;
  
  DenseForwardOperator(Graph& g, Bag& i, Bag& o, EdgeOperator op): 
    graph(g), input(i), output(o), op(op) { }

  void operator()(GNode n, galois::UserContext<GNode>&) {
    (*this)(n);
  }

  void operator()(GNode n) {
    if (!IgnoreInput && !input.contains(graph.idFromNode(n)))
      return;

    for (edge_iterator ii = this->edge_begin(graph, n), ei = this->edge_end(graph, n); ii != ei; ++ii) {
      GNode dst = this->getEdgeDst(graph, ii);
        
      if (op.cond(graph, n) && op(graph, n, dst, this->getEdgeData(graph, ii))) {
        output.pushDense(graph.idFromNode(dst), std::distance(this->edge_begin(graph, dst), this->edge_end(graph, dst)));
      }
    }
  }
};

template<typename Graph,typename Bag,typename EdgeOperator,bool Forward>
struct SparseOperator: public Transposer<Graph,Forward> { 
  typedef Transposer<Graph,Forward> Super;
  typedef typename Super::GNode GNode;
  typedef typename Super::in_edge_iterator in_edge_iterator;
  typedef typename Super::edge_iterator edge_iterator;

  typedef int tt_does_not_need_aborts;
  typedef int tt_does_not_need_push;

  Graph& graph;
  Bag& output;
  EdgeOperator op;
  GNode source;
  
  SparseOperator(Graph& g, Bag& o, EdgeOperator op, GNode s = GNode()):
    graph(g), output(o), op(op), source(s) { }

  void operator()(size_t n, galois::UserContext<size_t>&) {
    (*this)(n);
  }

  void operator()(size_t id) {
    GNode n = graph.nodeFromId(id);

    for (edge_iterator ii = this->edge_begin(graph, n), ei = this->edge_end(graph, n); ii != ei; ++ii) {
      GNode dst = this->getEdgeDst(graph, ii);

      if (op.cond(graph, dst) && op(graph, n, dst, this->getEdgeData(graph, ii))) {
        output.push(graph.idFromNode(dst), std::distance(this->edge_begin(graph, dst), this->edge_end(graph, dst)));
      }
    }
  }

  void operator()(edge_iterator ii, galois::UserContext<edge_iterator>&) {
    (*this)(ii);
  }

  void operator()(edge_iterator ii) {
    GNode dst = this->getEdgeDst(graph, ii);

    if (op.cond(graph, dst) && op(graph, source, dst, this->getEdgeData(graph, ii))) {
      output.push(graph.idFromNode(dst), std::distance(this->edge_begin(graph, dst), this->edge_end(graph, dst)));
    }
  }
};
} // end namespace

template<bool Forward,typename Graph,typename EdgeOperator,typename Bag>
void edgeMap(Graph& graph, EdgeOperator op, Bag& output) {
  output.densify();
  galois::for_each(galois::iterate(graph), internal::DenseForwardOperator<Graph,Bag,EdgeOperator,Forward,true>(graph, output, output, op));
}

template<bool Forward,typename Graph,typename EdgeOperator,typename Bag>
void edgeMap(Graph& graph, EdgeOperator op, typename Graph::GraphNode single, Bag& output) {
  if (Forward) {
    galois::for_each(
        galois::iterate(graph.out_edges(single, galois::MethodFlag::UNPROTECTED).begin(),
        graph.out_edges(single, galois::MethodFlag::UNPROTECTED).end()),

        internal::SparseOperator<Graph,Bag,EdgeOperator,true>(graph, output, op, single));
  } else {
    galois::for_each(
        galois::iterate(graph.in_edges(single, galois::MethodFlag::UNPROTECTED).begin(),
        graph.in_edges(single, galois::MethodFlag::UNPROTECTED).end()),

        internal::SparseOperator<Graph,Bag,EdgeOperator,false>(graph, output, op, single));
  }
}

template<bool Forward,typename Graph,typename EdgeOperator,typename Bag>
void edgeMap(Graph& graph, EdgeOperator op, Bag& input, Bag& output, bool denseForward) {
  using namespace galois::worklists;
  size_t count = input.getCount();

  if (!denseForward && count > graph.sizeEdges() / 20) {
    //std::cout << "(D) Count " << count << "\n"; // XXX
    input.densify();
    if (denseForward) {
      abort(); // Never executed
      output.densify();
      // typedef dChunkedFIFO<256*4> WL;
      galois::do_all(galois::iterate(graph), internal::DenseForwardOperator<Graph,Bag,EdgeOperator,Forward,false>(graph, input, output, op), galois::chunk_size<256>());
    } else {
      // typedef dChunkedFIFO<256> WL;
      galois::do_all(galois::iterate(graph), internal::DenseOperator<Graph,Bag,EdgeOperator,Forward>(graph, input, output, op), galois::chunk_size<256>());
    }
  } else {
    //std::cout << "(S) Count " << count << "\n"; // XXX
    // typedef dChunkedFIFO<64> WL;
    galois::do_all(galois::iterate(input), internal::SparseOperator<Graph,Bag,EdgeOperator,Forward>(graph, output, op), galois::chunk_size<256>());
  }
}

template<typename... Args>
void outEdgeMap(Args&&... args) {
  edgeMap<true>(std::forward<Args>(args)...);
}

template<typename... Args>
void inEdgeMap(Args&&... args) {
  edgeMap<false>(std::forward<Args>(args)...);
}

} // end namespace
} // end namespace

#endif
