#ifndef GRAPH_COLORING_LOCKING_H
#define GRAPH_COLORING_LOCKING_H

#include "GraphColoringBase.h"

#include "Galois/WorkList/ExternalReference.h"

struct NodeData {
  unsigned color;
  unsigned priority;
  unsigned id;

  explicit NodeData (unsigned _id=0): color (0), priority (0), id(_id) {}
};

typedef Galois::Graph::LC_CSR_Graph<NodeData, void>::with_numa_alloc<true>::type Graph;
typedef Graph::GraphNode GNode; 

class GraphColoringLocking: public GraphColoringBase<Graph> {
protected:

  struct ColorNodeLocking {
    typedef int tt_does_not_need_push;

    GraphColoringLocking& outer;

    template <typename C>
    void operator () (GNode src, C&) {

      Graph& graph = outer.graph;
      NodeData& sd = graph.getData (src, Galois::MethodFlag::WRITE);
      for (Graph::edge_iterator e = graph.edge_begin (src, Galois::MethodFlag::WRITE),
          e_end = graph.edge_end (src, Galois::MethodFlag::WRITE); e != e_end; ++e) {
        GNode dst = graph.getEdgeDst (e);
      }

      outer.colorNode (src);

    }
  };

  typedef Galois::WorkList::AltChunkedFIFO<DEFAULT_CHUNK_SIZE> WL_ty;
  // typedef Galois::WorkList::dChunkedFIFO<DEFAULT_CHUNK_SIZE> WL_ty;
  void firstFit (void) {

    Galois::for_each_local (
        graph, ColorNodeLocking {*this},
        Galois::loopname ("first-fit"),
        Galois::wl<WL_ty> ());

  }

  void priorityScheduling (void) {
    typedef std::unary_function<GNode, unsigned> Base_ty;
    struct GetPriority: public Base_ty {
      Graph& graph;

      GetPriority (Graph& g): Base_ty (), graph (g) {}

      unsigned operator () (GNode n) {
        auto& nd = graph.getData (n, Galois::MethodFlag::UNPROTECTED);
        return nd.priority;
      }
    };

    typedef Galois::WorkList::OrderedByIntegerMetric<GetPriority, WL_ty> OBIM;

    Galois::for_each_local(graph, 
        ColorNodeLocking {*this}, Galois::loopname("color-obim"), 
        Galois::wl<OBIM>(GetPriority {graph}));
  }

  virtual void colorGraph (void) {
    if (heuristic == FIRST_FIT) {
      firstFit ();

    } else {
      assignPriority ();
      priorityScheduling ();

    }
  }

};


#endif // GRAPH_COLORING_LOCKING_H
