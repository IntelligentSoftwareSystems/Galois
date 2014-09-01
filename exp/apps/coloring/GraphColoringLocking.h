#ifndef GRAPH_COLORING_LOCKING_H
#define GRAPH_COLORING_LOCKING_H

#include "GraphColoringBase.h"

#include "Galois/GaloisUnsafe.h"

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
      NodeData& sd = graph.getData (src, Galois::CHECK_CONFLICT);
      for (Graph::edge_iterator e = graph.edge_begin (src, Galois::CHECK_CONFLICT),
          e_end = graph.edge_end (src, Galois::CHECK_CONFLICT); e != e_end; ++e) {
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
        auto& nd = graph.getData (n, Galois::NONE);
        return nd.priority;
      }
    };

    typedef Galois::WorkList::OrderedByIntegerMetric<GetPriority, WL_ty> OBIM;

    OBIM wl {GetPriority {graph} };

    Galois::on_each (
        [&] (const unsigned tid, const unsigned numT) {
          wl.push_initial (Galois::Runtime::makeLocalRange (graph));
        }, 
        Galois::loopname ("wl_init"));

    Galois::for_each_wl ( wl, ColorNodeLocking {*this}, "color-obim");

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
