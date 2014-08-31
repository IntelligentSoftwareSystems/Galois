#ifndef GRAPH_COLORING_LOCKING_H
#define GRAPH_COLORING_LOCKING_H

#include "GraphColoringBase.h"

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

  void firstFit (void) {
    typedef Galois::WorkList::AltChunkedFIFO<DEFAULT_CHUNK_SIZE> WL_ty;
    // typedef Galois::WorkList::dChunkedFIFO<DEFAULT_CHUNK_SIZE> WL_ty;

    Galois::for_each_local (
        graph, ColorNodeLocking {*this},
        Galois::loopname ("first-fit"),
        Galois::wl<WL_ty> ());

  }

  virtual void colorGraph (void) {
    switch (heuristic) {
      case FIRST_FIT:
        firstFit ();
        break;

      default:
        std::abort ();
        
    }
  }

};


#endif // GRAPH_COLORING_LOCKING_H
