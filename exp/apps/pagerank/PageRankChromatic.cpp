#include "PageRankDet.h"

namespace cll = llvm::cl;

enum ExecType {
  CHROMATIC,
  EDGE_FLIP,
  TOPO,
};

static cll::opt<ExecType> execType (
    "executor",
    cll::desc ("Deterministic Executor Type"),
    cll::values (
      clEnumValN (CHROMATIC, "CHROMATIC", "Chromatic Executor"),
      clEnumValN (EDGE_FLIP, "EDGE_FLIP", "Edge Flipping DAG overlayed on input graph"),
      clEnumValN (TOPO, "TOPO", "Edge Flipping DAG overlayed on input graph"),
      clEnumValEnd),
    cll::init (CHROMATIC));


struct NodeData: public Galois::Runtime::InputDAGdata, PData {

  NodeData (void)
    : Galois::Runtime::InputDAGdata (0), PData ()
  {}

  NodeData (unsigned id, unsigned outdegree)
    : Galois::Runtime::InputDAGdata (id), PData (outdegree)
  {}


};

typedef typename Galois::Graph::LC_CSR_Graph<NodeData, void>
  ::with_numa_alloc<true>::type
  ::with_no_lockable<true>::type 
  InnerGraph;

class PageRankChromatic: public PageRankBase<InnerGraph> {
protected:

  struct ApplyOperator {

    static const unsigned CHUNK_SIZE = DEFAULT_CHUNK_SIZE;
    static const unsigned UNROLL_FACTOR = 32;

    PageRankChromatic& outer;

    template <typename C>
    void operator () (GNode src, C& ctx) {
      outer.applyOperator (src, ctx);
    }
  };

  virtual void runPageRank (void) {

    switch (execType) {
      
      case CHROMATIC:
        Galois::Runtime::for_each_det_chromatic (
            Galois::Runtime::makeLocalRange (graph),
            ApplyOperator {*this},
            graph,
            "page-rank-chromatic");
        break;

      case EDGE_FLIP:
        Galois::Runtime::for_each_det_edge_flip_ar (
            Galois::Runtime::makeLocalRange (graph),
            ApplyOperator {*this},
            graph,
            "page-rank-chromatic");
        break;

      case TOPO:
        Galois::Runtime::for_each_det_edge_flip_topo (
            Galois::Runtime::makeLocalRange (graph),
            ApplyOperator {*this},
            graph,
            "page-rank-chromatic");
        break;

      default:
        std::abort ();

    }
  }

};

int main (int argc, char* argv[]) {

  PageRankChromatic p;

  return p.run (argc, argv);
}
