#include "PageRankDet.h"

namespace cll = llvm::cl;

enum ExecType {
  CHROMATIC,
  EDGE_FLIP,
};

static cll::opt<ExecType> execType (
    "executor",
    cll::desc ("Deterministic Executor Type"),
    cll::values (
      clEnumValN (CHROMATIC, "CHROMATIC", "Chromatic Executor"),
      clEnumValN (EDGE_FLIP, "EDGE_FLIP", "Edge Flipping DAG overlayed on input graph"),
      clEnumValEnd),
    cll::init (CHROMATIC));


struct NodeData: public Galois::Runtime::DAGdata, PData {

  NodeData (void)
    : Galois::Runtime::DAGdata (0), PData ()
  {}

  NodeData (unsigned id, unsigned outdegree)
    : Galois::Runtime::DAGdata (id), PData (outdegree)
  {}


};

typedef typename Galois::Graph::LC_CSR_Graph<NodeData, void>
  ::with_numa_alloc<true>::type
  ::with_no_lockable<true>::type 
  InnerGraph;

class PageRankChromatic: public PageRankBase<InnerGraph> {
protected:

  template <bool useOnWL> 
  struct ApplyOperator {

    static const unsigned CHUNK_SIZE = DEFAULT_CHUNK_SIZE;
    static const unsigned UNROLL_FACTOR = 32;

    PageRankChromatic& outer;

    template <typename C>
    void operator () (GNode src, C& ctx) {
      outer.applyOperator<useOnWL> (src, ctx);
    }
  };

  virtual void runPageRank (void) {

    switch (execType) {
      
      case CHROMATIC:
        Galois::Runtime::for_each_det_chromatic (
            Galois::Runtime::makeLocalRange (graph),
            ApplyOperator<false> {*this},
            graph,
            "page-rank-chromatic");
        break;

      case EDGE_FLIP:
#if 1
        Galois::Runtime::for_each_det_edge_flip_ar (
            Galois::Runtime::makeLocalRange (graph),
            ApplyOperator<false> {*this},
            graph,
            "page-rank-chromatic");
#else
        abort();
#endif
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
