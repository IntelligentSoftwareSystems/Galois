#include "PageRankDet.h"


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

  struct ApplyOperator {

    static const unsigned CHUNK_SIZE = DEFAULT_CHUNK_SIZE;

    PageRankChromatic& outer;

    template <typename C>
    void operator () (GNode src, C& ctx) {
      outer.applyOperator (src, ctx);
    }
  };

  virtual void runPageRank (void) {
    Galois::Runtime::for_each_det_graph (
        Galois::Runtime::makeLocalRange (graph),
        graph,
        ApplyOperator {*this},
        "page-rank-chromatic");
  }

};

int main (int argc, char* argv[]) {

  PageRankChromatic p;

  return p.run (argc, argv);
}
