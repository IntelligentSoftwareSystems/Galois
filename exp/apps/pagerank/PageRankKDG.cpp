#include "PageRankDet.h"

namespace cll = llvm::cl;

enum ExecType {
  KDG_R_ALT,
  KDG_R,
  KDG_AR,
  IKDG,
};

static cll::opt<ExecType> execType (
    "executor",
    cll::desc ("Deterministic Executor Type"),
    cll::values (
      clEnumValN (KDG_R_ALT, "KDG_R_ALT", "KDG_R_ALT"),
      clEnumValN (KDG_R, "KDG_R", "KDG_R"),
      clEnumValN (KDG_AR, "KDG_AR", "KDG_AR"),
      clEnumValN (IKDG, "IKDG", "IKDG"),
      clEnumValEnd),
    cll::init (KDG_R));


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
  InnerGraph;

class PageRankChromatic: public PageRankBase<InnerGraph> {
protected:

  struct NodeComparator {
    typedef Galois::Runtime::DAGdataComparator<NodeData> DataCmp;

    Graph& graph;

    bool operator () (GNode left, GNode right) const {
      auto& ld = graph.getData (left, Galois::MethodFlag::UNPROTECTED);
      auto& rd = graph.getData (right, Galois::MethodFlag::UNPROTECTED);

      return DataCmp::compare (ld, rd);
    }

  };

  struct NhoodVisitor {
    static const unsigned CHUNK_SIZE = DEFAULT_CHUNK_SIZE;

    Graph& graph;

    template <typename C>
    void operator () (GNode src, C&) {
      graph.getData (src, Galois::MethodFlag::WRITE);

      for (auto i = graph.in_edge_begin (src, Galois::MethodFlag::READ)
          , end_i = graph.in_edge_end (src, Galois::MethodFlag::READ); i != end_i; ++i) {
      }

      // for (auto i = graph.edge_begin (src, Galois::MethodFlag::WRITE)
          // , end_i = graph.edge_end (src, Galois::MethodFlag::WRITE); i != end_i; ++i) {
      // }
    }
  };


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
    
    typedef typename Galois::Runtime::DAGmanagerInOut<Graph>::Manager Manager;
    Manager m {graph};
    m.assignPriority ();


    switch (execType) {
      case KDG_R_ALT:
        Galois::Runtime::for_each_det_kdg (
            Galois::Runtime::makeLocalRange (graph),
            NodeComparator {graph},
            NhoodVisitor {graph},
            ApplyOperator {*this},
            graph, 
            "page-rank-kdg-r-alt",
            Galois::Runtime::KDG_R_ALT);
        break;

      case KDG_R:
        Galois::Runtime::for_each_det_kdg (
            Galois::Runtime::makeLocalRange (graph),
            NodeComparator {graph},
            NhoodVisitor {graph},
            ApplyOperator {*this},
            graph, 
            "page-rank-kdg-r",
            Galois::Runtime::KDG_R);
        break;

      case KDG_AR:
        Galois::Runtime::for_each_det_kdg (
            Galois::Runtime::makeLocalRange (graph),
            NodeComparator {graph},
            NhoodVisitor {graph},
            ApplyOperator {*this},
            graph, 
            "page-rank-kdg-ar",
            Galois::Runtime::KDG_AR);
        break;

      case IKDG:
        Galois::Runtime::for_each_det_kdg (
            Galois::Runtime::makeLocalRange (graph),
            NodeComparator {graph},
            NhoodVisitor {graph},
            ApplyOperator {*this},
            graph, 
            "page-rank-kdg-ikdg",
            Galois::Runtime::IKDG);
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
