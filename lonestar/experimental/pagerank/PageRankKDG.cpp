#include "PageRankDet.h"

namespace cll = llvm::cl;

enum ExecType {
  KDG_REUSE,
  KDG_R_ALT,
  KDG_R,
  KDG_AR,
  IKDG,
  UNORD,
};

static cll::opt<ExecType> execType (
    "execType",
    cll::desc ("Deterministic Executor Type"),
    cll::values (
      clEnumValN (KDG_REUSE, "KDG_REUSE", "KDG_REUSE"),
      clEnumValN (KDG_R_ALT, "KDG_R_ALT", "KDG_R_ALT"),
      clEnumValN (KDG_R, "KDG_R", "KDG_R"),
      clEnumValN (KDG_AR, "KDG_AR", "KDG_AR"),
      clEnumValN (IKDG, "IKDG", "IKDG"),
      clEnumValN (UNORD, "UNORD", "IKDG"),
      clEnumValEnd),
    cll::init (KDG_R));


struct NodeData: public galois::runtime::TaskDAGdata, PData {

  NodeData (void)
    : galois::runtime::TaskDAGdata (0), PData ()
  {}

  NodeData (unsigned id, unsigned outdegree)
    : galois::runtime::TaskDAGdata (id), PData (outdegree)
  {}


};

typedef typename galois::graphs::LC_CSR_Graph<NodeData, void>
  ::with_numa_alloc<true>::type
  InnerGraph;

class PageRankChromatic: public PageRankBase<InnerGraph> {
protected:

  struct NodeComparator {
    typedef galois::runtime::DAGdataComparator<NodeData> DataCmp;

    Graph& graph;

    bool operator () (GNode left, GNode right) const {
      auto& ld = graph.getData (left, galois::MethodFlag::UNPROTECTED);
      auto& rd = graph.getData (right, galois::MethodFlag::UNPROTECTED);

      return DataCmp::compare (ld, rd);
    }

  };

  struct NhoodVisitor {
    static const unsigned CHUNK_SIZE = DEFAULT_CHUNK_SIZE;

    PageRankChromatic& outer;

    template <typename C>
    void operator () (GNode src, C&) {
      outer.visitNhood (src);
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
    
    typedef typename galois::runtime::DAGmanagerInOut<Graph>::Manager Manager;
    Manager m {graph};
    m.assignPriority ();


    switch (execType) {
      case KDG_REUSE:
        galois::runtime::for_each_det_kdg_ar_reuse (
            galois::runtime::makeLocalRange (graph),
            NodeComparator {graph},
            NhoodVisitor {*this},
            ApplyOperator {*this},
            graph, 
            "page-rank-kdg-reuse");
        break;

      case KDG_R_ALT:
        galois::runtime::for_each_det_kdg (
            galois::runtime::makeLocalRange (graph),
            NodeComparator {graph},
            NhoodVisitor {*this},
            ApplyOperator {*this},
            graph, 
            "page-rank-kdg-r-alt",
            galois::runtime::KDG_R_ALT);
        break;

      case KDG_R:
        galois::runtime::for_each_det_kdg (
            galois::runtime::makeLocalRange (graph),
            NodeComparator {graph},
            NhoodVisitor {*this},
            ApplyOperator {*this},
            graph, 
            "page-rank-kdg-r",
            galois::runtime::KDG_R);
        break;

      case KDG_AR:
        galois::runtime::for_each_det_kdg (
            galois::runtime::makeLocalRange (graph),
            NodeComparator {graph},
            NhoodVisitor {*this},
            ApplyOperator {*this},
            graph, 
            "page-rank-kdg-ar",
            galois::runtime::KDG_AR);
        break;

      case IKDG:
        galois::runtime::for_each_det_kdg (
            galois::runtime::makeLocalRange (graph),
            NodeComparator {graph},
            NhoodVisitor {*this},
            ApplyOperator {*this},
            graph, 
            "page-rank-kdg-ikdg",
            galois::runtime::IKDG);
        break;

      case UNORD:
        galois::for_each_local (
            graph,
            [this] (GNode src, galois::UserContext<GNode>& ctx) {
              // visitNhood (src);
              applyOperator<galois::UserContext<GNode>, true, true> (src, ctx);
            },
            galois::loopname ("page-rank-unordered"),
            galois::wl<galois::WorkList::AltChunkedFIFO<DEFAULT_CHUNK_SIZE> > ());
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
