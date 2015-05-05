#include "PageRankDet.h"


template <typename B>
struct NodeData: public B, PData {

  NodeData (void)
    : B (0), PData ()
  {}

  NodeData (unsigned id, unsigned outdegree)
    : B (id), PData (outdegree)
  {}


};

template <Galois::Runtime::InputDAG_ExecTy EXEC>
struct ChooseNodeDataType {
  using type = Galois::Runtime::InputDAGdataInOut;
};

template <>
struct ChooseNodeDataType<Galois::Runtime::InputDAG_ExecTy::PART> { 
  using type = Galois::Runtime::InputDAGdataPartInOut;
};




template <Galois::Runtime::InputDAG_ExecTy EXEC>
struct ChooseInnerGraph {
  using ND = NodeData<typename ChooseNodeDataType<EXEC>::type>; 

  using type = typename Galois::Graph::LC_CSR_Graph<ND, void>
    ::template with_numa_alloc<true>::type
    ::template with_no_lockable<true>::type;
};

template <Galois::Runtime::InputDAG_ExecTy EXEC>
class PageRankInputDAG: public PageRankBase<typename ChooseInnerGraph<EXEC>::type> {
protected:

  using Base = PageRankBase<typename ChooseInnerGraph<EXEC>::type>;
  using GNode = typename Base::GNode;

  struct ApplyOperator {

    static const unsigned CHUNK_SIZE = Base::DEFAULT_CHUNK_SIZE;
    static const unsigned UNROLL_FACTOR = 32;

    PageRankInputDAG& outer;

    template <typename C>
    void operator () (GNode src, C& ctx) {
      outer.applyOperator (src, ctx);
    }
  };

  virtual void runPageRank (void) {
    Galois::Runtime::ForEachDet_InputDAG<EXEC>::run (
        Galois::Runtime::makeLocalRange(Base::graph),
        ApplyOperator {*this},
        Base::graph,
        "page-rank-input-dag"
        );
  }

};

int main (int argc, char* argv[]) {
  LonestarStart (argc, argv, name, desc, url);

  switch (Galois::Runtime::inputDAG_ExecTy) {
    case Galois::Runtime::InputDAG_ExecTy::CHROMATIC: 
      {
        PageRankInputDAG<Galois::Runtime::InputDAG_ExecTy::CHROMATIC> p;
        p.run ();
        break;
      }
    case Galois::Runtime::InputDAG_ExecTy::EDGE_FLIP: 
      {
        PageRankInputDAG<Galois::Runtime::InputDAG_ExecTy::EDGE_FLIP> p;
        p.run ();
        break;
      }
    case Galois::Runtime::InputDAG_ExecTy::TOPO: 
      {
        PageRankInputDAG<Galois::Runtime::InputDAG_ExecTy::TOPO> p;
        p.run ();
        break;
      }
    case Galois::Runtime::InputDAG_ExecTy::PART: 
      {
        PageRankInputDAG<Galois::Runtime::InputDAG_ExecTy::PART> p;
        p.run ();
        break;
      }
    case Galois::Runtime::InputDAG_ExecTy::HYBRID: 
      {
        PageRankInputDAG<Galois::Runtime::InputDAG_ExecTy::HYBRID> p;
        p.run ();
        break;
      }

    default:
      std::abort ();
  }

} // end main


