/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#include "PageRankDet.h"

template <typename B>
struct NodeData : public B, PData {

  NodeData(void) : B(0), PData() {}

  NodeData(unsigned id, unsigned outdegree) : B(id), PData(outdegree) {}
};

template <galois::runtime::InputDAG_ExecTy EXEC>
struct ChooseNodeDataType {
  using type = galois::runtime::InputDAGdataInOut;
};

template <>
struct ChooseNodeDataType<galois::runtime::InputDAG_ExecTy::PART> {
  using type = galois::runtime::InputDAGdataPartInOut;
};

template <galois::runtime::InputDAG_ExecTy EXEC>
struct ChooseInnerGraph {
  using ND = NodeData<typename ChooseNodeDataType<EXEC>::type>;

  using type =
      typename galois::graphs::LC_CSR_Graph<ND, void>::template with_numa_alloc<
          true>::type ::template with_no_lockable<true>::type;
};

template <galois::runtime::InputDAG_ExecTy EXEC>
class PageRankInputDAG
    : public PageRankBase<typename ChooseInnerGraph<EXEC>::type> {
protected:
  using Base  = PageRankBase<typename ChooseInnerGraph<EXEC>::type>;
  using GNode = typename Base::GNode;

  struct ApplyOperator {

    static const unsigned CHUNK_SIZE    = Base::DEFAULT_CHUNK_SIZE;
    static const unsigned UNROLL_FACTOR = 32;

    PageRankInputDAG& outer;

    template <typename C>
    void operator()(GNode src, C& ctx) {
      outer.applyOperator(src, ctx);
    }
  };

  virtual void runPageRank(void) {
    galois::runtime::ForEachDet_InputDAG<EXEC>::run(
        galois::runtime::makeLocalRange(Base::graph), ApplyOperator{*this},
        Base::graph, "page-rank-input-dag");
  }
};

int main(int argc, char* argv[]) {
  LonestarStart(argc, argv, name, desc, url);

  switch (galois::runtime::inputDAG_ExecTy) {
  case galois::runtime::InputDAG_ExecTy::CHROMATIC: {
    PageRankInputDAG<galois::runtime::InputDAG_ExecTy::CHROMATIC> p;
    p.run();
    break;
  }
  case galois::runtime::InputDAG_ExecTy::EDGE_FLIP: {
    PageRankInputDAG<galois::runtime::InputDAG_ExecTy::EDGE_FLIP> p;
    p.run();
    break;
  }
  case galois::runtime::InputDAG_ExecTy::TOPO: {
    PageRankInputDAG<galois::runtime::InputDAG_ExecTy::TOPO> p;
    p.run();
    break;
  }
  case galois::runtime::InputDAG_ExecTy::PART: {
    PageRankInputDAG<galois::runtime::InputDAG_ExecTy::PART> p;
    p.run();
    break;
  }
  case galois::runtime::InputDAG_ExecTy::HYBRID: {
    PageRankInputDAG<galois::runtime::InputDAG_ExecTy::HYBRID> p;
    p.run();
    break;
  }

  default:
    std::abort();
  }

} // end main
