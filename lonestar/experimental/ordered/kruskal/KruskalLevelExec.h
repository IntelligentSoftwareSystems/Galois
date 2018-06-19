/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

#ifndef KRUSKAL_LEVEL_EXEC_H
#define KRUSKAL_LEVEL_EXEC_H

#include "galois/graphs/Graph.h"
#include "galois/runtime/LevelExecutor.h"

#include "Kruskal.h"
#include "KruskalParallel.h"

namespace kruskal {

class KruskalLevelExec : public Kruskal {
protected:
  typedef galois::graphs::MorphGraph<void*, void, true> Graph;
  typedef Graph::GraphNode Lockable;
  typedef std::vector<Lockable> VecLocks;

  virtual const std::string getVersion() const {
    return "Parallel Kruskal using Speculative Ordered Runtime";
  }

  struct FindLoopSpec {

    Graph& graph;
    VecLocks& locks;
    VecRep& repVec;
    Accumulator& findIter;

    FindLoopSpec(Graph& graph, VecLocks& locks, VecRep& repVec,
                 Accumulator& findIter)
        : graph(graph), locks(locks), repVec(repVec), findIter(findIter)

    {}

    template <typename C>
    void operator()(const Edge& e, C& ctx) const {
      int repSrc = kruskal::findPCiter_int(e.src, repVec);
      int repDst = kruskal::findPCiter_int(e.dst, repVec);
      // int repSrc = kruskal::getRep_int (e.src, repVec);
      // int repDst = kruskal::getRep_int (e.dst, repVec);

      if (repSrc != repDst) {
        graph.getData(locks[repSrc]);
        graph.getData(locks[repDst]);
      }

      findIter += 1;
    }
  };

  struct LinkUpLoopSpec {

    static const unsigned CHUNK_SIZE = 64;

    VecRep& repVec;
    Accumulator& mstSum;
    Accumulator& linkUpIter;

    LinkUpLoopSpec(VecRep& repVec, Accumulator& mstSum, Accumulator& linkUpIter)
        : repVec(repVec), mstSum(mstSum), linkUpIter(linkUpIter)

    {}

    template <typename C>
    void operator()(const Edge& e, C& ctx) const {
      int repSrc = kruskal::findPCiter_int(e.src, repVec);
      int repDst = kruskal::findPCiter_int(e.dst, repVec);

      // int repSrc = kruskal::getRep_int (e.src, repVec);
      // int repDst = kruskal::getRep_int (e.dst, repVec);

      if (repSrc != repDst) {
        unionByRank_int(repSrc, repDst, repVec);
        linkUpIter += 1;
        mstSum += e.weight;
      }
    }
  };

  struct GetWeight {
    Weight_ty operator()(const Edge& e) const { return e.weight; }
  };

  virtual void runMST(const size_t numNodes, VecEdge& edges, size_t& mstWeight,
                      size_t& totalIter) {

    Graph graph;
    VecLocks locks;
    locks.reserve(numNodes);
    for (size_t i = 0; i < numNodes; ++i) {
      locks.push_back(graph.createNode(nullptr));
    }

    VecRep repVec(numNodes, -1);
    Accumulator findIter;
    Accumulator linkUpIter;
    Accumulator mstSum;

    FindLoopSpec findLoop(graph, locks, repVec, findIter);
    LinkUpLoopSpec linkUpLoop(repVec, mstSum, linkUpIter);

    galois::TimeAccumulator runningTime;

    runningTime.start();
    galois::runtime::for_each_ordered_level(
        galois::runtime::makeStandardRange(edges.begin(), edges.end()),
        GetWeight(), std::less<Weight_ty>(), findLoop, linkUpLoop);

    runningTime.stop();

    mstWeight = mstSum.reduce();
    totalIter = findIter.reduce();

    std::cout << "Number of FindLoop iterations = " << findIter.reduce()
              << std::endl;
    std::cout << "Number of LinkUpLoop iterations = " << linkUpIter.reduce()
              << std::endl;

    std::cout << "MST running time without initialization/destruction: "
              << runningTime.get() << std::endl;
  }
};

} // end namespace kruskal

#endif //  KRUSKAL_LEVEL_EXEC_H
