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

#include "KruskalRuntime.h"

namespace kruskal {

class KruskalIKDG : public KruskalRuntime {

  struct LinkUpLoopIKDG {

    static const unsigned CHUNK_SIZE = DEFAULT_CHUNK_SIZE;

    VecLock& lockVec;
    VecRep& repVec;
    Accumulator& mstSum;
    Accumulator& linkUpIter;

    template <typename C>
    void operator()(const EdgeCtxt& e, C& ctx) {

      if (e.repSrc != e.repDst) {

        bool srcFail = !galois::runtime::owns(&lockVec[e.repSrc],
                                              galois::MethodFlag::WRITE);
        bool dstFail = !galois::runtime::owns(&lockVec[e.repDst],
                                              galois::MethodFlag::WRITE);

        if (srcFail && dstFail) {
          galois::runtime::signalConflict();

        } else {

          if (srcFail) {
            linkUp_int(e.repDst, e.repSrc, repVec);

          } else {
            linkUp_int(e.repSrc, e.repDst, repVec);
          }

          linkUpIter += 1;
          mstSum += e.weight;
        }
      }
    }
  };

  struct UnionByRankIKDG {

    static const unsigned CHUNK_SIZE = DEFAULT_CHUNK_SIZE;

    VecLock& lockVec;
    VecRep& repVec;
    Accumulator& mstSum;
    Accumulator& linkUpIter;

    template <typename C>
    void operator()(const EdgeCtxt& e, C& ctx) {
      // int repSrc = kruskal::getRep_int (e.src, repVec);
      // int repDst = kruskal::getRep_int (e.dst, repVec);

      if (e.repSrc != e.repDst) {
        unionByRank_int(e.repSrc, e.repDst, repVec);

        linkUpIter += 1;
        mstSum += e.weight;
      }
    }
  };

  struct RunOrderedSpecOpt {

    template <typename R>
    void operator()(const R& edgeRange, VecLock& lockVec, VecRep& repVec,
                    Accumulator& mstSum, Accumulator& findIter,
                    Accumulator& linkUpIter) {

      FindLoopRuntime findLoop{lockVec, repVec, findIter};
      LinkUpLoopIKDG linkUpLoop{lockVec, repVec, mstSum, linkUpIter};

      galois::runtime::for_each_ordered_ikdg(
          edgeRange, Edge::Comparator(), findLoop, linkUpLoop,
          std::make_tuple(galois::needs_custom_locking<>(),
                          galois::loopname("kruskal-speculative-opt")));
    }
  };

  struct RunOrderedSpecBase {

    template <typename R>
    void operator()(const R& edgeRange, VecLock& lockVec, VecRep& repVec,
                    Accumulator& mstSum, Accumulator& findIter,
                    Accumulator& linkUpIter) {

      FindLoopRuntime findLoop{lockVec, repVec, findIter};
      UnionByRankIKDG linkUpLoop{lockVec, repVec, mstSum, linkUpIter};

      galois::runtime::for_each_ordered_ikdg(
          edgeRange, Edge::Comparator(), findLoop, linkUpLoop,
          std::make_tuple(galois::loopname("kruskal-speculative-base")));
    }
  };

  virtual const std::string getVersion() const {
    return "Parallel Kruskal using IKDG";
  }

  virtual void runMST(const size_t numNodes, VecEdge& edges, size_t& mstWeight,
                      size_t& totalIter) {

    if (useCustomLocking) {
      runMSTwithOrderedLoop(numNodes, edges, mstWeight, totalIter,
                            RunOrderedSpecOpt{});
    } else {
      runMSTwithOrderedLoop(numNodes, edges, mstWeight, totalIter,
                            RunOrderedSpecBase{});
    }
  }
};

} // end namespace kruskal

int main(int argc, char* argv[]) {
  kruskal::KruskalIKDG k;
  k.run(argc, argv);
  return 0;
}
