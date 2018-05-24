#include "KruskalRuntime.h"

namespace kruskal {


class KruskalSpec: public KruskalRuntime {

  struct LinkUpLoopSpec {

    static const unsigned CHUNK_SIZE = DEFAULT_CHUNK_SIZE;

    VecLock& lockVec;
    VecRep& repVec;
    Accumulator& mstSum;
    Accumulator& linkUpIter;

    template <typename C>
    void operator () (const EdgeCtxt& e, C& ctx) {

      if (e.repSrc != e.repDst) {

        bool srcFail = !galois::runtime::owns (&lockVec[e.repSrc], galois::MethodFlag::WRITE);
        bool dstFail = !galois::runtime::owns (&lockVec[e.repDst], galois::MethodFlag::WRITE);

        if (srcFail && dstFail) {
          galois::runtime::signalConflict();

        } else { 

          if (srcFail) {
            int origDst = repVec[e.repDst];

            auto f0 = [this, origDst, &e] (void) {
              repVec[e.repDst] = origDst;
            };

            ctx.addUndoAction (f0);

            linkUp_int (e.repDst, e.repSrc, repVec);

          } else {
            int origSrc = repVec[e.repSrc];

            auto f1 = [this, origSrc, &e] (void) {
              repVec[e.repSrc] = origSrc;
            };

            ctx.addUndoAction (f1);

            linkUp_int (e.repSrc, e.repDst, repVec);


          }

          auto f2 = [this, &e] (void) {
            linkUpIter += 1;
            mstSum += e.weight;
          };

          ctx.addCommitAction (f2);
        }
      }
    }
  };

  struct UnionByRankSpec {

    static const unsigned CHUNK_SIZE = DEFAULT_CHUNK_SIZE;

    VecLock& lockVec;
    VecRep& repVec;
    Accumulator& mstSum;
    Accumulator& linkUpIter;


    template <typename C>
    void operator () (const EdgeCtxt& e, C& ctx) {
      // int repSrc = kruskal::getRep_int (e.src, repVec);
      // int repDst = kruskal::getRep_int (e.dst, repVec);

      if (e.repSrc != e.repDst) {

        int origSrc = repVec[e.repSrc];
        int origDst = repVec[e.repDst];

        auto u = [this, &e, origSrc, origDst] (void) {
          repVec[e.repSrc] = origSrc;
          repVec[e.repDst] = origDst;
        };

        ctx.addUndoAction (u);

        unionByRank_int (e.repSrc, e.repDst, repVec);

        auto f = [&e, this] (void) {
          linkUpIter += 1;
          mstSum += e.weight;
        };

        ctx.addCommitAction (f);
      }
    }
  };

  struct RunOrderedSpecOpt {

    template <typename R>
    void operator () (
        const R& edgeRange,
        VecLock& lockVec,
        VecRep& repVec,
        Accumulator& mstSum,
        Accumulator&  findIter, 
        Accumulator& linkUpIter) {

      FindLoopRuntime findLoop {lockVec, repVec, findIter};
      LinkUpLoopSpec linkUpLoop {lockVec, repVec, mstSum, linkUpIter};

      galois::runtime::for_each_ordered_spec (
          edgeRange,
          Edge::Comparator (), findLoop, linkUpLoop,
          std::make_tuple (
            galois::needs_custom_locking<> (),
            galois::loopname ("kruskal-speculative-opt")));

    }
  };

  struct RunOrderedSpecBase {

    template <typename R>
    void operator () (
        const R& edgeRange,
        VecLock& lockVec,
        VecRep& repVec,
        Accumulator& mstSum,
        Accumulator&  findIter, 
        Accumulator& linkUpIter) {

      FindLoopRuntime findLoop {lockVec, repVec, findIter};
      UnionByRankSpec linkUpLoop {lockVec, repVec, mstSum, linkUpIter};

      galois::runtime::for_each_ordered_spec (
          edgeRange,
          Edge::Comparator (), findLoop, linkUpLoop,
          std::make_tuple (
            galois::loopname ("kruskal-speculative-base")));

    }
  };

  virtual const std::string getVersion () const { return "Parallel Kruskal using Speculative Ordered Runtime"; }

  virtual void runMST (const size_t numNodes, VecEdge& edges,
      size_t& mstWeight, size_t& totalIter) {

    if (useCustomLocking) {
      runMSTwithOrderedLoop (numNodes, edges, mstWeight, totalIter, RunOrderedSpecOpt {});
    } else {
      runMSTwithOrderedLoop (numNodes, edges, mstWeight, totalIter, RunOrderedSpecBase {});
    }
  }

};

} // end namespace kruskal

int main (int argc, char* argv[]) {
  kruskal::KruskalSpec k;
  k.run (argc, argv);
  return 0;
}
