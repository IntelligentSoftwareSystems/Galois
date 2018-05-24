#ifndef KRUSKALSTRICTOBIM_H
#define KRUSKALSTRICTOBIM_H

#include "galois/graphs/Graph.h"
#include "galois/runtime/LevelExecutor.h"
#include "galois/worklists/WorkListExperimental.h"

#include "Kruskal.h"
#include "KruskalParallel.h"


namespace kruskal {


class KruskalStrictOBIM: public Kruskal {
  protected:

  typedef galois::graphs::FirstGraph<void,void,false> Graph;
  typedef Graph::GraphNode Lockable;
  typedef std::vector<Lockable> VecLocks;

  virtual const std::string getVersion () const { return "Parallel Kruskal using Strict OBIM"; }
  
  struct FindLoopSpec {

    Graph& graph;
    VecLocks& locks;
    VecRep& repVec;
    Accumulator& findIter;


    FindLoopSpec (
        Graph& graph,
        VecLocks& locks,
        VecRep& repVec,
        Accumulator& findIter)
      :
        graph (graph),
        locks (locks),
        repVec (repVec),
        findIter (findIter)

    {}


    template <typename C>
    void operator () (const Edge& e, C& ctx) {
      int repSrc = kruskal::findPCiter_int (e.src, repVec);
      int repDst = kruskal::findPCiter_int (e.dst, repVec);
      // int repSrc = kruskal::getRep_int (e.src, repVec);
      // int repDst = kruskal::getRep_int (e.dst, repVec);
      

      if (repSrc != repDst) {
        graph.getData (locks[repSrc]);
        graph.getData (locks[repDst]);
      }

      findIter += 1;
    }
  };


  struct LinkUpLoopSpec {

    static const unsigned CHUNK_SIZE = 64;

    VecRep& repVec;
    Accumulator& mstSum;
    Accumulator& linkUpIter;

    LinkUpLoopSpec (
        VecRep& repVec,
        Accumulator& mstSum,
        Accumulator& linkUpIter)
      :
        repVec (repVec),
        mstSum (mstSum),
        linkUpIter (linkUpIter)

    {}


    template <typename C>
    void operator () (const Edge& e, C& ctx) {
      int repSrc = kruskal::findPCiter_int (e.src, repVec);
      int repDst = kruskal::findPCiter_int (e.dst, repVec);

      // int repSrc = kruskal::getRep_int (e.src, repVec);
      // int repDst = kruskal::getRep_int (e.dst, repVec);

      if (repSrc != repDst) {
        unionByRank_int (repSrc, repDst, repVec);
        linkUpIter += 1;
        mstSum += e.weight;
      }

    }
  };

  struct GetWeight: public std::unary_function<Edge,Weight_ty> {
    Weight_ty operator () (const Edge& e) const {
      return e.weight;
    }
  };

  virtual void runMST (const size_t numNodes, VecEdge& edges,
      size_t& mstWeight, size_t& totalIter) {

    Graph graph;
    VecLocks locks;
    locks.resize(numNodes);
    galois::do_all(boost::counting_iterator<size_t>(0), boost::counting_iterator<size_t>(numNodes), [&](size_t i) {
      locks[i] = graph.createNode();
      graph.addNode(locks[i]);
    });
      
    VecRep repVec (numNodes, -1);
    Accumulator findIter;
    Accumulator linkUpIter;
    Accumulator mstSum;

    FindLoopSpec findLoop (graph, locks, repVec, findIter);
    LinkUpLoopSpec linkUpLoop (repVec, mstSum, linkUpIter);

    galois::TimeAccumulator runningTime;

    runningTime.start ();
    typedef galois::worklists::OrderedByIntegerMetric<GetWeight, galois::worklists::ChunkedFIFO<128>>::with_barrier<true>::type WL;
    galois::for_each(edges.begin(), edges.end(),
        [&](const Edge& e, galois::UserContext<Edge>& ctx) {
          findLoop(e, ctx);
          linkUpLoop(e, ctx);
    }, galois::wl<WL>());
    //}, galois::wl<galois::worklists::StableIterator<>>());
    //galois::runtime::for_each_ordered_level (
    //    galois::runtime::makeStandardRange (edges.begin (), edges.end ()),
    //    GetWeight (), std::less<Weight_ty> (), findLoop, linkUpLoop);

    runningTime.stop ();

    mstWeight = mstSum.reduce ();
    totalIter = findIter.reduce ();

    std::cout << "Number of FindLoop iterations = " << findIter.reduce () << std::endl;
    std::cout << "Number of LinkUpLoop iterations = " << linkUpIter.reduce () << std::endl;

    std::cout << "MST running time without initialization/destruction: " << runningTime.get () << std::endl;
  }
};



}// end namespace kruskal




#endif //  KRUSKAL_LEVEL_EXEC_H

