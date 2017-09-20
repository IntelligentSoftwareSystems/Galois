#ifndef KRUSKAL_RUNTIME_H_
#define KRUSKAL_RUNTIME_H_

#include "Kruskal.h"

#include "Galois/DynamicArray.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/Runtime/OrderedSpeculation.h"
#include "Galois/Runtime/KDGtwoPhase.h"
#include "Galois/Accumulator.h"
#include "Galois/PerThreadContainer.h"

#include <boost/iterator/transform_iterator.hpp>

namespace kruskal {

static cll::opt<bool> useCustomLocking (
    "custLock",
    cll::desc ("use Custom Locking"),
    cll::init (true));



struct KruskalRuntime: public Kruskal {

  using VecRep =  galois::LazyDynamicArray<int, galois::runtime::SerialNumaAllocator<int> >;
  using Lock = galois::runtime::Lockable;
  using VecLock =  galois::LazyDynamicArray<Lock, galois::runtime::SerialNumaAllocator<Lock> >;

  struct EdgeCtxt: public Edge {
    int repSrc;
    int repDst;

    EdgeCtxt (const Edge& e): 
      Edge (e), 
      repSrc (e.src),
      repDst (e.dst)
    {}
  };

  struct FindLoopRuntime {
    static const unsigned CHUNK_SIZE = DEFAULT_CHUNK_SIZE;

    VecLock& lockVec;
    VecRep& repVec;
    Accumulator& findIter;

    template <typename C>
    void operator () (EdgeCtxt& e, C& ctx) {
      e.repSrc = kruskal::findPCiter_int (e.src, repVec);
      e.repDst = kruskal::findPCiter_int (e.dst, repVec);
      // int repSrc = kruskal::getRep_int (e.src, repVec);
      // int repDst = kruskal::getRep_int (e.dst, repVec);
      

      if (e.repSrc != e.repDst) {
        galois::runtime::acquire (&lockVec[e.repSrc], galois::MethodFlag::WRITE);
        galois::runtime::acquire (&lockVec[e.repDst], galois::MethodFlag::WRITE);
      }

      findIter += 1;
    }
  };

  template <typename L>
  void runMSTwithOrderedLoop (const size_t numNodes, VecEdge& edges,
      size_t& mstWeight, size_t& totalIter, L orderedLoop) {

    VecLock lockVec (numNodes);
    VecRep repVec (numNodes);


    galois::StatTimer timeInit("kruska-init-Time");

    timeInit.start ();

    galois::substrate::getThreadPool().burnPower(galois::getActiveThreads());

    galois::do_all_choice (
        galois::runtime::makeStandardRange(
          boost::counting_iterator<size_t>(0),
          boost::counting_iterator<size_t>(numNodes)),
        [&lockVec, &repVec] (size_t i) {
          repVec.initialize (i, -1);
          lockVec.initialize (i, Lock ());
        },
        std::make_tuple (
          galois::chunk_size<DEFAULT_CHUNK_SIZE> (),
          galois::loopname ("init-vectors")));


    
    timeInit.stop ();


    Accumulator findIter;
    Accumulator linkUpIter;
    Accumulator mstSum;


    // auto makeEdgeCtxt = [] (const Edge& e) { return EdgeCtxt (e); };
    struct MakeEdgeCtxt: public std::unary_function<const Edge&, EdgeCtxt> {
      EdgeCtxt operator () (const Edge& e) const { 
        return EdgeCtxt (e);
      }
    };
    MakeEdgeCtxt makeEdgeCtxt;
    auto edge_beg = boost::make_transform_iterator (edges.begin(), makeEdgeCtxt);
    auto edge_end = boost::make_transform_iterator (edges.end(), makeEdgeCtxt);

    galois::StatTimer runningTime("time for running MST loop:");

    runningTime.start ();
    orderedLoop (
        galois::runtime::makeStandardRange (edge_beg, edge_end),
        lockVec,
        repVec,
        mstSum, 
        findIter, 
        linkUpIter);
    runningTime.stop ();

    mstWeight = mstSum.reduce ();
    totalIter = findIter.reduce ();

    std::cout << "Weight caclulated by accumulator: " << mstSum.reduce () << std::endl;
    std::cout << "Number of FindLoop iterations = " << findIter.reduce () << std::endl;
    std::cout << "Number of LinkUpLoop iterations = " << linkUpIter.reduce () << std::endl;
  }

};



} // end namespace kruskal


#endif //  KRUSKAL_RUNTIME_H_
