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

  using VecRep =  Galois::LazyDynamicArray<int, Galois::Runtime::SerialNumaAllocator<int> >;
  using Lock = Galois::Runtime::Lockable;
  using VecLock =  Galois::LazyDynamicArray<Lock, Galois::Runtime::SerialNumaAllocator<Lock> >;

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
        Galois::Runtime::acquire (&lockVec[e.repSrc], Galois::MethodFlag::WRITE);
        Galois::Runtime::acquire (&lockVec[e.repDst], Galois::MethodFlag::WRITE);
      }

      findIter += 1;
    }
  };

  template <typename L>
  void runMSTwithOrderedLoop (const size_t numNodes, VecEdge& edges,
      size_t& mstWeight, size_t& totalIter, L orderedLoop) {

    VecLock lockVec (numNodes);
    VecRep repVec (numNodes);


    Galois::StatTimer timeInit("kruska-init-Time");

    timeInit.start ();

    Galois::Substrate::getThreadPool().burnPower(Galois::getActiveThreads());

    Galois::do_all_choice (
        Galois::Runtime::makeStandardRange(
          boost::counting_iterator<size_t>(0),
          boost::counting_iterator<size_t>(numNodes)),
        [&lockVec, &repVec] (size_t i) {
          repVec.initialize (i, -1);
          lockVec.initialize (i, Lock ());
        },
        std::make_tuple (
          Galois::chunk_size<DEFAULT_CHUNK_SIZE> (),
          Galois::loopname ("init-vectors")));


    
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

    Galois::StatTimer runningTime("time for running MST loop:");

    runningTime.start ();
    orderedLoop (
        Galois::Runtime::makeStandardRange (edge_beg, edge_end),
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
