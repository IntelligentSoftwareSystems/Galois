#ifndef KRUSKAL_ORDERED_H
#define KRUSKAL_ORDERED_H

#include "Kruskal.h"
#include "KruskalParallel.h"
#include "galois/runtime/KDGtwoPhase.h"

namespace kruskal {


struct UnionFindUsingRuntime {
  void operator () (
      EdgeCtxWL& perThrdWL, 
      VecRep_ty& repVec, 
      VecAtomicCtxPtr& repOwnerCtxVec, 
      size_t& mstWeight, 
      size_t& totalIter,
      galois::TimeAccumulator& sortTimer,
      galois::TimeAccumulator& findTimer,
      galois::TimeAccumulator& linkUpTimer,
      Accumulator& findIter,
      Accumulator& linkUpIter) const {


    EdgeCtxWL* nextWL = NULL; // not used actually
    Accumulator mstSum;

    // galois::for_each_ordered (perThrdWL.begin_all (), perThrdWL.end_all (),
    galois::runtime::for_each_ordered_ikdg (galois::runtime::makeLocalRange (perThrdWL),
        Edge::Comparator (),
        FindLoop (repVec, repOwnerCtxVec, findIter),
        LinkUpLoop<true> (repVec, repOwnerCtxVec, *nextWL, mstSum, linkUpIter),
        std::make_tuple (
          galois::needs_custom_locking<> (),
          galois::loopname ("kruskal-ikdg")));


    totalIter += findIter.reduce ();
    mstWeight += mstSum.reduce ();


  }
};



class KruskalOrdered: public Kruskal {
  protected:

  virtual const std::string getVersion () const { return "Parallel Kruskal using Ordered Runtime"; }

  virtual void runMST (const size_t numNodes, VecEdge& edges,
      size_t& mstWeight, size_t& totalIter) {

    if (! bool(galois::runtime::useParaMeterOpt) && (edges.size () >= 2 * numNodes)) {
      runMSTfilter (numNodes, edges, mstWeight, totalIter, UnionFindUsingRuntime ());
    } else {
      runMSTsimple (numNodes, edges, mstWeight, totalIter, UnionFindUsingRuntime ());
    }
  }
};




















}// end namespace kruskal




#endif //  KRUSKAL_ORDERED_H

