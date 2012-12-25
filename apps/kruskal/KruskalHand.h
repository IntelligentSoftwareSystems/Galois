/** Kruskal MST -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
 *
 * @section Description
 *
 * Kruskal MST.
 *
 * @author <ahassaan@ices.utexas.edu>
 */

#ifndef KRUSKAL_HAND_H
#define KRUSKAL_HAND_H

#include "Galois/Atomic.h"
#include "Galois/Accumulator.h"
#include "Galois/Runtime/PerThreadWorkList.h"

#include "Kruskal.h"
#include "KruskalParallel.h"

#include <algorithm>

#include <boost/iterator/counting_iterator.hpp>

namespace kruskal {

// struct PtrCmp {
// 
  // inline bool operator () (const EdgeCtx* left, const EdgeCtx* right) {
    // assert (left != NULL && right != NULL);
    // return (Cmp::compare (*left, *right) < 0);
  // }
// };

struct PreSort {
  EdgeCtxWL& wl;

  PreSort (EdgeCtxWL& wl): wl (wl) {}

  GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (unsigned r) {
    std::sort (wl[r].begin (), wl[r].end (), Cmp ());
  }
};


void runTwoPhaseKruskal (
    EdgeCtxWL& perThrdWL, 
    VecRep_ty& repVec, 
    VecAtomicCtxPtr& minEdgeCtxVec, 
    size_t& mstWeight, 
    size_t& totalIter,
    Galois::TimeAccumulator& findTimer,
    Galois::TimeAccumulator& linkUpTimer,
    Accumulator& findIter,
    Accumulator& linkUpIter) {



  unsigned round = 0;
  size_t numUnions = 0;

  Accumulator mstSum;

  EdgeCtxWL* const tmp = new EdgeCtxWL ();

  EdgeCtxWL* nextWL = &perThrdWL;
  EdgeCtxWL* currWL = tmp;

#ifdef GALOIS_USE_EXP
  Galois::do_all_choice (boost::counting_iterator<unsigned> (0), boost::counting_iterator<unsigned> (perThrdWL.numRows ()), 
      PreSort (perThrdWL), "pre_sort");
#else
  Galois::do_all (boost::counting_iterator<unsigned> (0), boost::counting_iterator<unsigned> (perThrdWL.numRows ()), 
      PreSort (perThrdWL), "pre_sort");
#endif

  while (!nextWL->empty_all ()) {
    ++round;
    std::swap (nextWL, currWL);
    nextWL->clear_all ();


    GaloisRuntime::beginSampling ();
    findTimer.start ();
#ifdef GALOIS_USE_EXP
    Galois::do_all_choice (*currWL,
        FindLoop (repVec, minEdgeCtxVec, findIter),
        "find_loop");
#else
    Galois::do_all (currWL->begin_all (), currWL->end_all (),
        FindLoop (repVec, minEdgeCtxVec, findIter),
        "find_loop");
#endif
    findTimer.stop ();
    GaloisRuntime::endSampling ();


    // GaloisRuntime::beginSampling ();
    linkUpTimer.start ();
#ifdef GALOIS_USE_EXP
    Galois::do_all_choice (*currWL,
        LinkUpLoop<false> (repVec, minEdgeCtxVec, *nextWL, mstSum, linkUpIter),
        "link_up_loop");
#else
    Galois::do_all (currWL->begin_all (), currWL->end_all (),
        LinkUpLoop<false> (repVec, minEdgeCtxVec, *nextWL, mstSum, linkUpIter),
        "link_up_loop");
#endif
    linkUpTimer.stop ();
    // GaloisRuntime::endSampling ();

    int u = linkUpIter.reduce () - numUnions;
    numUnions = linkUpIter.reduce ();

    if (!nextWL->empty_all ()) {
      assert (u > 0 && "no unions, no progress?");
    }

  }

  totalIter += findIter.reduce ();
  mstWeight += mstSum.reduce ();

  std::cout << "Number of rounds: " << round << std::endl;


  delete tmp;
}



class KruskalHand: public Kruskal {
  protected:
  // static const double EDGE_FRAC = 4/3;

  virtual const std::string getVersion () const { return "Parallel Handwritten Ordered Kruskal"; }

  virtual void runMST (const size_t numNodes, const VecEdge& edges,
      size_t& mstWeight, size_t& totalIter) {

    runMSTparallel (numNodes, edges, mstWeight, totalIter, &runTwoPhaseKruskal);

  }
};

struct FillUp {
  EdgeCtxAlloc& alloc;
  EdgeCtxWL& wl;

  FillUp (EdgeCtxAlloc& alloc, EdgeCtxWL& wl): alloc (alloc), wl (wl) {}

  GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (const Edge& edge) {

    // EdgeCtx* ctx = alloc.allocate (1);
    // assert (ctx != NULL);
    // alloc.construct (ctx, EdgeCtx (edge));
    EdgeCtx ctx (edge);

    wl.get ().push_back (ctx);
  }
};


void runMSTnaive (const size_t numNodes, const VecEdge& edges, 
    size_t& mstWeight, size_t& totalIter) {

  totalIter = 0;
  mstWeight = 0;

  Galois::TimeAccumulator runningTime;
  Galois::TimeAccumulator findTimer;
  Galois::TimeAccumulator linkUpTimer;

  Accumulator findIter;
  Accumulator linkUpIter;


  VecRep_ty repVec (numNodes, -1);
  VecAtomicCtxPtr minEdgeCtxVec (numNodes, AtomicCtxPtr (NULL));
  EdgeCtxAlloc alloc;



  Galois::preAlloc (128*Galois::getActiveThreads ());

  EdgeCtxWL initWL;

#ifdef GALOIS_USE_EXP
  Galois::do_all_choice (edges.begin (), edges.end (), FillUp (alloc, initWL), "fill_init");
#else
  Galois::do_all (edges.begin (), edges.end (), FillUp (alloc, initWL), "fill_init");
#endif
  runningTime.start ();

  runTwoPhaseKruskal (initWL, repVec, minEdgeCtxVec, 
      mstWeight, totalIter, findTimer, linkUpTimer, findIter, linkUpIter);

  runningTime.stop ();

  std::cout << "Number of FindLoop iterations = " << findIter.reduce () << std::endl;
  std::cout << "Number of LinkUpLoop iterations = " << linkUpIter.reduce () << std::endl;

  std::cout << "MST running time without initialization/destruction: " << runningTime.get () << std::endl;
  std::cout << "Time taken by FindLoop: " << findTimer.get () << std::endl;
  std::cout << "Time taken by LinkUpLoop: " << linkUpTimer.get () << std::endl;

}

class KruskalNaive: public Kruskal {
  protected:

  virtual const std::string getVersion () const { return "Parallel Handwritten Ordered Kruskal"; }

  virtual void runMST (const size_t numNodes, const VecEdge& edges,
      size_t& mstWeight, size_t& totalIter) {

    runMSTnaive (numNodes, edges, mstWeight, totalIter);

  }
};

}// end namespace kruskal



#endif //  KRUSKAL_HAND_H

