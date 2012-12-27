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

#ifndef KRUSKAL_RECURSIVE_H
#define KRUSKAL_RECURSIVE_H


#include "Kruskal.h"
#include "KruskalParallel.h"

namespace kruskal {


void runMSTrecursive (const size_t numNodes, const VecEdge& edges,
    size_t& mstWeight, size_t& totalIter) {

  totalIter = 0;
  mstWeight = 0;

  Galois::TimeAccumulator runningTime;
  Galois::TimeAccumulator partitionTimer;
  Galois::TimeAccumulator findTimer;
  Galois::TimeAccumulator linkUpTimer;
  Galois::TimeAccumulator filterTimer;

  Accumulator findIter;
  Accumulator linkUpIter;



  VecRep_ty repVec (numNodes, -1);
  VecAtomicCtxPtr repOwnerCtxVec (numNodes, AtomicCtxPtr (NULL));



  Galois::preAlloc (16*Galois::getActiveThreads ());

  runningTime.start ();


  EdgeCtxWL lighter;
  EdgeWL heavier;
  EdgeWL filterWL;

  typedef std::iterator_traits<VecEdge::iterator>::difference_type Dist_ty;


  Dist_ty splitPoint = std::max (numNodes/4, size_t (1)); 
  size_t lowThresh = std::max (splitPoint/4, Dist_ty (1));

  partitionTimer.start ();
  partition_edges (edges.begin (), edges.end (), splitPoint,  lighter, heavier);
  partitionTimer.stop ();


  size_t innerRound = 0;
  size_t outerRound = 0;
  size_t numUnions = 0;
  bool lastRound = false;

  Accumulator mstSum;

  EdgeCtxWL* const tmp = new EdgeCtxWL ();
  EdgeCtxWL* nextWL = &lighter;
  EdgeCtxWL* currWL = tmp;

  while (!nextWL->empty_all ()) {

    ++outerRound;

    // presort (*nextWL);


    while (true) {
      ++innerRound;
      std::swap (nextWL, currWL);

      nextWL->clear_all ();

      // GaloisRuntime::beginSampling ();
      findTimer.start ();
      Galois::do_all_choice (*currWL,
          FindLoop (repVec, repOwnerCtxVec, findIter),
          "find_loop");
      findTimer.stop ();
      // GaloisRuntime::endSampling ();


      // GaloisRuntime::beginSampling ();
      linkUpTimer.start ();
      Galois::do_all_choice (*currWL,
          LinkUpLoop<false> (repVec, repOwnerCtxVec, *nextWL, mstSum, linkUpIter),
          "link_up_loop");
      linkUpTimer.stop ();
      // GaloisRuntime::endSampling ();

      int u = linkUpIter.reduce () - numUnions;
      numUnions = linkUpIter.reduce ();

      if (!nextWL->empty_all ()) {
        assert (u > 0 && "no unions, no progress?");
      }

      if (nextWL->size_all () < lowThresh) {
        if (!lastRound) {
          break;
        } 
        if (nextWL->empty_all ()) {
          break;
        }
      }
    }

    // GaloisRuntime::beginSampling ();
    filterTimer.start ();
    filterWL.clear_all ();
    Galois::do_all_choice (heavier,
        FilterSelfEdges<EdgeWL> (repVec, filterWL),
        "filter_loop");
    filterTimer.stop ();
    // GaloisRuntime::endSampling ();


    // GaloisRuntime::beginSampling ();
    partitionTimer.start ();
    heavier.clear_all ();
    if (filterWL.size_all () <= size_t (splitPoint)) {
      lastRound = true;
      Galois::do_all_choice (filterWL.begin_all (), filterWL.end_all (),
          FillUp (*nextWL), "fill_up_last");
      
    } else {
      partition_edges (filterWL.begin_all (), filterWL.end_all (), splitPoint, *nextWL, heavier);
    }

    partitionTimer.stop ();
    // GaloisRuntime::endSampling ();



  }

  totalIter += findIter.reduce ();
  mstWeight += mstSum.reduce ();

  runningTime.stop ();

  delete tmp;



  std::cout << "Number of FindLoop iterations = " << findIter.reduce () << std::endl;
  std::cout << "Number of LinkUpLoop iterations = " << linkUpIter.reduce () << std::endl;

  std::cout << "MST running time without initialization/destruction: " << runningTime.get () << std::endl;
  std::cout << "Time taken by FindLoop: " << findTimer.get () << std::endl;
  std::cout << "Time taken by LinkUpLoop: " << linkUpTimer.get () << std::endl;
  std::cout << "Time taken by partitioning Loop: " << partitionTimer.get () << std::endl;
  std::cout << "Time taken by filter Loop: " << filterTimer.get () << std::endl;
  std::cout << "Number of outer rounds: " << outerRound << std::endl;
  std::cout << "Number of inner rounds: " << innerRound << std::endl;


}


class KruskalRecursive: public Kruskal {
  protected:

  virtual const std::string getVersion () const { return "Kruskal with recursive partitioning and filtering"; }

  virtual void runMST (const size_t numNodes, const VecEdge& edges,
      size_t& mstWeight, size_t& totalIter) {

    runMSTrecursive (numNodes, edges, mstWeight, totalIter);

  }
};

}// end namespace kruskal



#endif //  KRUSKAL_RECURSIVE_H

