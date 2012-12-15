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

#ifndef KRUSKAL_ORDERED_H
#define KRUSKAL_ORDERED_H

#include "Galois/Atomic.h"
#include "Galois/Accumulator.h"
#include "Galois/Runtime/PerThreadWorkList.h"

#include "Kruskal.h"
#include "KruskalParallel.h"

namespace kruskal {


void runKruskalDet (
    EdgeWL& perThrdWL, 
    VecRep& repVec, 
    VecAtomicInt& minEdgeIDVec, 
    const VecEdge& edges, 
    size_t& mstWeight, 
    size_t& totalIter,
    Accumulator& findIter,
    Accumulator& linkUpIter) {

  EdgeWL* nextWL = NULL;
  Accumulator mstSum;
  

  Galois::for_each_ordered (perThrdWL.begin_all (), perThrdWL.end_all (),
        Edge::Comparator (),
        FindLoop (repVec, minEdgeIDVec, edges, findIter),
        LinkUpLoop<true> (repVec, minEdgeIDVec, *nextWL, mstSum, linkUpIter));


  totalIter += findIter.reduce ();
  mstWeight += mstSum.reduce ();


}



class KruskalOrdered: public Kruskal {
  protected:
  // static const double EDGE_FRAC = 4/3;

  virtual const std::string getVersion () const { return "Parallel Ordered Runtime Kruskal"; }

  virtual void runMST (const size_t numNodes, const VecEdge& edges,
      size_t& mstWeight, size_t& totalIter) {

    runMSTparallel (numNodes, edges, mstWeight, totalIter, &runKruskalDet);

  }
};




















}// end namespace kruskal




#endif //  KRUSKAL_ORDERED_H

