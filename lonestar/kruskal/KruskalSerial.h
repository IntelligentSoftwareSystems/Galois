/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
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

#ifndef KRUSKAL_SERIAL_H_
#define KRUSKAL_SERIAL_H_

#include "galois/substrate/CompilerSpecific.h"

#include "Kruskal.h"

namespace kruskal {
class KruskalSerial: public Kruskal {
protected:
// #define EDGE_FRAC  (4/3);

  virtual const std::string getVersion () const { return "Serial Ordered Kruskal"; }


  GALOIS_ATTRIBUTE_PROF_NOINLINE static void mstFunc (const Edge& e, VecRep& repVec, size_t& mstSum) {
      int rep1 = findPCiter_int (e.src, repVec);
      int rep2 = findPCiter_int (e.dst, repVec);

      if (rep1 != rep2) {
        unionByRank_int (rep1, rep2, repVec);

        mstSum += e.weight;
      }
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE static void filterFunc (const Edge& e, VecRep& repVec, VecEdge& remaining) {
      int rep1 = findPCiter_int (e.src, repVec);
      int rep2 = findPCiter_int (e.dst, repVec);

      if (rep1 != rep2) {
        remaining.push_back (e);
      }
  }

  virtual void runMSTsplit (const size_t numNodes, const VecEdge& in_edges,
      size_t& mstWeight, size_t& totalIter) {

    galois::TimeAccumulator t_run;
    galois::TimeAccumulator t_init;
    galois::TimeAccumulator t_sort;
    galois::TimeAccumulator t_loop;

    t_init.start ();
    VecEdge edges (in_edges);
    VecRep repVec (numNodes, -1);
    t_init.stop ();

    t_run.start ();

    // size_t splitSize = EDGE_FRAC * numNodes;
    size_t splitSize = (4 * numNodes) / 3;

    t_sort.start ();
    VecEdge::iterator splitPoint = edges.begin () + splitSize;

    std::nth_element (edges.begin (), splitPoint, edges.end (), Edge::Comparator ());

    std::sort (edges.begin (), splitPoint, Edge::Comparator ());
    t_sort.stop ();


    size_t mstSum = 0;
    size_t iter = 0;

    t_loop.start ();
    for (VecEdge::const_iterator i = edges.begin (), ei = splitPoint;
        i != ei; ++i) {

      ++iter;
      mstFunc (*i, repVec, mstSum);
    }

    VecEdge remaining;

    for (VecEdge::const_iterator i = splitPoint, ei = edges.end ();
        i != ei; ++i) {
      filterFunc (*i, repVec, remaining);
    }
    t_loop.stop ();

    std::cout << "Number of remaining edges needing to be processed: " << remaining.size () << std::endl;

    t_sort.start ();
    std::sort (remaining.begin (), remaining.end (), Edge::Comparator ());
    t_sort.stop ();

    t_loop.start ();
    for (VecEdge::const_iterator i = remaining.begin (), ei = remaining.end ();
        i != ei; ++i) {

      ++iter;
      mstFunc (*i, repVec, mstSum);
    }
    t_loop.stop ();


    mstWeight = mstSum;
    totalIter = iter;

    t_run.stop ();

    std::cout << "Running time excluding initialization and destruction: " << t_run.get () << std::endl;
    std::cout << "Initialization time: " << t_init.get () << std::endl;
    std::cout << "Sorting time: " << t_sort.get () << std::endl;
    std::cout << "Loop time: " << t_loop.get () << std::endl;

  }

  virtual void runMSTsimple (const size_t numNodes, const VecEdge& in_edges,
      size_t& mstWeight, size_t& totalIter) {


    galois::StatTimer t_run("Running time excluding initialization & destruction: ");
    galois::StatTimer t_init("initialization time: ");
    galois::StatTimer t_sort("serial sorting time: ");
    galois::StatTimer t_loop("serial loop time: ");

    t_init.start ();
    VecRep repVec (numNodes, -1);
    VecEdge edges (in_edges);
    t_init.stop ();



    t_run.start ();

    t_sort.start ();
    std::sort (edges.begin (), edges.end (), Edge::Comparator ());
    t_sort.stop ();


    size_t mstSum = 0;
    size_t iter = 0;

    t_loop.start ();
    for (VecEdge::const_iterator i = edges.begin (), ei = edges.end ();
        i != ei; ++i) {

      ++iter;

      int rep1 = findPCiter_int (i->src, repVec);
      int rep2 = findPCiter_int (i->dst, repVec);

      if (rep1 != rep2) {
        unionByRank_int (rep1, rep2, repVec);

        mstSum += i->weight;
      }

    }

    mstWeight = mstSum;
    totalIter = iter;

    t_loop.stop ();

    t_run.stop ();
  }

  virtual void runMST (const size_t numNodes, VecEdge& edges,
      size_t& mstWeight, size_t& totalIter) {

    if (edges.size () >= 2 * numNodes) {
      runMSTsplit (numNodes, edges, mstWeight, totalIter);

    } else {
      runMSTsimple (numNodes, edges, mstWeight, totalIter);
    }


  }

};

} // end namespace kruskal
#endif // KRUSKAL_SERIAL_H_

