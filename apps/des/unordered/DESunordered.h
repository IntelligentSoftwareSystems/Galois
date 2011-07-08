/** DES unordered Galois version -*- C++ -*-
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
 * @author M. Amber Hassaan <ahassaan@ices.utexas.edu>
 */


#ifndef _DES_UNORDERED_H_
#define _DES_UNORDERED_H_

#include "Galois/Galois.h"
#include "Galois/Runtime/WorkList.h"
#include "Galois/util/Atomic.h"

#include "DESabstractMain.h"

typedef Galois::GAtomic<int> AtomicInteger;

class DESunordered: public DESabstractMain {


  /**
   * contains the loop body, called 
   * by @see for_each
   */
  struct process {
    Graph& graph;
    std::vector<bool>& onWlFlags;
    AtomicInteger& numEvents;
    AtomicInteger& numIter;

    
    process (
    Graph& graph,
    std::vector<bool>& onWlFlags,
    AtomicInteger& numEvents,
    AtomicInteger& numIter)
      : graph (graph), onWlFlags (onWlFlags), numEvents (numEvents), numIter (numIter) {}



    /**
     *
     * Called by @see GaloisRuntime::for_each during
     * every iteration
     *
     * @param activeNode: the current active element
     * @param lwl: the worklist type
     */

    template <typename ContextTy>
    void operator () (GNode& activeNode, ContextTy& lwl) {
        SimObject* srcObj = graph.getData (activeNode, Galois::Graph::CHECK_CONFLICT);

        // acquire locks on neighborhood: one shot
        for (Graph::neighbor_iterator i = graph.neighbor_begin (activeNode, Galois::Graph::CHECK_CONFLICT)
            , ei = graph.neighbor_end (activeNode, Galois::Graph::CHECK_CONFLICT); i != ei; ++i) {
          // const GNode& dst = *i;
          // SimObject* dstObj = graph.getData (dst, Galois::Graph::CHECK_CONFLICT);
        }



        // should be past the fail-safe point by now


        int proc = srcObj->simulate(graph, activeNode); // number of events processed
        numEvents += proc;

        for (Graph::neighbor_iterator i = graph.neighbor_begin (activeNode, Galois::Graph::NONE)
            , ei = graph.neighbor_end (activeNode, Galois::Graph::NONE); i != ei; ++i) {
          const GNode& dst = *i;

          SimObject* dstObj = graph.getData (dst, Galois::Graph::NONE);

          dstObj->updateActive ();

          if (dstObj->isActive ()) {

            if (!onWlFlags[dstObj->getId ()]) {
              onWlFlags[dstObj->getId ()] = true;
              lwl.push (dst);
            }

          }
        }

        srcObj->updateActive();

        if (srcObj->isActive()) {
          lwl.push (activeNode);
        }
        else {
          onWlFlags[srcObj->getId ()] = false;
        }

        ++numIter;

    }
  };

  /**
   * Run loop.
   *
   */
  virtual void runLoop (const SimInit<Graph, GNode>& simInit) {
    const std::vector<GNode>& initialActive = simInit.getInputNodes();


    std::vector<bool> onWlFlags (simInit.getNumNodes (), false);
    // set onWlFlags for input objects
    for (std::vector<GNode>::const_iterator i = simInit.getInputNodes ().begin (), ei = simInit.getInputNodes ().end ();
        i != ei; ++i) {
      SimObject* srcObj = graph.getData (*i, Galois::Graph::NONE);
      onWlFlags[srcObj->getId ()] = true;
    }



    AtomicInteger numEvents(0);
    AtomicInteger numIter(0);


    process p(graph, onWlFlags, numEvents, numIter);

    Galois::for_each < GaloisRuntime::WorkList::FIFO<GNode> > (initialActive.begin (), initialActive.end (), p);

    std::cout << "Number of events processed = " << (int)numEvents << std::endl;
    std::cout << "Number of iterations performed = " << (int)numIter << std::endl;
  }

  virtual bool isSerial () const { return false; }
};


#endif // _DES_UNORDERED_H_
