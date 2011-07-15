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
#include "Galois/util/Accumulator.h"

#include "DESabstractMain.h"


static const bool DEBUG = false;

using Galois::PerCPUcounter;

class DESunordered: public DESabstractMain {

  /**
   * contains the loop body, called 
   * by @see for_each
   */
  struct process {
    Graph& graph;
    std::vector<unsigned int>& onWlFlags;
    PerCPUcounter& numEvents;
    PerCPUcounter& numIter;

    
    process (
    Graph& graph,
    std::vector<unsigned int>& onWlFlags,
    PerCPUcounter& numEvents,
    PerCPUcounter& numIter)
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
        SimObject* srcObj = graph.getData (activeNode, Galois::CHECK_CONFLICT);

        // acquire locks on neighborhood: one shot
        for (Graph::neighbor_iterator i = graph.neighbor_begin (activeNode, Galois::CHECK_CONFLICT)
            , ei = graph.neighbor_end (activeNode, Galois::CHECK_CONFLICT); i != ei; ++i) {
          // const GNode& dst = *i;
          // SimObject* dstObj = graph.getData (dst, Galois::CHECK_CONFLICT);
        }



        // should be past the fail-safe point by now

        if (DEBUG) {
          // DEBUG
          printf ("%d processing : %s\n", ThreadPool::getMyID (), srcObj->toString ().c_str ());
        }

        int proc = srcObj->simulate(graph, activeNode); // number of events processed
        numEvents.get () += proc;

        for (Graph::neighbor_iterator i = graph.neighbor_begin (activeNode, Galois::NONE)
            , ei = graph.neighbor_end (activeNode, Galois::NONE); i != ei; ++i) {
          const GNode& dst = *i;

          SimObject* dstObj = graph.getData (dst, Galois::NONE);

          dstObj->updateActive ();

          if (dstObj->isActive ()) {

            if (onWlFlags[dstObj->getId ()] == 0) {
              if (DEBUG) {
                // DEBUG
                printf ("%d Added %d neighbor: %s\n" , ThreadPool::getMyID (), onWlFlags[dstObj->getId ()], dstObj->toString ().c_str ());
              }
              onWlFlags[dstObj->getId ()] = 1;
              lwl.push (dst);

            }

          }
        }

        srcObj->updateActive();

        if (srcObj->isActive()) {
          lwl.push (activeNode);
          
          if (DEBUG) {
            //DEBUG
            printf ("%d Added %d self: %s\n" , ThreadPool::getMyID (), onWlFlags[srcObj->getId ()], srcObj->toString ().c_str ());
          }

        }
        else {
          onWlFlags[srcObj->getId ()] = 0;
          if (DEBUG) {
            //DEBUG
            printf ("%d not adding %d self: %s\n" , ThreadPool::getMyID (), onWlFlags[srcObj->getId ()], srcObj->toString ().c_str ());
          }
        }

        ++(numIter.get ());

    }
  };

  /**
   * Run loop.
   *
   * Galois worklists, currently, do not support set semantics, therefore, duplicates can be present on the workset. 
   * To ensure uniqueness of items on the worklist, we keep a list of boolean flags for each node,
   * which indicate whether the node is on the worklist. When adding a node to the worklist, the
   * flag corresponding to a node is set to True if it was previously False. The flag reset to False
   * when the node is removed from the worklist. This list of flags provides a cheap way of
   * implementing set semantics.
   *
   * Normally, one would use a vector<bool>, but std::vector<bool> uses bit vector implementation,
   * where different indices i!=j share a common memory word. To protect against concurrent
   * accesses, we would need to acquire abstract locks corresponding to the memory word rather than acquiring locks
   * on locations iand j. This requires knowledge of std::vector<bool> implementation. Instead, we use a
   * std::vector<unsigned int> so that each index i!=j is stored in a separate word and we can
   * acquire lock on i safely.
   */
  virtual void runLoop (const SimInit<Graph, GNode>& simInit) {
    const std::vector<GNode>& initialActive = simInit.getInputNodes();


    std::vector<unsigned int> onWlFlags (simInit.getNumNodes (), 0);
    // set onWlFlags for input objects
    for (std::vector<GNode>::const_iterator i = simInit.getInputNodes ().begin (), ei = simInit.getInputNodes ().end ();
        i != ei; ++i) {
      SimObject* srcObj = graph.getData (*i, Galois::NONE);
      onWlFlags[srcObj->getId ()] = 1;
    }



    PerCPUcounter numEvents;
    PerCPUcounter numIter;

    process p(graph, onWlFlags, numEvents, numIter);

    Galois::for_each < GaloisRuntime::WorkList::FIFO<GNode> > (initialActive.begin (), initialActive.end (), p);

    std::cout << "Number of events processed = " << numEvents.get () << std::endl;
    std::cout << "Number of iterations performed = " << numIter.get () << std::endl;
  }

  virtual bool isSerial () const { return false; }
};


#endif // _DES_UNORDERED_H_

