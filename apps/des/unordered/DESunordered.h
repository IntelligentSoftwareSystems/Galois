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
#include "Galois/Accumulator.h"
#include "Galois/Atomic.h"
#include "Galois/Runtime/WorkList.h"
#include "Galois/Runtime/ll/gio.h"

#include "DESunorderedBase.h"



namespace des_unord {
static const bool DEBUG = false;


class DESunordered: public DESunorderedBase {
  typedef Galois::GAccumulator<size_t> Accumulator;
  typedef Galois::GReduceMax<size_t> ReduceMax;
  typedef Galois::GAtomicPadded<bool> AtomicBool;
  typedef std::vector<AtomicBool> VecAtomicBool;



  /**
   * contains the loop body, called 
   * by @see for_each
   */
  struct Process {
    Graph& graph;
    VecAtomicBool& onWLflags;
    Accumulator& numEvents;
    Accumulator& numIter;
    ReduceMax& maxPending;


    Process (
        Graph& graph,
        VecAtomicBool& onWLflags,
        Accumulator& numEvents,
        Accumulator& numIter,
        ReduceMax& maxPending)
      : 
        graph (graph), 
        onWLflags (onWLflags), 
        numEvents (numEvents), 
        numIter (numIter), 
        maxPending (maxPending) 
    {}


    void lockNeighborhood (GNode& activeNode) {
        // acquire locks on neighborhood: one shot
        graph.getData (activeNode, Galois::CHECK_CONFLICT);

        // for (Graph::edge_iterator i = graph.edge_begin (activeNode, Galois::CHECK_CONFLICT)
            // , ei = graph.edge_end (activeNode, Galois::CHECK_CONFLICT); i != ei; ++i) {
          // GNode dst = graph.getEdgeDst (i);
          // graph.getData (dst, Galois::CHECK_CONFLICT);
        // }

    }


    /**
     *
     * Called by @see GaloisRuntime::for_each during
     * every iteration
     *
     * @param activeNode: the current active element
     * @param lwl: the worklist type
     */

    template <typename WL>
    void operator () (GNode& activeNode, WL& lwl) {

        lockNeighborhood (activeNode);

        SimObj_ty* srcObj = static_cast<SimObj_ty*> (graph.getData (activeNode, Galois::NONE));
        // should be past the fail-safe point by now

        if (DEBUG) {
          GALOIS_DEBUG ("processing : %s\n", srcObj->str ().c_str ());
        }

        maxPending.update (srcObj->numPendingEvents ());

        size_t proc = srcObj->simulate(graph, activeNode); // number of events processed
        numEvents += proc;

        for (Graph::edge_iterator i = graph.edge_begin (activeNode, Galois::NONE)
            , ei = graph.edge_end (activeNode, Galois::NONE); i != ei; ++i) {

          const GNode dst = graph.getEdgeDst(i);
          SimObj_ty* dstObj = static_cast<SimObj_ty*> (graph.getData (dst, Galois::NONE));

          if (dstObj->isActive () 
              && !bool (onWLflags [dstObj->getID ()])
              && onWLflags[dstObj->getID ()].cas (false, true)) {
            if (DEBUG) {
              GALOIS_DEBUG ("Added %d neighbor: %s\n", 
                  bool (onWLflags[dstObj->getID ()]), dstObj->str ().c_str ());
            }

            lwl.push (dst);

          }


        }
        

        if (srcObj->isActive()) {
          lwl.push (activeNode);
          
          if (DEBUG) {
            GALOIS_DEBUG ("Added %d self: %s\n" 
                , bool (onWLflags[srcObj->getID ()]), srcObj->str ().c_str ());
          }

        } else {
          onWLflags[srcObj->getID ()] = false;

          if (DEBUG) {
            GALOIS_DEBUG ("not adding %d self: %s\n", 
                bool (onWLflags[srcObj->getID ()]), srcObj->str ().c_str ());
          }
        }
        

        numIter += 1;


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
   */
  virtual void runLoop (const SimInit_ty& simInit, Graph& graph) {

    std::vector<GNode> initialActive;
    VecAtomicBool onWLflags;

    initWorkList (graph, initialActive, onWLflags);


    Accumulator numEvents;
    Accumulator numIter;
    ReduceMax maxPending;



    Process p(graph, onWLflags, numEvents, numIter, maxPending);

    typedef GaloisRuntime::WorkList::dChunkedFIFO<CHUNK_SIZE, GNode> WL_ty;
    // typedef GaloisRuntime::WorkList::GFIFO<GNode> WL_ty;

    Galois::for_each<WL_ty>(initialActive.begin (), initialActive.end (), p);

    std::cout << "Number of events processed = " << numEvents.reduce () << std::endl;
    std::cout << "Number of iterations performed = " << numIter.reduce () << std::endl;
    std::cout << "Maximum size of pending events = " << maxPending.reduce() << std::endl;
  }

  void checkPostState (Graph& graph, VecAtomicBool& onWLflags) {
    for (Graph::iterator n = graph.begin (),
        endn = graph.end (); n != endn; ++n) {

      SimObj_ty* so = static_cast<SimObj_ty*> (graph.getData (*n, Galois::NONE));
      if (so->isActive ()) {
        std::cout << "ERROR: Found Active: " << so->str () << std::endl
          << "onWLflags = " << onWLflags[so->getID ()] << ", numPendingEvents = " << so->numPendingEvents () 
          << std::endl;
      }
    }

  }

  virtual std::string getVersion () const { return "Unordered (Chandy-Misra) parallel"; }
};
} // end namespace des_unord


#endif // _DES_UNORDERED_H_

