/** DES unordered Galois version -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @author M. Amber Hassaan <ahassaan@ices.utexas.edu>
 */


#ifndef _DES_UNORDERED_H_
#define _DES_UNORDERED_H_

#include "Galois/Galois.h"
#include "Galois/Accumulator.h"
#include "Galois/Atomic.h"
#include "Galois/WorkList/WorkList.h"
#include "Galois/gIO.h"

#include "DESunorderedBase.h"



namespace des_unord {
static const bool DEBUG = false;


class DESunordered: public DESunorderedBase {
  typedef galois::GAccumulator<size_t> Accumulator;
  typedef galois::GReduceMax<size_t> ReduceMax;
  typedef galois::GAtomicPadded<bool> AtomicBool;
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
        graph.getData (activeNode, galois::MethodFlag::WRITE);

        // for (Graph::edge_iterator i = graph.edge_begin (activeNode, galois::MethodFlag::WRITE)
            // , ei = graph.edge_end (activeNode, galois::MethodFlag::WRITE); i != ei; ++i) {
          // GNode dst = graph.getEdgeDst (i);
          // graph.getData (dst, galois::MethodFlag::WRITE);
        // }

    }


    /**
     *
     * Called by @see galois::runtime::for_each during
     * every iteration
     *
     * @param activeNode: the current active element
     * @param lwl: the worklist type
     */

    template <typename WL>
    void operator () (GNode& activeNode, WL& lwl) {

        lockNeighborhood (activeNode);

        SimObj_ty* srcObj = static_cast<SimObj_ty*> (graph.getData (activeNode, galois::MethodFlag::UNPROTECTED));
        // should be past the fail-safe point by now

        if (DEBUG) {
          galois::gDebug("processing : ", srcObj->str ().c_str ());
        }

        maxPending.update (srcObj->numPendingEvents ());

        size_t proc = srcObj->simulate(graph, activeNode); // number of events processed
        numEvents += proc;

        for (Graph::edge_iterator i = graph.edge_begin (activeNode, galois::MethodFlag::UNPROTECTED)
            , ei = graph.edge_end (activeNode, galois::MethodFlag::UNPROTECTED); i != ei; ++i) {

          const GNode dst = graph.getEdgeDst(i);
          SimObj_ty* dstObj = static_cast<SimObj_ty*> (graph.getData (dst, galois::MethodFlag::UNPROTECTED));

          if (dstObj->isActive () 
              && !bool (onWLflags [dstObj->getID ()])
              && onWLflags[dstObj->getID ()].cas (false, true)) {
            if (DEBUG) {
              galois::gDebug ("Added %d neighbor: ", 
                  bool (onWLflags[dstObj->getID ()]), dstObj->str ().c_str ());
            }

            lwl.push (dst);

          }


        }
        

        if (srcObj->isActive()) {
          lwl.push (activeNode);
          
          if (DEBUG) {
            galois::gDebug ("Added %d self: " 
                , bool (onWLflags[srcObj->getID ()]), srcObj->str ().c_str ());
          }

        } else {
          onWLflags[srcObj->getID ()] = false;

          if (DEBUG) {
            galois::gDebug ("not adding %d self: ", 
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

    typedef galois::worklists::dChunkedFIFO<AbstractBase::DEFAULT_CHUNK_SIZE, GNode> WL_ty;
    // typedef galois::runtime::worklists::GFIFO<GNode> WL_ty;

    galois::for_each(initialActive.begin (), initialActive.end (), p, galois::wl<WL_ty>());

    std::cout << "Number of events processed = " << numEvents.reduce () << std::endl;
    std::cout << "Number of iterations performed = " << numIter.reduce () << std::endl;
    std::cout << "Maximum size of pending events = " << maxPending.reduce() << std::endl;
  }

  void checkPostState (Graph& graph, VecAtomicBool& onWLflags) {
    for (Graph::iterator n = graph.begin (),
        endn = graph.end (); n != endn; ++n) {

      SimObj_ty* so = static_cast<SimObj_ty*> (graph.getData (*n, galois::MethodFlag::UNPROTECTED));
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

