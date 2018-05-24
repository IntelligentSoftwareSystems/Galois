#ifndef _DES_UNORDERED_SERIAL_H_
#define _DES_UNORDERED_SERIAL_H_

#include <deque>
#include <functional>

#include <cassert>

#include "DESunorderedBase.h"

namespace des_unord {


class DESunorderedSerial: public des_unord::DESunorderedBase {

  virtual std::string getVersion () const { return "Unordered (Chandy-Misra) serial"; }

  /**
   * Run loop.
   * Does not use galois::runtime or Galois worklists
   *
   * To ensure uniqueness of items on the workList, we keep a list of boolean flags for each node,
   * which indicate whether the node is on the workList. When adding a node to the workList, the
   * flag corresponding to a node is set to True if it was previously False. The flag reset to False
   * when the node is removed from the workList. This list of flags provides a cheap way of
   * implementing set semantics.
   *
   */

  virtual void runLoop (const SimInit_ty& simInit, Graph& graph) {

    std::deque<GNode> workList;
    std::vector<bool> onWLflags;

    initWorkList (graph, workList, onWLflags);

    size_t maxPending = 0;
    size_t numEvents = 0;
    size_t numIter = 0;

    while (!workList.empty ()) {

      GNode activeNode = workList.front ();
      workList.pop_front ();

      SimObj_ty* srcObj = static_cast<SimObj_ty*> (graph.getData (activeNode, galois::MethodFlag::UNPROTECTED));

      maxPending = std::max (maxPending, srcObj->numPendingEvents ());

      numEvents += srcObj->simulate(graph, activeNode);


      for (Graph::edge_iterator i = graph.edge_begin (activeNode, galois::MethodFlag::UNPROTECTED)
          , ei = graph.edge_end (activeNode, galois::MethodFlag::UNPROTECTED); i != ei; ++i) {

        GNode dst = graph.getEdgeDst(i);
        SimObj_ty* dstObj = static_cast<SimObj_ty*> (graph.getData (dst, galois::MethodFlag::UNPROTECTED));

        if (dstObj->isActive ()) {
          if (!onWLflags[dstObj->getID ()]) {
            // set the flag to indicate presence on the workList
            onWLflags[dstObj->getID ()] = true;
            workList.push_back (dst);
          }
        }
      }

      if (srcObj->isActive()) {
        workList.push_back (activeNode);

      } else { 
        // reset the flag to indicate absence on the workList
        onWLflags[srcObj->getID ()] = false;
      }

      ++numIter;

    }


    std::cout << "Simulation ended" << std::endl;
    std::cout << "Number of events processed = " << numEvents << " Iterations = " << numIter << std::endl;
    std::cout << "Max size of pending events = " << maxPending << std::endl;

  }

};

} // end namspace des_unord
#endif // _DES_UNORDERED_SERIAL_H_
