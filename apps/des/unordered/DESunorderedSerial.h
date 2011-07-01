/*
 * DESunorderedSerial.h
 *
 *  Created on: Jun 24, 2011
 *      Author: amber
 */

#ifndef _DES_UNORDERED_SERIAL_H_
#define _DES_UNORDERED_SERIAL_H_

#include <deque>

#include <cassert>

#include "AbstractDESmain.h"

class DESunorderedSerial: public AbstractDESmain {

  virtual bool isSerial () const { return true; }

  /**
   * Run loop.
   *
   */
  virtual void runLoop (const SimInit<Graph, GNode>& simInit) {
    std::deque<GNode> worklist (simInit.getInputNodes ().begin (), simInit.getInputNodes ().end ());

    std::vector<bool> onWlFlags (simInit.getNumNodes (), false);

    // set onWlFlags for input objects
    for (std::vector<GNode>::const_iterator i = simInit.getInputNodes ().begin (), ei = simInit.getInputNodes ().end ();
        i != ei; ++i) {
      SimObject* srcObj = graph.getData (*i, Galois::Graph::NONE);
      onWlFlags[srcObj->getId ()] = true;
    }

    size_t numEvents = 0;
    size_t numIter = 0;
    while (!worklist.empty ()) {

      GNode activeNode = worklist.front ();
      worklist.pop_front ();

      SimObject* srcObj = graph.getData (activeNode, Galois::Graph::NONE);

      numEvents += srcObj->simulate(graph, activeNode);


      for (Graph::neighbor_iterator i = graph.neighbor_begin (activeNode, Galois::Graph::NONE), ei =
          graph.neighbor_end (activeNode, Galois::Graph::NONE); i != ei; ++i) {
        const GNode& dst = *i;

        SimObject* dstObj = graph.getData (dst, Galois::Graph::NONE);

        dstObj->updateActive ();

        if (dstObj->isActive ()) {
          if (!onWlFlags[dstObj->getId ()]) {
            onWlFlags[dstObj->getId ()] = true;
            worklist.push_back (dst);
          }
        }
      }

      srcObj->updateActive();
      if (srcObj->isActive()) {
        worklist.push_back (activeNode);

      } else { 
        // reset the flag to indicate absence on the worklist
        onWlFlags[srcObj->getId ()] = false;
      }

      ++numIter;

    }


    std::cout << "Simulation ended" << std::endl;
    std::cout << "Number of events processed = " << numEvents << " Iterations = " << numIter << std::endl;
  }

};

#endif // _DES_UNORDERED_SERIAL_H_
