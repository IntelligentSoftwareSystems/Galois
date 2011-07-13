/** DES serial unordered version -*- C++ -*-
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



#ifndef _DES_UNORDERED_SERIAL_H_
#define _DES_UNORDERED_SERIAL_H_

#include <deque>

#include <cassert>

#include "DESabstractMain.h"

class DESunorderedSerial: public DESabstractMain {

  virtual bool isSerial () const { return true; }

  /**
   * Run loop.
   * Does not use GaloisRuntime or Galois worklists
   *
   * To ensure uniqueness of items on the worklist, we keep a list of boolean flags for each node,
   * which indicate whether the node is on the worklist. When adding a node to the worklist, the
   * flag corresponding to a node is set to True if it was previously False. The flag reset to False
   * when the node is removed from the worklist. This list of flags provides a cheap way of
   * implementing set semantics.
   *
   */

  virtual void runLoop (const SimInit<Graph, GNode>& simInit) {
    std::deque<GNode> worklist (simInit.getInputNodes ().begin (), simInit.getInputNodes ().end ());

    std::vector<bool> onWlFlags (simInit.getNumNodes (), false);

    // set onWlFlags for input objects
    for (std::vector<GNode>::const_iterator i = simInit.getInputNodes ().begin (), ei = simInit.getInputNodes ().end ();
        i != ei; ++i) {
      SimObject* srcObj = graph.getData (*i, Galois::NONE);
      onWlFlags[srcObj->getId ()] = true;
    }

    size_t numEvents = 0;
    size_t numIter = 0;
    while (!worklist.empty ()) {

      GNode activeNode = worklist.front ();
      worklist.pop_front ();

      SimObject* srcObj = graph.getData (activeNode, Galois::NONE);

      numEvents += srcObj->simulate(graph, activeNode);


      for (Graph::neighbor_iterator i = graph.neighbor_begin (activeNode, Galois::NONE), ei =
          graph.neighbor_end (activeNode, Galois::NONE); i != ei; ++i) {
        const GNode& dst = *i;

        SimObject* dstObj = graph.getData (dst, Galois::NONE);

        dstObj->updateActive ();

        if (dstObj->isActive ()) {
          if (!onWlFlags[dstObj->getId ()]) {
            // set the flag to indicate presence on the worklist
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
