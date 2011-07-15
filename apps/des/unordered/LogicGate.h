/** defines the interface for a LogicGate and implements some common functionality -*- C++ -*-
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
 *  Created on: Jun 22, 2011
 *
 * @author M. Amber Hassaan <ahassaan@ices.utexas.edu>
 */

#ifndef LOGICGATE_H_
#define LOGICGATE_H_

#include <string>
#include <iostream>

#include <cstdlib>
#include <cassert>

#include "Galois/Graphs/Graph.h"


#include "Event.h"
#include "AbsSimObject.h"
#include "SimObject.h"
#include "BaseLogicGate.h"
#include "logicDefs.h"
#include "LogicUpdate.h"

/**
 * The Class LogicGate represents an abstract logic gate.
 */
template <typename GraphTy, typename GNodeTy>
class LogicGate: public AbsSimObject<GraphTy, GNodeTy, Event<GNodeTy, LogicUpdate> >, public BaseLogicGate  {

public:
  typedef AbsSimObject< GraphTy, GNodeTy, Event<GNodeTy, LogicUpdate> > AbsSimObj;
  typedef typename Event<GNodeTy, LogicUpdate>::Type EventType;

  LogicGate (size_t numOutputs, size_t numInputs, SimTime delay): AbsSimObj (numOutputs, numInputs), BaseLogicGate (delay)  {}

  LogicGate (const LogicGate<GraphTy, GNodeTy>& that): AbsSimObj (that), BaseLogicGate (that) {}

  /**
   * Gets the index of the input whose name is matching the net
   *
   * @param net name of the net an input of this gate is connected to
   * @return index of that input
   */
  virtual size_t getInputIndex(const std::string& net) const = 0;

protected:
  /**
   * Net name mismatch.
   *
   * @param le the le
   */
  void netNameMismatch (const LogicUpdate& le) const {
    std::cerr << "Received logic update : " << le.toString () << " with mismatching net name, this = " <<
      AbsSimObj::toString () << std::endl;
    exit (-1);
  }

  /**
   * Send events to fanout, which are the out going neighbors in the circuit graph.
   *
   * @param graph: the circuit graph
   * @param myNode: the my node
   * @param inputEvent: the input event
   * @param type: the type
   * @param msg: the logic update
   */
  void sendEventsToFanout(GraphTy& graph, GNodeTy& myNode, const Event<GNodeTy, LogicUpdate>& inputEvent, 
      const EventType& type, const LogicUpdate& msg) const {

    // assert (&myNode == &inputEvent.getRecvObj());

    assert (graph.getData (myNode, Galois::NONE) == this);

    SimTime sendTime = inputEvent.getRecvTime();

    for (typename GraphTy::neighbor_iterator i = graph.neighbor_begin (myNode, Galois::NONE), 
        e = graph.neighbor_end (myNode, Galois::NONE); i != e; ++i) {

      const GNodeTy& dst = *i;

      Event<GNodeTy, LogicUpdate> ne = Event<GNodeTy, LogicUpdate>::makeEvent (myNode, dst, type, msg, sendTime,
          BaseLogicGate::getDelay ());


      SimObject* dstObj = graph.getData (dst, Galois::NONE);
      LogicGate<GraphTy, GNodeTy>* dstGate = dynamic_cast< LogicGate<GraphTy, GNodeTy>* > (dstObj);

      if (dstGate == NULL) {
        std::cerr << "dynamic_cast failed" << std::endl;
        abort ();

      } else {
        const std::string& outNet = getOutputName();
        size_t dstIn = dstGate->getInputIndex(outNet); // get the input index of the net to which my output is connected
        dstGate->recvEvent(dstIn, ne);
      }


    } // end for


  }

};

#endif /* LOGICGATE_H_ */
