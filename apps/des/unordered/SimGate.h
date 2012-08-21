/** defines the interface for a SimGate and implements some common functionality -*- C++ -*-
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

#include "Galois/Graphs/Graph2.h"


#include "logicDefs.h"
#include "LogicUpdate.h"
#include "LogicGate.h"

#include "Event.h"
#include "AbstractSimObject.h"
#include "SimObject.h"

/**
 * The Class SimGate represents an abstract logic gate.
 */
class SimGate: public AbstractSimObject  {
private:
  LogicGate& impl;

public:
  typedef EventTy::Type EventKind;

  SimGate (size_t id, LogicGate& impl): AbstractSimObject (id, impl.getNumOutputs (), impl.getNumInputs ()), impl(impl)  {}

  SimGate (const SimGate& that): AbstractSimObject (that), impl (that.impl) {}

  virtual SimGate* clone () const {
    return new SimGate (*this);
  }

  virtual LogicGate& getImpl () const {
    return impl;
  }

  static const LogicGate& getImpl (Graph& graph, GNode& src) {
    SimGate* sg = dynamic_cast<SimGate*> (graph.getData (src, Galois::NONE));
    assert  (sg != NULL);
    return sg->getImpl ();
  }

  /**
   * A string representation
   */
  virtual const std::string toString () const {
    std::ostringstream ss;
    ss << AbstractSimObject::toString () << ": " << impl.toString ();
    return ss.str ();
  }


protected:
  virtual void execEvent (Graph& graph, GNode& myNode, const EventTy& event) {

     if (event.getType () == EventTy::NULL_EVENT) {
       // send out null messages
       sendEventsToFanout (graph, myNode, event, EventTy::NULL_EVENT, LogicUpdate ());

     } else {
       // update the inputs of fanout gates
       const LogicUpdate& lu = event.getAction ();

       impl.applyUpdate (lu);

       // output has been changed
       // generate events to send to all fanout gates to update their inputs afer delay

       LogicUpdate drvFanout (impl.getOutputName (), impl.getOutputVal ());

       sendEventsToFanout (graph, myNode, event, EventTy::REGULAR_EVENT, drvFanout);

     }

   }

  /**
   * Net name mismatch.
   *
   * @param le the le
   */

  /**
   * Send events to fanout, which are the out going neighbors in the circuit graph.
   *
   * @param graph: the circuit graph
   * @param myNode: the my node
   * @param inputEvent: the input event
   * @param type: the type
   * @param msg: the logic update
   */
  void sendEventsToFanout(Graph& graph, GNode& myNode, const EventTy& inputEvent, 
      const EventKind& type, const LogicUpdate& msg) {

    // assert (&myNode == &inputEvent.getRecvObj());

    // TODO: fix this code

    SimGate* srcGate = dynamic_cast<SimGate*> (graph.getData (myNode, Galois::NONE));

    assert (srcGate != NULL);
    assert (srcGate == this);

    SimTime sendTime = inputEvent.getRecvTime();

    for (Graph::edge_iterator i = graph.edge_begin (myNode, Galois::NONE), 
        e = graph.edge_end (myNode, Galois::NONE); i != e; ++i) {

      const GNode dst = graph.getEdgeDst(i);
      SimGate* dstGate = dynamic_cast<SimGate*> (graph.getData (dst, Galois::NONE));

      assert (dstGate != NULL);

      EventTy ne = srcGate->makeEvent (srcGate, dstGate, type, msg, sendTime, impl.getDelay ());


      const std::string& outNet = srcGate->getImpl ().getOutputName ();
      size_t dstIn = dstGate->getImpl ().getInputIndex (outNet); // get the input index of the net to which my output is connected
      dstGate->recvEvent(dstIn, ne);


    } // end for


  }

};

#endif /* LOGICGATE_H_ */
