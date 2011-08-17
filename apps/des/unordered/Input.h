/** Input represents an input port in the circuit -*- C++ -*-
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

#ifndef INPUT_H_
#define INPUT_H_

#include <string>
#include <cassert>

#include "SimGate.h"
#include "BasicPort.h"

// may as well be inherited from OneInputGate
class Input: public SimGate {

public: 
  /**
   * Instantiates a new Input.
   *
   * @param id
   * @param outputName the output name
   * @param inputName the Input name
   */
  Input(size_t id, BasicPort& impl)
    : SimGate (id, impl) {}


  virtual Input* clone () const {
    return new Input (*this);
  }

  virtual BasicPort& getImpl () const {
    BasicPort* ptr = dynamic_cast<BasicPort*> (&SimGate::getImpl ());
    assert (ptr != NULL);
    return *ptr;
  }
  
  /**
   * A string representation
   */
  virtual const std::string toString () const {
    std::ostringstream ss;
    ss << AbstractSimObject::toString () << ": " << "Input " << getImpl ().toString ();
    return ss.str ();
  }

protected:
  /**
   * Sends a copy of event at the input to all the outputs in the circuit
   * @see OneInputGate::execEvent()
   *
   */
  virtual void execEvent(Graph& graph, GNode& myNode, const EventTy& event) {

    if (event.getType () == EventTy::NULL_EVENT) {
      // send out null messages

      SimGate::execEvent(graph, myNode, event); // same functionality as OneInputGate

    } else {

     const LogicUpdate& lu = event.getAction ();
      if (getImpl().getOutputName () == lu.getNetName()) {
        getImpl().setInputVal (lu.getNetVal());
        getImpl().setOutputVal (lu.getNetVal());

        LogicUpdate drvFanout (getImpl ().getOutputName (), getImpl ().getOutputVal ());

        sendEventsToFanout (graph, myNode, event, EventTy::REGULAR_EVENT, drvFanout);
      } else {
        getImpl ().netNameMismatch (lu);
      }

    }

  }

};
#endif /* INPUT_H_ */
