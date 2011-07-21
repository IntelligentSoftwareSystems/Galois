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

#include "BaseOneInputGate.h"
#include "LogicFunctions.h"
#include "OneInputGate.h"

// may as well be inherited from OneInputGate
class Input: public OneInputGate {

public: 
  /**
   * Instantiates a new Input.
   *
   * @param id
   * @param outputName the output name
   * @param inputName the Input name
   */
  Input(size_t id, const std::string& outputName, const std::string& inputName)
    : OneInputGate (id, BUF(), outputName, inputName, 0l)
  {}


  Input (const Input & that) 
    : OneInputGate (that) {}


  virtual Input* clone () const {
    return new Input (*this);
  }

  virtual const std::string getGateName() const {
    return "Input";
  }
  
protected:

  /** 
   * @see LogicGate::evalOutput()
   */
  virtual LogicVal evalOutput() const {
    return this->BaseOneInputGate::inputVal;
  }

  /**
   * Sends a copy of event at the input to all the outputs in the circuit
   * @see OneInputGate::execEvent()
   *
   */
  virtual void execEvent(Graph& graph, GNode& myNode, const EventTy& event) {

    if (event.getType () == EventTy::NULL_EVENT) {
      // send out null messages

      OneInputGate::execEvent(graph, myNode, event); // same functionality as OneInputGate

    } else {

     const LogicUpdate& lu = (LogicUpdate) event.getAction ();
      if (this->BaseOneInputGate::outputName == lu.getNetName()) {
        this->BaseOneInputGate::inputVal = lu.getNetVal();

        this->BaseOneInputGate::outputVal = this->BaseOneInputGate::inputVal;

        LogicUpdate drvFanout(this->BaseOneInputGate::outputName, this->BaseOneInputGate::outputVal);

        sendEventsToFanout(graph, myNode, event, EventTy::REGULAR_EVENT, drvFanout);
      } else {
        LogicGate::netNameMismatch(lu);
      }

    }

  }

};
#endif /* INPUT_H_ */
