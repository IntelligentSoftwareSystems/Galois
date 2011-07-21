/** OneInputGate: a logic gate with one input and one output -*- C++ -*-
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


#ifndef ONEINPUTGATE_H_
#define ONEINPUTGATE_H_

#include <string>

#include <cassert>

#include "Event.h"
#include "LogicGate.h"
#include "LogicFunctions.h"
#include "LogicUpdate.h"
#include "BaseOneInputGate.h"

/**
 * The Class OneInputGate.
 */
class OneInputGate: public LogicGate, public BaseOneInputGate {

private:

  /**
   * a functor which computes an output LogicVal when
   * provided an input LogicVal
   */
  const OneInputFunc& func;

public:
  /**
   * Instantiates a new one input gate.
   *
   * @param id
   * @param func: the functor providing mapping from input values to output values
   * @param outputName the output name
   * @param inputName the input name
   * @param delay the delay
   */
  OneInputGate(size_t id, const OneInputFunc& func, const std::string& outputName, const std::string& inputName, const SimTime& delay)
    : LogicGate (id, 1, 1, delay), BaseOneInputGate (outputName, inputName), func (func) {}

  OneInputGate (const OneInputGate& that)
    : LogicGate (that), BaseOneInputGate (that), func(that.func) {}

  virtual OneInputGate* clone () const {
    return new OneInputGate (*this);
  }



  /**
   *
   * execEvent follows the same logic as TwoInputGate::execEvent()
   *
   */

protected:

  /**
   * calls the functor to compute output value
   */
  virtual LogicVal evalOutput () const {
    return func (getInputVal());
  }

  virtual void execEvent (Graph& graph, GNode& myNode, const EventTy& event) {

    if (event.getType() == EventTy::NULL_EVENT) {
      // send out null messages
      sendEventsToFanout(graph, myNode, event, EventTy::NULL_EVENT, LogicUpdate ());

    } else {
      // update the inputs of fanout gates
      const LogicUpdate& lu =  event.getAction();
      if (inputName == (lu.getNetName())) {
        inputVal = lu.getNetVal();
      } else {
        LogicGate::netNameMismatch(lu);
      }

      // output has been changed
      // update output immediately
      // generate events to send to all fanout gates to update their inputs afer delay
      LogicVal newOutput = evalOutput();
      this->outputVal = newOutput;

      LogicUpdate drvFanout(outputName, newOutput);

      sendEventsToFanout(graph, myNode, event, EventTy::REGULAR_EVENT, drvFanout);

    }

  }


public:
  /**
   * Checks for input name.
   *
   * @param net the wire name
   * @return true, if input is connected to net
   */
  virtual bool hasInputName(const std::string& net) const {
    return BaseOneInputGate::hasInputName (net);
  }

  /**
   * Checks for output name.
   *
   * @param net the wire name
   * @return true, if this gate's output is connected to net
   *
   */
  virtual bool hasOutputName(const std::string& net) const {
    return BaseOneInputGate::hasOutputName (net);
  }

  /**
   * Gets the output name.
   *
   * @return the output name
   */
  virtual const std::string& getOutputName() const {
    return BaseOneInputGate::getOutputName ();
  }


  /**
   * name depends on the logic function this gate is implementing
   */
  virtual const std::string getGateName () const {
    return func.toString ();
  }

  /* 
   * @see LogicGate::getInputIndex()
   */
  virtual size_t getInputIndex(const std::string& net) const {
    if (this->inputName == (net)) {
      return 0; // since there is only one input
    }
    abort ();
    return -1; // error
  }

  /** 
   * A string representation
   */
  virtual const std::string toString() const {
    std::ostringstream ss;
    ss << AbstractSimObject::toString () << getGateName () << ": " << BaseOneInputGate::toString () 
       << " delay = " << BaseLogicGate::getDelay ();
    return ss.str ();
  }



};

#endif /* ONEINPUTGATE_H_ */
