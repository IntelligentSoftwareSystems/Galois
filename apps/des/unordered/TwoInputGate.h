/** TwoInputGate: represents a general gate with two inputs and one output -*- C++ -*-
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


#ifndef TWOINPUTGATE_H_
#define TWOINPUTGATE_H_


#include <string>

#include <cassert>

#include "Event.h"
#include "LogicGate.h"
#include "LogicFunctions.h"
#include "LogicUpdate.h"
#include "BaseTwoInputGate.h"

/**
 * The Class TwoInputGate.
 */
template <typename GraphTy, typename GNodeTy>
class TwoInputGate: public LogicGate<GraphTy, GNodeTy>, public BaseTwoInputGate {

private:
  typedef typename LogicGate<GraphTy, GNodeTy>::AbsSimObj AbsSimObj;

  /**
   * The functor that computes a value for the output of the gate
   * when provided with the values of the input
   */

  const TwoInputFunc& func;

public:
  /**
   * Instantiates a new two input gate.
   *
   * @param func: functor implementing the function from inputs to outputs
   * @param outputName the output name
   * @param input1Name the input 1 name
   * @param input2Name the input 2 name
   * @param delay the delay
   */
  TwoInputGate(const TwoInputFunc& func, const std::string& outputName, const std::string& input1Name, const std::string& input2Name,
      const SimTime& delay)
    : LogicGate<GraphTy, GNodeTy> (1, 2, delay), BaseTwoInputGate (outputName, input1Name, input2Name), func (func) {}

  TwoInputGate (const TwoInputGate<GraphTy, GNodeTy>& that)
    : LogicGate<GraphTy, GNodeTy> (that), BaseTwoInputGate (that), func(that.func) {}

  virtual TwoInputGate<GraphTy, GNodeTy>* clone () const {
    return new TwoInputGate<GraphTy, GNodeTy> (*this);
  }





protected:

  /**
   * compute the new value of the output by calling the functor
   */
  virtual LogicVal evalOutput () const {
    return func (getInput1Val (), getInput2Val ());
  }

  /**
   *
   * The basic idea is to find which input is updated by the LogicUpdate stored in the event. 
   * Then after updating the input, the output is evaluated, and the updated value of the output
   * is sent as a LogicUpdate to all the out-going neighbors to which the output feeds
   *
   */
  virtual void execEvent (GraphTy& graph, GNodeTy& myNode, const Event<GNodeTy, LogicUpdate>& event) {

    if (event.getType() == Event<GNodeTy, LogicUpdate>::NULL_EVENT) {
      // send out null messages

      sendEventsToFanout(graph, myNode, event, Event<GNodeTy, LogicUpdate>::NULL_EVENT, LogicUpdate ());

    } else {
      // update the inputs of fanout gates
      const LogicUpdate& lu =  event.getAction();

      if (input1Name == (lu.getNetName())) {
        input1Val = lu.getNetVal();

      } else if (input2Name == (lu.getNetName())) {
        input2Val = lu.getNetVal();

      } else {
        LogicGate<GraphTy, GNodeTy>::netNameMismatch(lu);
      }

      // output has been changed
      // update output immediately
      // generate events to send to all fanout gates to update their inputs afer delay
      LogicVal newOutput = evalOutput();
      this->outputVal = newOutput;

      LogicUpdate drvFanout(outputName, newOutput);

      sendEventsToFanout(graph, myNode, event, Event<GNodeTy, LogicUpdate>::REGULAR_EVENT, drvFanout);

    }

  }




public:
  /**
   * Checks for input name.
   *
   * @param net the net
   * @return true, if this gate has an input with the name net
   */
  virtual bool hasInputName(const std::string& net) const {
    return BaseTwoInputGate::hasInputName (net);
  }

  /**
   * Checks for output name.
   *
   * @param net the net
   * @return true, if this gate has an output with the name net
   */
  virtual bool hasOutputName(const std::string& net) const {
    return BaseTwoInputGate::hasOutputName (net);
  }

  /**
   * Gets the output name.
   *
   * @return the output name
   */
  virtual const std::string& getOutputName() const {
    return BaseTwoInputGate::getOutputName ();
  }

  /**
   *
   * @param net name of the wire to query
   *
   * @returns 0 if the first input has the same name as 'net'
   *  and 1 if the 2nd input has the same name as 'net'
   *  otherwise throws and error
   */

  virtual size_t getInputIndex(const std::string& net) const {
    if (this->input2Name == (net)) {
      return 1;

    } else if (this->input1Name == (net)) {
      return 0;

    } else {
      abort ();
      return -1; // error
    }
  }

  /**
   * The name of the gate depends on the functionality it's implementing
   */
  virtual const std::string getGateName () const {
    return func.toString ();
  }

  // for debugging
  virtual const std::string toString() const {
    std::ostringstream ss;
    ss << AbsSimObj::toString () << getGateName () << ": " << BaseTwoInputGate::toString () 
       << " delay = " << BaseLogicGate::getDelay ();
    return ss.str ();
  }
};

#endif /* TWOINPUTGATE_H_ */
