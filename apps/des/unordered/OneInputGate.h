/*
 * OneInputGate.h
 *
 *  Created on: Jun 23, 2011
 *      Author: amber
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
template <typename GraphTy, typename GNodeTy>
class OneInputGate: public LogicGate<GraphTy, GNodeTy>, public BaseOneInputGate {

private:
  const OneInputFunc& func;
  typedef typename LogicGate<GraphTy, GNodeTy>::AbsSimObj AbsSimObj;

public:
  /**
   * Instantiates a new one input gate.
   *
   * @param outputName the output name
   * @param inputName the input name
   * @param delay the delay
   */
  OneInputGate(const OneInputFunc& func, const std::string& outputName, const std::string& inputName, const SimTime& delay)
    : LogicGate<GraphTy, GNodeTy> (1, 1, delay), BaseOneInputGate (outputName, inputName), func (func) {}

  OneInputGate (const OneInputGate<GraphTy, GNodeTy>& that)
    : LogicGate<GraphTy, GNodeTy> (that), BaseOneInputGate (that), func(that.func) {}

  virtual OneInputGate<GraphTy, GNodeTy>* clone () const {
    return new OneInputGate<GraphTy, GNodeTy> (*this);
  }



  // override the SimObject methods
  /*
   * execEvent follows the same logic as TwoInputGate.execEvent()
   *
   */

protected:

  virtual LogicVal evalOutput () const {
    return func (getInputVal());
  }

  /* (non-Javadoc)
   * @see des.unordered.SimObject#execEvent(galois.objects.graph.GNode, des.unordered.Event)
   */
  virtual void execEvent (GraphTy& graph, GNodeTy& myNode, const Event<GNodeTy, LogicUpdate>& event) {

    if (event.getType() == Event<GNodeTy, LogicUpdate>::NULL_EVENT) {
      // send out null messages
      sendEventsToFanout(graph, myNode, event, Event<GNodeTy, LogicUpdate>::NULL_EVENT, LogicUpdate ());

    } else {
      // update the inputs of fanout gates
      LogicUpdate lu =  event.getAction();
      if (inputName == (lu.getNetName())) {
        inputVal = lu.getNetVal();
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
   * @return true, if successful
   */
  virtual bool hasInputName(const std::string& net) const {
    return BaseOneInputGate::hasInputName (net);
  }

  /**
   * Checks for output name.
   *
   * @param net the net
   * @return true, if successful
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


  virtual const std::string getGateName () const {
    return func.toString ();
  }

  /* (non-Javadoc)
   * @see des.unordered.circuitlib.LogicGate#getInputIndex(java.lang.const std::string&)
   */
  virtual size_t getInputIndex(const std::string& net) const {
    if (this->inputName == (net)) {
      return 0; // since there is only one input
    }
    assert (false);
    return -1; // error
  }

  // for debugging
  /* (non-Javadoc)
   * @see des.unordered.SimObject#toString()
   */
  virtual const std::string toString() const {
    std::ostringstream ss;
    ss << AbsSimObj::toString () << getGateName () << ": " << BaseOneInputGate::toString () 
       << " delay = " << BaseLogicGate::getDelay ();
    return ss.str ();
  }



};

#endif /* ONEINPUTGATE_H_ */
