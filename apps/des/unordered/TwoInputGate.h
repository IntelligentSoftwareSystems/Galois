/*
 * TwoInputGate.h
 *
 *  Created on: Jun 23, 2011
 *      Author: amber
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
  const TwoInputFunc& func;
  typedef typename LogicGate<GraphTy, GNodeTy>::AbsSimObj AbsSimObj;

public:
  /**
   * Instantiates a new one input gate.
   *
   * @param outputName the output name
   * @param inputName the input name
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




  // override the SimObject methods
  /*
   * execEvent follows the same logic as TwoInputGate.execEvent()
   *
   */

protected:

  virtual LogicVal evalOutput () const {
    return func (getInput1Val (), getInput2Val ());
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
   * @return true, if successful
   */
  virtual bool hasInputName(const std::string& net) const {
    return BaseTwoInputGate::hasInputName (net);
  }

  /**
   * Checks for output name.
   *
   * @param net the net
   * @return true, if successful
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
  /* (non-Javadoc)
   * @see des.unordered.circuitlib.LogicGate#getInputIndex(java.lang.const std::string&)
   */
  virtual size_t getInputIndex(const std::string& net) const {
    if (this->input2Name == (net)) {
      return 1;

    } else if (this->input1Name == (net)) {
      return 0;

    } else {
      assert (false);
      return -1; // error
    }
  }

  virtual const std::string getGateName () const {
    return func.toString ();
  }
  // for debugging
  /* (non-Javadoc)
   * @see des.unordered.SimObject#toString()
   */
  virtual const std::string toString() const {
    std::ostringstream ss;
    ss << AbsSimObj::toString () << getGateName () << ": " << BaseTwoInputGate::toString () 
       << " delay = " << BaseLogicGate::getDelay ();
    return ss.str ();
  }
};

#endif /* TWOINPUTGATE_H_ */
