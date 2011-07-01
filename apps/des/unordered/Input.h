/*
 * Input.h
 *
 *  Created on: Jun 23, 2011
 *      Author: amber
 */

#ifndef INPUT_H_
#define INPUT_H_

#include <string>
#include <cassert>

#include "BaseOneInputGate.h"
#include "LogicFunctions.h"
#include "OneInputGate.h"

// may as well be inherited from OneInputGate
/**
 * The Class Input.
 */
template <typename GraphTy, typename GNodeTy>
class Input: public OneInputGate <GraphTy, GNodeTy> {

public: 
  /**
   * Instantiates a new Input.
   *
   * @param outputName the output name
   * @param inputName the Input name
   */
  Input(const std::string& outputName, const std::string& inputName)
    : OneInputGate<GraphTy, GNodeTy> (BUF(), outputName, inputName, 0l)
  {}


  Input (const Input<GraphTy, GNodeTy> & that) 
    : OneInputGate<GraphTy, GNodeTy> (that) {}


  virtual Input<GraphTy, GNodeTy>* clone () const {
    return new Input<GraphTy, GNodeTy> (*this);
  }

  virtual const std::string getGateName() const {
    return "Input";
  }
  
protected:

  /* (non-Javadoc)
   * @see des.unordered.circuitlib.LogicGate#evalOutput()
   */
  virtual LogicVal evalOutput() const {
    return this->BaseOneInputGate::inputVal;
  }

  /* (non-Javadoc)
   * @see des.unordered.circuitlib.OneInputGate#execEvent(galois.objects.graph.GNode, des.unordered.Event)
   */
  virtual void execEvent(GraphTy& graph, GNodeTy& myNode, const Event<GNodeTy, LogicUpdate>& event) {

    if (event.getType () == Event<GNodeTy, LogicUpdate>::NULL_EVENT) {
      // send out null messages

      OneInputGate<GraphTy, GNodeTy>::execEvent(graph, myNode, event); // same functionality as OneInputGate

    } else {

      LogicUpdate lu = (LogicUpdate) event.getAction ();
      if (this->BaseOneInputGate::outputName == lu.getNetName()) {
        this->BaseOneInputGate::inputVal = lu.getNetVal();

        this->BaseOneInputGate::outputVal = this->BaseOneInputGate::inputVal;

        LogicUpdate drvFanout(this->BaseOneInputGate::outputName, this->BaseOneInputGate::outputVal);

        sendEventsToFanout(graph, myNode, event, Event<GNodeTy, LogicUpdate>::REGULAR_EVENT, drvFanout);
      } else {
        LogicGate<GraphTy, GNodeTy>::netNameMismatch(lu);
      }

    }

  }

};
#endif /* INPUT_H_ */
