/*
 * Output.h
 *
 *  Created on: Jun 23, 2011
 *      Author: amber
 */

#ifndef OUTPUT_H_
#define OUTPUT_H_


#include <string>
#include <cassert>

#include "OneInputGate.h"
#include "Event.h"
#include "LogicFunctions.h"

// may as well be inherited from OneInputGate
/**
 * The Class Output.
 */
template <typename GraphTy, typename GNodeTy>
class Output: public OneInputGate <GraphTy, GNodeTy> {

public: 
  /**
   * Instantiates a new Output.
   *
   * @param outputName the output name
   * @param inputName the Output name
   */
  Output(const std::string& outputName, const std::string& inputName)
    : OneInputGate<GraphTy, GNodeTy> (BUF(), outputName, inputName, 0l)
  {}



  Output (const Output<GraphTy, GNodeTy> & that) 
    : OneInputGate<GraphTy, GNodeTy> (that) {}


  virtual Output<GraphTy, GNodeTy>* clone () const {
    return new Output<GraphTy, GNodeTy> (*this);
  }

  virtual const std::string getGateName() const {
    return "Output";
  }
  
protected:

  /* (non-Javadoc)
   * @see des.unordered.circuitlib.LogicGate#evalOutput()
   */
  virtual LogicVal evalOutput() const {
    return BaseOneInputGate::inputVal;
  }

  /* (non-Javadoc)
   * @see des.unordered.circuitlib.OneInputGate#execEvent(galois.objects.graph.GNode, des.unordered.Event)
   */
  virtual void execEvent(GNodeTy& myNode, const Event<GNodeTy, LogicUpdate>& event) {

     if (event.getType() == Event<GNodeTy, LogicUpdate>::NULL_EVENT) {
       // do nothing
     } else {
       // update the inputs of fanout gates
       LogicUpdate lu = (LogicUpdate) event.getAction();
       if (BaseOneInputGate::inputName == (lu.getNetName())) {
         BaseOneInputGate::inputVal = lu.getNetVal();
         BaseOneInputGate::outputVal = BaseOneInputGate::inputVal;

       } else {
         LogicGate<GraphTy, GNodeTy>::netNameMismatch(lu);
       }

     }
  }

};

#endif /* OUTPUT_H_ */
