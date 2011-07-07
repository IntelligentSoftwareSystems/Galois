/** Output is an output port in the circuit -*- C++ -*-
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

  /**
   * just replicates the output
   */
  virtual LogicVal evalOutput() const {
    return this->BaseOneInputGate::inputVal;
  }

  /**
   * Output just receives events and updates its state, does not send out any events
   */
  virtual void execEvent(GNodeTy& myNode, const Event<GNodeTy, LogicUpdate>& event) {

     if (event.getType() == Event<GNodeTy, LogicUpdate>::NULL_EVENT) {
       // do nothing
     } else {
       // update the inputs of fanout gates
       LogicUpdate lu = (LogicUpdate) event.getAction();
       if (this->BaseOneInputGate::inputName == (lu.getNetName())) {
         this->BaseOneInputGate::inputVal = lu.getNetVal();
         this->BaseOneInputGate::outputVal = this->BaseOneInputGate::inputVal;

       } else {
         LogicGate<GraphTy, GNodeTy>::netNameMismatch(lu);
       }

     }
  }

};

#endif /* OUTPUT_H_ */
