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


#include <iostream>
#include <string>
#include <cassert>


#include "Input.h"

// may as well be inherited from OneInputGate
/**
 * The Class Output.
 */
class Output: public Input {

public: 
  /**
   * Instantiates a new Output.
   *
   * @param id
   * @param outputName the output name
   * @param inputName the Output name
   */
  Output(size_t id, BasicPort& impl)
    : Input (id, impl) {}



  virtual Output* clone () const {
    return new Output (*this);
  }

  /**
   * A string representation
   */
  virtual const std::string toString () const {
    std::ostringstream ss;
    ss << AbstractSimObject::toString () << ": " << "Output " << getImpl ().toString ();
    return ss.str ();
  }

protected:

  /**
   * Output just receives events and updates its state, does not send out any events
   */
  virtual void execEvent(Graph& g, GNode& myNode, const EventTy& event) {

     if (event.getType() != EventTy::NULL_EVENT) {
       const LogicUpdate& lu = event.getAction ();
       getImpl ().applyUpdate (lu);
     }
  }

};

#endif /* OUTPUT_H_ */
