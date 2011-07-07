/** BaseLogicGate implements the basic structure of a logic gate -*- C++ -*-
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

#ifndef _BASE_LOGIC_GATE_H_
#define _BASE_LOGIC_GATE_H_

#include <string>

#include "comDefs.h"
#include "logicDefs.h"

class BaseLogicGate {
  static const SimTime MIN_DELAY = 1l;

private:
  /** The delay. */
  SimTime delay;

public:
  BaseLogicGate (const SimTime& delay) {
    setDelay (delay);
  }

  BaseLogicGate (const BaseLogicGate& that) {
    setDelay (delay);
  }

  /**
   * Gets the delay.
   *
   * @return the delay
   */
  const SimTime& getDelay() const {
    return delay;
  }

  /**
   * Sets the delay.
   *
   * @param delay the new delay
   */
  void setDelay(const SimTime& delay) {
    this->delay = delay;
    if (this->delay <= 0) {
      this->delay = MIN_DELAY;
    }
  }

  /**
   * Evaluate output based on the current state of the input
   *
   * @return the 
   */
  virtual LogicVal evalOutput() const = 0;

  /**
   * @param net: name of a wire
   * @return true if has an input with the name equal to 'net'
   */
  virtual bool hasInputName(const std::string& net) const = 0;

  /**
   * @param net: name of a wire
   * @return true if has an output with the name equal to 'net'
   */
  virtual bool hasOutputName(const std::string& net) const = 0;

  /**
   * Gets the output name.
   *
   * @return the output name
   */
  virtual const std::string& getOutputName() const = 0;
};

#endif
