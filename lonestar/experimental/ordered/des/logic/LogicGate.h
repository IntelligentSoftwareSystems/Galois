/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#ifndef DES_BASE_LOGIC_GATE_H_
#define DES_BASE_LOGIC_GATE_H_

#include <string>
#include <iostream>

#include "comDefs.h"
#include "logicDefs.h"
#include "LogicUpdate.h"

namespace des {

class LogicGate {

public:
  LogicGate() {}

  LogicGate(const LogicGate& that) {}

  virtual ~LogicGate() {}

  virtual LogicGate* makeClone() const = 0;

  /**
   * @return number of inputs
   */
  virtual size_t getNumInputs() const = 0;

  /**
   * @return number of outputs
   */
  virtual size_t getNumOutputs() const = 0;

  /**
   * @return current output value
   */
  virtual LogicVal getOutputVal() const = 0;

  /**
   * @return namem of the output
   */
  virtual const std::string& getOutputName() const = 0;

  /**
   * @param update
   *
   * applies the update to internal state e.g. change to some input. Must update
   * the output if the inputs have changed
   */
  virtual void applyUpdate(const LogicUpdate& update) = 0;

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
   * @param inputName net name
   * @return index of the input matching the net name provided
   */
  virtual size_t getInputIndex(const std::string& inputName) const = 0;

  /**
   * @param net: name of a wire
   * @return true if has an output with the name equal to 'net'
   */
  virtual bool hasOutputName(const std::string& net) const = 0;

  /**
   * @return string representation
   */
  virtual std::string str() const = 0;

  /**
   * @return delay of the gate
   */
  virtual const SimTime& getDelay() const = 0;

  /**
   * Handles an erroneous situation, where the net name in
   * LogicUpdate provided does not match any of the inputs.
   *
   * @param le
   */
  virtual void netNameMismatch(const LogicUpdate& le) const = 0;

  virtual size_t getStateSize() const = 0;

  virtual void copyState(void* const buf, const size_t bufSize) const = 0;

  virtual void restoreState(void* const buf, const size_t bufSize) = 0;
};

template <size_t Nout, size_t Nin>
class BaseLogicGate : public LogicGate {

  friend class NetlistParser;

protected:
  /** The output name. */
  std::string outputName;

  /** The output val. */
  LogicVal outputVal;

  /** The delay. */
  SimTime delay;

public:
  BaseLogicGate(const std::string& outputName, const LogicVal& outVal,
                const SimTime& delay)
      : outputName(outputName), outputVal(outVal) {
    setDelay(delay);
  }

  /**
   * Gets the delay.
   *
   * @return the delay
   */
  virtual const SimTime& getDelay() const { return delay; }

  /**
   * @return number of inputs
   */
  virtual size_t getNumInputs() const { return Nin; }

  /**
   * @return number of outputs
   */
  virtual size_t getNumOutputs() const { return Nout; }

  /**
   * @return current output value
   */
  virtual LogicVal getOutputVal() const { return outputVal; }

  /**
   * @return namem of the output
   */
  virtual const std::string& getOutputName() const { return outputName; }

  /**
   * @param net: name of a wire
   * @return true if has an output with the name equal to 'net'
   */
  virtual bool hasOutputName(const std::string& net) const {
    return (outputName == net);
  }

  /**
   * Sets the output val.
   *
   * @param outputVal the new output val
   */
  void setOutputVal(const LogicVal& outputVal) { this->outputVal = outputVal; }

  virtual void netNameMismatch(const LogicUpdate& le) const {
    std::cerr << "Received logic update : " << le.str()
              << " with mismatching net name, this = " << str() << std::endl;
    exit(-1);
  }

private:
  /**
   * Sets the delay.
   *
   * @param delay the new delay
   */
  virtual void setDelay(const SimTime& delay) {
    this->delay = delay;
    if (this->delay <= 0) {
      this->delay = MIN_DELAY;
    }
  }

protected:
  struct State {
    LogicVal outputVal;
    SimTime delay;

    State(const BaseLogicGate& g) : outputVal(g.outputVal), delay(g.delay) {}

    void restore(BaseLogicGate& g) {
      g.outputVal = outputVal;
      g.delay     = delay;
    }
  };
};

} // namespace des

#endif
