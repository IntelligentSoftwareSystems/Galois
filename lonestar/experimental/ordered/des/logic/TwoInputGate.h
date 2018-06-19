/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

#ifndef DES_BASE_TWO_INPUT_GATE_H_
#define DES_BASE_TWO_INPUT_GATE_H_

#include <string>
#include <sstream>

#include <cassert>

#include "comDefs.h"
#include "logicDefs.h"
#include "LogicFunctions.h"
#include "LogicGate.h"

namespace des {

struct TwoInputGateTraits {
  static const size_t N_OUT = 1;
  static const size_t N_IN  = 2;
};

class TwoInputGate : public BaseLogicGate<TwoInputGateTraits::N_OUT,
                                          TwoInputGateTraits::N_IN> {
private:
  typedef BaseLogicGate<TwoInputGateTraits::N_OUT, TwoInputGateTraits::N_IN>
      SuperTy;

protected:
  /**
   * The functor that computes a value for the output of the gate
   * when provided with the values of the input
   */
  const TwoInputFunc& func;

  /** The input1 name. */
  std::string input1Name;

  /** The input2 name. */
  std::string input2Name;

  /** The input1 val. */
  LogicVal input1Val;

  /** The input2 val. */
  LogicVal input2Val;

public:
  /**
   * Instantiates a new two input gate.
   */
  TwoInputGate(const TwoInputFunc& func, const std::string& outputName,
               const std::string& input1Name, const std::string& input2Name,
               const SimTime& delay = MIN_DELAY)
      : SuperTy(outputName, LOGIC_ZERO, delay), func(func),
        input1Name(input1Name), input2Name(input2Name), input1Val(LOGIC_ZERO),
        input2Val(LOGIC_ZERO) {}

  virtual TwoInputGate* makeClone() const { return new TwoInputGate(*this); }

  /**
   * Applies the update to internal state e.g. change to some input. Must update
   * the output if the inputs have changed
   *
   * @param lu the update
   */
  virtual void applyUpdate(const LogicUpdate& lu) {

    if (input1Name == (lu.getNetName())) {
      input1Val = lu.getNetVal();

    } else if (input2Name == (lu.getNetName())) {
      input2Val = lu.getNetVal();

    } else {
      SuperTy::netNameMismatch(lu);
    }

    // output has been changed
    // update output immediately
    // generate events to send to all fanout gates to update their inputs afer
    // delay
    this->outputVal = evalOutput();
  }

  /**
   * Evaluate output based on the current state of the input
   *
   * @return the
   */
  virtual LogicVal evalOutput() const { return func(input1Val, input2Val); }

  /**
   * @param net: name of a wire
   * @return true if has an input with the name equal to 'net'
   */
  virtual bool hasInputName(const std::string& net) const {
    return (input1Name == (net) || input2Name == (net));
  }

  /**
   * @param inputName net name
   * @return index of the input matching the net name provided
   */
  virtual size_t getInputIndex(const std::string& inputName) const {
    if (this->input2Name == (inputName)) {
      return 1;

    } else if (this->input1Name == (inputName)) {
      return 0;

    } else {
      abort();
      return -1; // error
    }
  }

  /**
   * @return string representation
   */
  virtual std::string str() const {
    std::ostringstream ss;
    ss << func.str() << " output: " << outputName << " = " << outputVal
       << " input1: " << input1Name << " = " << input1Val
       << " input2: " << input2Name << " = " << input2Val;
    return ss.str();
  }

  /**
   * Gets the input1 name.
   *
   * @return the input1 name
   */
  const std::string& getInput1Name() const { return input1Name; }

  /**
   * Sets the input1 name.
   *
   * @param input1Name the new input1 name
   */
  void setInput1Name(const std::string& input1Name) {
    this->input1Name = input1Name;
  }

  /**
   * Gets the input1 val.
   *
   * @return the input1 val
   */
  const LogicVal& getInput1Val() const { return input1Val; }

  /**
   * Sets the input1 val.
   *
   * @param input1Val the new input1 val
   */
  void setInput1Val(const LogicVal& input1Val) { this->input1Val = input1Val; }

  /**
   * Gets the input2 name.
   *
   * @return the input2 name
   */
  const std::string& getInput2Name() { return input2Name; }

  /**
   * Sets the input2 name.
   *
   * @param input2Name the new input2 name
   */
  void setInput2Name(const std::string& input2Name) {
    this->input2Name = input2Name;
  }

  /**
   * Gets the input2 val.
   *
   * @return the input2 val
   */
  const LogicVal& getInput2Val() const { return input2Val; }

  /**
   * Sets the input2 val.
   *
   * @param input2Val the new input2 val
   */
  void setInput2Val(const LogicVal& input2Val) { this->input2Val = input2Val; }

protected:
  struct State : public BaseLogicGate::State {
    LogicVal input1Val;
    LogicVal input2Val;

    State(const TwoInputGate& g)
        : BaseLogicGate::State(g), input1Val(g.input1Val),
          input2Val(g.input2Val) {}

    void restore(TwoInputGate& g) {
      BaseLogicGate::State::restore(g);

      g.input1Val = input1Val;
      g.input2Val = input2Val;
    }
  };

public:
  virtual size_t getStateSize() const { return sizeof(State); }

  virtual void copyState(void* const buf, const size_t bufSize) const {
    assert(bufSize >= getStateSize() && "insufficient buffer for state");
    new (buf) State(*this);
  }

  virtual void restoreState(void* const buf, const size_t bufSize) {
    assert(bufSize >= getStateSize() && "insufficient buffer for state");

    State* s = reinterpret_cast<State*>(buf);
    s->restore(*this);
  }
};

} // namespace des

#endif
