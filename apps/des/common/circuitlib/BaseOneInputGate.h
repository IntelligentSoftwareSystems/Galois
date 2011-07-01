#ifndef _BASE_ONE_INPUT_GATE_H_
#define _BASE_ONE_INPUT_GATE_H_

#include <string>
#include <sstream>

#include "comDefs.h"
#include "logicDefs.h"

class BaseOneInputGate {
protected:

  /** The output name. */
  std::string outputName;

  /** The input name. */
  std::string inputName;

  /** The output val. */
  LogicVal outputVal;

  /** The input val. */
  LogicVal inputVal;


public:
  /**
   * Instantiates a new one input gate.
   *
   * @param outputName the output name
   * @param inputName the input name
   * @param delay the delay
   */
  BaseOneInputGate (const std::string& outputName, const std::string& inputName)
    : outputName (outputName), inputName (inputName), outputVal ('0'), inputVal ('0') {}



  bool hasInputName(const std::string& net) const {
    return (inputName == (net));
  }

  bool hasOutputName(const std::string& net) const {
    return outputName == (net);
  }

  const std::string toString() const {
    std::ostringstream ss;

    ss << "output: " << outputName << " = " << outputVal << ", input: " << inputName << " = " << inputVal;
    return ss.str ();
  }

  /**
   * Gets the input name.
   *
   * @return the input name
   */
  const std::string& getInputName() const {
    return inputName;
  }

  /**
   * Sets the input name.
   *
   * @param inputName the new input name
   */
  void setInputName(const std::string& inputName) {
    this->inputName = inputName;
  }

  /**
   * Gets the input val.
   *
   * @return the input val
   */
  const LogicVal& getInputVal() const {
    return inputVal;
  }

  /**
   * Sets the input val.
   *
   * @param inputVal the new input val
   */
  void setInputVal(const LogicVal& inputVal) {
    this->inputVal = inputVal;
  }

  /* (non-Javadoc)
   * @see des.unordered.circuitlib.LogicGate#getOutputName()
   */
  const std::string& getOutputName() const {
    return outputName;
  }

  /**
   * Sets the output name.
   *
   * @param outputName the new output name
   */
  void setOutputName(const std::string& outputName) {
    this->outputName = outputName;
  }

  /**
   * Gets the output val.
   *
   * @return the output val
   */
  const LogicVal& getOutputVal() const {
    return outputVal;
  }

  /**
   * Sets the output val.
   *
   * @param outputVal the new output val
   */
  void setOutputVal(const LogicVal& outputVal) {
    this->outputVal = outputVal;
  }
};

#endif
