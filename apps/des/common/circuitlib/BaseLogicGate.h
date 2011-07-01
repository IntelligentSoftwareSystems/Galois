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
   * Eval output.
   *
   * @return the 
   */
  virtual LogicVal evalOutput() const = 0;

  /**
   * Checks for input name.
   *
   * @param net the net
   * @return true, if successful
   */
  virtual bool hasInputName(const std::string& net) const = 0;

  /**
   * Checks for output name.
   *
   * @param net the net
   * @return true, if successful
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
