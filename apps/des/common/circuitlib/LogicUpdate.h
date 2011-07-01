/*
 * LogicUpdate.h
 *
 *  Created on: Jun 23, 2011
 *      Author: amber
 */

#ifndef LOGICUPDATE_H_
#define LOGICUPDATE_H_

#include <string>
#include <sstream>

#include "logicDefs.h"
/**
 * The Class LogicUpdate is the msg carried by events. represents a change in the value of a net.
 */
struct LogicUpdate {

  /** The net name. */
  std::string netName;

  /** The net val. */
  LogicVal netVal;

  /**
   * Instantiates a new logic event.
   *
   * @param netName the net name
   * @param netVal the net val
   */
  LogicUpdate(std::string netName, LogicVal netVal) 
    : netName (netName), netVal (netVal) {}

  LogicUpdate (): netName (""), netVal('0') {}

  /* (non-Javadoc)
   * @see java.lang.Object#toString()
   */
  const std::string toString() const {
    std::ostringstream ss;
    ss << "netName = " << netName << " netVal = " << netVal;
    return ss.str ();
  }

  /**
   * Gets the net name.
   *
   * @return the net name
   */
  const std::string getNetName() const {
    return netName;
  }

  /**
   * Sets the net name.
   *
   * @param netName the new net name
   */
  void setNetName(const std::string& netName) {
    this->netName = netName;
  }

  /**
   * Gets the net val.
   *
   * @return the net val
   */
  const LogicVal getNetVal() const {
    return netVal;
  }

  /**
   * Sets the net val.
   *
   * @param netVal the new net val
   */
  void setNetVal(const LogicVal& netVal) {
    this->netVal = netVal;
  }
};
#endif /* LOGICUPDATE_H_ */
