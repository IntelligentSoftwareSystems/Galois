/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
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

#ifndef DES_LOGICUPDATE_H_
#define DES_LOGICUPDATE_H_

#include <string>
#include <sstream>

#include "logicDefs.h"


namespace des {

/**
 * The Class LogicUpdate is the msg carried by events. represents a change in the value of a net.
 */
class LogicUpdate {

  /** The net name. */
  const std::string* netName;

  /** The net val. */
  LogicVal netVal;

public:

  /**
   * Instantiates a new logi update
   *
   * @param netName the net name
   * @param netVal the net val
   */
  LogicUpdate(const std::string& netName, const LogicVal& netVal)
    : netName (&netName), netVal (netVal) {}

  LogicUpdate (): netName (NULL), netVal(LOGIC_UNKNOWN) {}

  friend bool operator == (const LogicUpdate& left, const LogicUpdate& right) {
    return ((*left.netName) == (*right.netName)) && (left.netVal == right.netVal);
  }

  friend bool operator != (const LogicUpdate& left, const LogicUpdate& right) {
    return !(left == right);
  }

  /**
   * string representation
   */
  const std::string str() const {
    std::ostringstream ss;
    ss << "netName = " << *netName << " netVal = " << netVal;
    return ss.str ();
  }

  /**
   * Gets the net name.
   *
   * @return the net name
   */
  const std::string& getNetName() const {
    return *netName;
  }


  /**
   * Gets the net val.
   *
   * @return the net val
   */
  LogicVal getNetVal() const {
    return netVal;
  }

};

} // namespace des

#endif /* DES_LOGICUPDATE_H_ */
