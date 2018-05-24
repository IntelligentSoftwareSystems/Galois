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
