#ifndef DES_BASIC_PORT_H
#define DES_BASIC_PORT_H

#include <string>

#include "LogicFunctions.h"
#include "OneInputGate.h"

namespace des {

class BasicPort: public OneInputGate {
private:
  static const BUF& BUFFER;

public:
  BasicPort (const std::string&  outputName, const std::string& inputName)
    : OneInputGate (BUFFER, outputName, inputName)  {}

  BasicPort* makeClone () const { return new BasicPort (*this); }


};


} // namespace des


#endif // DES_BASIC_PORT_H
