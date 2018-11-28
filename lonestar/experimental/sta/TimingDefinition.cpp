#include "TimingDefinition.h"

#ifdef GALOIS_EDA_USE_DOUBLE_AS_MY_FLOAT
MyFloat getMyFloat(std::string& str) {
  return std::stod(str);
}
#else
MyFloat getMyFloat(std::string& str) {
  return std::stof(str);
}
#endif
