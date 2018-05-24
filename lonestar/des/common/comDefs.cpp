#include "comDefs.h"

std::string  des::toLowerCase (std::string str) {
  std::transform (str.begin (), str.end (), str.begin (), ::tolower);
  return str;
}
