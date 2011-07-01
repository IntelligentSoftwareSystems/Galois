#include <string>
#include <algorithm>
#include "comDefs.h"

std::string  toLowerCase (std::string str) {
  std::transform (str.begin (), str.end (), str.begin (), ::tolower);
  return str;
}
