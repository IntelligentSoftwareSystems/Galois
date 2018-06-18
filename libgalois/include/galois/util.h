#ifndef GALOIS_UTIL_H_
#define GALOIS_UTIL_H_

#include <string>
#include <sstream>

namespace galois {

template <typename C>
void splitCSVstr(const std::string& inputStr, C& output,
                 const char delim = ',') {
  std::stringstream ss(inputStr);

  for (std::string item; std::getline(ss, item, delim);) {
    output.push_back(item);
  }
}

} // namespace galois

#endif // GALOIS_UTIL_H_
