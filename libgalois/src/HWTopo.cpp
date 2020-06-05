#include "galois/substrate/HWTopo.h"

#include <stdexcept>

std::vector<int> galois::substrate::parseCPUList(const std::string& line) {
  std::vector<int> vals;

  size_t current;
  size_t next = -1;
  try {
    do {
      current  = next + 1;
      next     = line.find_first_of(',', current);
      auto buf = line.substr(current, next - current);
      if (!buf.empty()) {
        size_t dash = buf.find_first_of('-', 0);
        if (dash != std::string::npos) { // range
          auto first  = buf.substr(0, dash);
          auto second = buf.substr(dash + 1, std::string::npos);
          unsigned b  = std::stoi(first.data());
          unsigned e  = std::stoi(second.data());
          while (b <= e) {
            vals.push_back(b++);
          }
        } else { // singleton
          vals.push_back(std::stoi(buf.data()));
        }
      }
    } while (next != std::string::npos);
  } catch (const std::invalid_argument&) {
    return std::vector<int>{};
  } catch (const std::out_of_range&) {
    return std::vector<int>{};
  }

  return vals;
}
