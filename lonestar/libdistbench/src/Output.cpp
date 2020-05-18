#include "DistBench/Output.h"
#include "galois/runtime/Network.h"

#include <iomanip>

namespace {
std::string zeroPad(int num, int width) {
  std::ostringstream out;

  out << std::setw(width) << std::setfill('0') << num;

  return out.str();
}

} // namespace

std::string makeOutputFilename(const std::string& outputDir) {
  std::string filename = zeroPad(galois::runtime::getHostID(), 8);

  std::string output{outputDir};
  if (output.empty() || output.compare(output.size() - 1, 1, "/") == 0) {
    output += filename;
  } else {
    output += "/" + filename;
  }

  return output;
}
