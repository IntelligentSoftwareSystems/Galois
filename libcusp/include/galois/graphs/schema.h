//===------------------------------------------------------------*- C++ -*-===//
//
//                            The AGILE Workflows
//
//===----------------------------------------------------------------------===//
// ** Pre-Copyright Notice
//
// This computer software was prepared by Battelle Memorial Institute,
// hereinafter the Contractor, under Contract No. DE-AC05-76RL01830 with the
// Department of Energy (DOE). All rights in the computer software are reserved
// by DOE on behalf of the United States Government and the Contractor as
// provided in the Contract. You are authorized to use this computer software
// for Governmental purposes but it is not to be released or distributed to the
// public. NEITHER THE GOVERNMENT NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS
// OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This
// notice including this sentence must appear on any copies of this computer
// software.
//
// ** Disclaimer Notice
//
// This material was prepared as an account of work sponsored by an agency of
// the United States Government. Neither the United States Government nor the
// United States Department of Energy, nor Battelle, nor any of their employees,
// nor any jurisdiction or organization that has cooperated in the development
// of these materials, makes any warranty, express or implied, or assumes any
// legal liability or responsibility for the accuracy, completeness, or
// usefulness or any information, apparatus, product, software, or process
// disclosed, or represents that its use would not infringe privately owned
// rights. Reference herein to any specific commercial product, process, or
// service by trade name, trademark, manufacturer, or otherwise does not
// necessarily constitute or imply its endorsement, recommendation, or favoring
// by the United States Government or any agency thereof, or Battelle Memorial
// Institute. The views and opinions of authors expressed herein do not
// necessarily state or reflect those of the United States Government or any
// agency thereof.
//
//                    PACIFIC NORTHWEST NATIONAL LABORATORY
//                                 operated by
//                                   BATTELLE
//                                   for the
//                      UNITED STATES DEPARTMENT OF ENERGY
//                       under Contract DE-AC05-76RL01830
//===----------------------------------------------------------------------===//

#include <string>
#include <variant>
#include <vector>
#include "graph.h"

namespace galois {
namespace graphs {

using ParsedUID = uint64_t;

template <typename E>
struct ParsedGraphStructure {
  ParsedGraphStructure(std::vector<E> edges_)
      : edges(edges_) {}

  std::vector<E> edges;
};

template <typename E>
class FileParser {
public:
  FileParser(std::string files) : csvFields_(2), files_(files) {}
  FileParser(uint64_t csvFields, std::string files)
      : csvFields_(csvFields), files_(files) {}
  std::string GetFiles() { return files_; }
  ParsedGraphStructure<E> ParseLine(char* line, uint64_t lineLength) {
    std::vector<std::string> tokens =
        this->SplitLine(line, lineLength, ' ', csvFields_);

      std::vector<E> edges;
      E edge(tokens);
      E inverseEdge    = edge;
      inverseEdge.src = edge.dst;
      inverseEdge.dst = edge.src;
      edges.emplace_back(edge);
      edges.emplace_back(inverseEdge);

      return ParsedGraphStructure<E>(edges);
  }
  std::vector<std::string> SplitLine(const char* line,
                                            uint64_t lineLength, char delim,
                                            uint64_t numTokens) {
    uint64_t ndx = 0, start = 0, end = 0;
    std::vector<std::string> tokens(numTokens);

    for (; end < lineLength - 1; end++) {
      if (line[end] == delim) {
        tokens[ndx] = std::string(line + start, end - start);
        start       = end + 1;
        ndx++;
      }
    }

    tokens[numTokens - 1] =
        std::string(line + start, end - start); // flush last token
    return tokens;
  }
private:
  uint64_t csvFields_;
  std::string files_;
};


} // namespace graphs
} // namespace galois

