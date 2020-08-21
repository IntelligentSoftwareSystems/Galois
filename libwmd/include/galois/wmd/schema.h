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

#ifndef GALOIS_WMD_SCHEMA_H_
#define GALOIS_WMD_SCHEMA_H_

#include "graph.h"

#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace galois {
namespace graphs {

using ParsedUID = uint64_t;

template <typename V, typename E>
struct ParsedGraphStructure {
  ParsedGraphStructure() : isNode(false), isEdge(false) {}
  ParsedGraphStructure(V node_) : node(node_), isNode(true), isEdge(false) {}
  ParsedGraphStructure(std::vector<E> edges_)
      : edges(edges_), isNode(false), isEdge(true) {}

  V node;
  std::vector<E> edges;
  bool isNode;
  bool isEdge;
};

template <typename V, typename E>
class FileParser {
public:
  virtual const std::vector<std::string>& GetFiles()                = 0;
  virtual ParsedGraphStructure<V, E> ParseLine(char* line,
                                               uint64_t lineLength) = 0;
  static std::vector<std::string> SplitLine(const char* line,
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
};

template <typename V, typename E>
class WMDParser : public FileParser<V, E> {
public:
  WMDParser(std::vector<std::string> files) : csvFields_(10), files_(files) {}
  WMDParser(uint64_t csvFields, std::vector<std::string> files)
      : csvFields_(csvFields), files_(files) {}

  virtual const std::vector<std::string>& GetFiles() override { return files_; }
  virtual ParsedGraphStructure<V, E> ParseLine(char* line,
                                               uint64_t lineLength) override {
    std::vector<std::string> tokens =
        this->SplitLine(line, lineLength, ',', csvFields_);

    if (tokens[0] == "Person") {
      ParsedUID uid =
          shad::data_types::encode<uint64_t, std::string, UINT>(tokens[1]);
      return ParsedGraphStructure<V, E>(
          V(uid, 0, agile::workflow1::TYPES::PERSON));
    } else if (tokens[0] == "ForumEvent") {
      ParsedUID uid =
          shad::data_types::encode<uint64_t, std::string, UINT>(tokens[4]);
      return ParsedGraphStructure<V, E>(
          V(uid, 0, agile::workflow1::TYPES::FORUMEVENT));
    } else if (tokens[0] == "Forum") {
      ParsedUID uid =
          shad::data_types::encode<uint64_t, std::string, UINT>(tokens[3]);
      return ParsedGraphStructure<V, E>(
          V(uid, 0, agile::workflow1::TYPES::FORUM));
    } else if (tokens[0] == "Publication") {
      ParsedUID uid =
          shad::data_types::encode<uint64_t, std::string, UINT>(tokens[5]);
      return ParsedGraphStructure<V, E>(
          V(uid, 0, agile::workflow1::TYPES::PUBLICATION));
    } else if (tokens[0] == "Topic") {
      ParsedUID uid =
          shad::data_types::encode<uint64_t, std::string, UINT>(tokens[6]);
      return ParsedGraphStructure<V, E>(
          V(uid, 0, agile::workflow1::TYPES::TOPIC));
    } else { // edge type
      agile::workflow1::TYPES inverseEdgeType;
      if (tokens[0] == "Sale") {
        inverseEdgeType = agile::workflow1::TYPES::PURCHASE;
      } else if (tokens[0] == "Author") {
        inverseEdgeType = agile::workflow1::TYPES::WRITTENBY;
      } else if (tokens[0] == "Includes") {
        inverseEdgeType = agile::workflow1::TYPES::INCLUDEDIN;
      } else if (tokens[0] == "HasTopic") {
        inverseEdgeType = agile::workflow1::TYPES::TOPICIN;
      } else if (tokens[0] == "HasOrg") {
        inverseEdgeType = agile::workflow1::TYPES::ORGIN;
      } else {
        // skip nodes
        return ParsedGraphStructure<V, E>();
      }
      std::vector<E> edges;
      E edge(tokens);

      // insert inverse edges to the graph
      E inverseEdge    = edge;
      inverseEdge.type = inverseEdgeType;
      std::swap(inverseEdge.src, inverseEdge.dst);
      std::swap(inverseEdge.src_type, inverseEdge.dst_type);

      edges.emplace_back(edge);
      edges.emplace_back(inverseEdge);

      return ParsedGraphStructure<V, E>(edges);
    }
  }

private:
  uint64_t csvFields_;
  std::vector<std::string> files_;
};

} // namespace graphs
} // namespace galois

#endif
