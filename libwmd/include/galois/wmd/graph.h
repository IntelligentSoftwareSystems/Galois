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

#ifndef GRAPH_H_
#define GRAPH_H_

#include <cstdint>
#include <limits>
#include <vector>
#include <unordered_map>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include "data_types.h"
#include "graphTypes.h"
#include "galois/graphs/LS_LC_CSR_64_Graph.h"

#define UINT shad::data_types::UINT
// #define DOUBLE shad::data_types::DOUBLE
#define USDATE shad::data_types::USDATE
#define ENCODE shad::data_types::encode

namespace agile::workflow1 {

class Vertex {
public:
  uint64_t id;    // user id of the vertex given in dataset (a.k.a token)
  uint64_t glbid; // global id of the vertex in the graph
  uint64_t edges; // number of edges
  TYPES type;

  Vertex() {
    id    = shad::data_types::kNullValue<uint64_t>;
    glbid = shad::data_types::kNullValue<uint64_t>;
    edges = 0;
    type  = TYPES::NONE;
  }

  Vertex(uint64_t glbid_, uint64_t id_, TYPES type_) {
    id    = id_;
    glbid = glbid_;
    edges = 0;
    type  = type_;
  }
};

class Edge {
public:
  uint64_t src; // vertex id of src
  uint64_t dst; // vertex id of dst
  TYPES type;
  TYPES src_type;
  TYPES dst_type;
  uint64_t src_glbid;
  uint64_t dst_glbid;

  Edge() {
    src       = shad::data_types::kNullValue<uint64_t>;
    dst       = shad::data_types::kNullValue<uint64_t>;
    type      = TYPES::NONE;
    src_type  = TYPES::NONE;
    dst_type  = TYPES::NONE;
    src_glbid = shad::data_types::kNullValue<uint64_t>;
    dst_glbid = shad::data_types::kNullValue<uint64_t>;
  }

  Edge(std::vector<std::string>& tokens) {
    if (tokens[0] == "Sale") {
      src       = ENCODE<uint64_t, std::string, UINT>(tokens[1]);
      dst       = ENCODE<uint64_t, std::string, UINT>(tokens[2]);
      type      = TYPES::SALE;
      src_type  = TYPES::PERSON;
      dst_type  = TYPES::PERSON;
      src_glbid = shad::data_types::kNullValue<uint64_t>;
      dst_glbid = shad::data_types::kNullValue<uint64_t>;
    } else if (tokens[0] == "Author") {
      src       = ENCODE<uint64_t, std::string, UINT>(tokens[1]);
      type      = TYPES::AUTHOR;
      src_type  = TYPES::PERSON;
      src_glbid = shad::data_types::kNullValue<uint64_t>;
      dst_glbid = shad::data_types::kNullValue<uint64_t>;
      if (tokens[3] != "")
        dst = ENCODE<uint64_t, std::string, UINT>(tokens[3]);
      else if (tokens[4] != "")
        dst = ENCODE<uint64_t, std::string, UINT>(tokens[4]);
      else if (tokens[5] != "")
        dst = ENCODE<uint64_t, std::string, UINT>(tokens[5]);
      if (tokens[3] != "")
        dst_type = TYPES::FORUM;
      else if (tokens[4] != "")
        dst_type = TYPES::FORUMEVENT;
      else if (tokens[5] != "")
        dst_type = TYPES::PUBLICATION;
    } else if (tokens[0] == "Includes") {
      src       = ENCODE<uint64_t, std::string, UINT>(tokens[3]);
      dst       = ENCODE<uint64_t, std::string, UINT>(tokens[4]);
      type      = TYPES::INCLUDES;
      src_type  = TYPES::FORUM;
      dst_type  = TYPES::FORUMEVENT;
      src_glbid = shad::data_types::kNullValue<uint64_t>;
      dst_glbid = shad::data_types::kNullValue<uint64_t>;
    } else if (tokens[0] == "HasTopic") {
      dst       = ENCODE<uint64_t, std::string, UINT>(tokens[6]);
      type      = TYPES::HASTOPIC;
      dst_type  = TYPES::TOPIC;
      src_glbid = shad::data_types::kNullValue<uint64_t>;
      dst_glbid = shad::data_types::kNullValue<uint64_t>;
      if (tokens[3] != "")
        src = ENCODE<uint64_t, std::string, UINT>(tokens[3]);
      else if (tokens[4] != "")
        src = ENCODE<uint64_t, std::string, UINT>(tokens[4]);
      else if (tokens[5] != "")
        src = ENCODE<uint64_t, std::string, UINT>(tokens[5]);
      if (tokens[3] != "")
        src_type = TYPES::FORUM;
      else if (tokens[4] != "")
        src_type = TYPES::FORUMEVENT;
      else if (tokens[5] != "")
        src_type = TYPES::PUBLICATION;
    } else if (tokens[0] == "HasOrg") {
      src       = ENCODE<uint64_t, std::string, UINT>(tokens[5]);
      dst       = ENCODE<uint64_t, std::string, UINT>(tokens[6]);
      type      = TYPES::HASORG;
      src_type  = TYPES::PUBLICATION;
      dst_type  = TYPES::TOPIC;
      src_glbid = shad::data_types::kNullValue<uint64_t>;
      dst_glbid = shad::data_types::kNullValue<uint64_t>;
    }
  }

private:
  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar& src;
    ar& dst;
    ar& type;
    ar& src_type;
    ar& dst_type;
    ar& src_glbid;
    ar& dst_glbid;
  }
};

} // namespace agile::workflow1

#endif // GRAPH_H
