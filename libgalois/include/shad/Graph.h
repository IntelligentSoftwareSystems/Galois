//TODO(hc): Upgrade copyright if it is necessary; for now, we have no plan
// to make this public.

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
//===----------------------------------------------------------------------===//

#ifndef LIBGALOIS_INCLUDE_SHAD_GRAPH_H_
#define LIBGALOIS_INCLUDE_SHAD_GRAPH_H_

#include <cstdint>
#include <limits>
#include <vector>

#include "DataTypes.h"
#include "GraphTypes.h"

#define UINT   shad::data_types::UINT
#define DOUBLE shad::data_types::DOUBLE
#define USDATE shad::data_types::USDATE
#define ENCODE shad::data_types::encode

namespace shad {

class Vertex {
  public:
    // Vertex id; initially it is set
    // to a local node id while CuSP reads a file and constructs
    // this vertex. After each host finishes and synchronizes it to construct
    // a full CSR graph, it is updated to a global node id.
    uint64_t id;
    TYPES    type;
    uint64_t shadKey;
    // Number of edges.
    // This is incremented while reads a graph.
    uint64_t numEdges{0};

    Vertex () {
      this->id    = shad::data_types::kNullValue<uint64_t>;
      this->type  = TYPES::NONE;
      this->shadKey = shad::data_types::kNullValue<uint64_t>;
    }

    Vertex (uint64_t id_, TYPES type_, uint64_t shadKey_) {
      this->id    = id_;
      this->type  = type_;
      this->shadKey = shadKey_;
    }

    void incrNumEdges() {
      this->numEdges += 1;
    }

    uint64_t getNumEdges() {
      return this->numEdges;
    }
};

class Edge {
  public:
    uint64_t src;     // vertex id of src
    uint64_t dst;     // vertex id of dst
    TYPES    type;
    TYPES    src_type;
    TYPES    dst_type;
    uint64_t src_glbid;
    uint64_t dst_glbid;

    Edge () {
      src       = shad::data_types::kNullValue<uint64_t>;
      dst       = shad::data_types::kNullValue<uint64_t>;
      type      = TYPES::NONE;
      src_type  = TYPES::NONE;
      dst_type  = TYPES::NONE;
      src_glbid = shad::data_types::kNullValue<uint64_t>;
      dst_glbid = shad::data_types::kNullValue<uint64_t>;
    }

    Edge (std::vector <std::string> & tokens) {
      if (tokens[0] == "Sale") {
         src       = ENCODE<uint64_t, std::string, UINT>(tokens[1]);
         dst       = ENCODE<uint64_t, std::string, UINT>(tokens[2]);
         type      = TYPES::SALE;
         src_type  = TYPES::PERSON;
         dst_type  = TYPES::PERSON;
         src_glbid = shad::data_types::kNullValue<uint64_t>;
         dst_glbid = shad::data_types::kNullValue<uint64_t>;
      } else if (tokens[0] == "Author") {
         src  = ENCODE<uint64_t, std::string, UINT>(tokens[1]);
         type      = TYPES::AUTHOR;
         src_type  = TYPES::PERSON;
         src_glbid = shad::data_types::kNullValue<uint64_t>;
         dst_glbid = shad::data_types::kNullValue<uint64_t>;
         if      (tokens[3] != "") dst = ENCODE<uint64_t, std::string, UINT>(tokens[3]);
         else if (tokens[4] != "") dst = ENCODE<uint64_t, std::string, UINT>(tokens[4]);
         else if (tokens[5] != "") dst = ENCODE<uint64_t, std::string, UINT>(tokens[5]);
         if      (tokens[3] != "") dst_type = TYPES::FORUM;
         else if (tokens[4] != "") dst_type = TYPES::FORUMEVENT;
         else if (tokens[5] != "") dst_type = TYPES::PUBLICATION;
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
         if      (tokens[3] != "") src = ENCODE<uint64_t, std::string, UINT>(tokens[3]);
         else if (tokens[4] != "") src = ENCODE<uint64_t, std::string, UINT>(tokens[4]);
         else if (tokens[5] != "") src = ENCODE<uint64_t, std::string, UINT>(tokens[5]);
         if      (tokens[3] != "") src_type = TYPES::FORUM;
         else if (tokens[4] != "") src_type = TYPES::FORUMEVENT;
         else if (tokens[5] != "") src_type = TYPES::PUBLICATION;
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
};

} // namespace agile::workflow1

#endif // GRAPH_H
