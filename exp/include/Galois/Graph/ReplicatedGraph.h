/** Simple replicated graph -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @section Description
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#ifndef GALOIS_GRAPH_REPLICATEDGRAPH_H
#define GALOIS_GRAPH_REPLICATEDGRAPH_H

//#include "Galois/Graph/OCGraph.h"
#include "Galois/Graph/LCGraph.h"
#include "Galois/Runtime/Serialize.h"

namespace Galois {
namespace Graph {

using namespace Galois::Runtime::Distributed;

template<typename NodeTy, typename EdgeTy>
class ReplicatedGraph: public LC_CSR_InOutGraph<NodeTy, EdgeTy, true> { 
  std::string filename;
  std::string transposeName;
  bool symmetric;

public:
  typedef int tt_has_serialize;
  typedef int tt_is_persistent;

  ReplicatedGraph() { }
  ReplicatedGraph(DeSerializeBuffer& s) { }

  void serialize(SerializeBuffer& s) const {
    gSerialize(s, filename); gSerialize(s, transposeName); gSerialize(s, symmetric); 
  }

  void deserialize(DeSerializeBuffer& s) { 
    gDeserialize(s, filename); gDeserialize(s, transposeName); gDeserialize(s, symmetric);
  }

  // XXX load
};

} // end namespace
} // end namespace

#endif
