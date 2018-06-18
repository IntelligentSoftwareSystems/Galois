/**
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of XYZ License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#ifndef GALOIS_GRAPH_REPLICATEDGRAPH_H
#define GALOIS_GRAPH_REPLICATEDGRAPH_H

//#include "galois/Graph/OCGraph.h"
#include "galois/Graph/LCGraph.h"
#include "galois/runtime/PerHostStorage.h"
#include "galois/runtime/Serialize.h"

namespace galois {
namespace graphs {

using namespace galois::runtime::Distributed;

template <typename NodeTy, typename EdgeTy>
class ReplicatedGraph {
  typedef LC_CSR_InOutGraph<NodeTy, EdgeTy, true> Inner;

  gptr<Inner> graph;
  Inner* pGraph;

  struct StructureFromFile {
    gptr<ReplicatedGraph> self;
    std::string fname;
    std::string tname;
    bool symmetric;

    StructureFromFile() {}
    StructureFromFile(gptr<ReplicatedGraph> self, const std::string& f, bool s)
        : self(self), fname(f), symmetric(s) {}
    StructureFromFile(gptr<ReplicatedGraph> self, const std::string& f,
                      const std::string& t)
        : self(self), fname(f), tname(t), symmetric(false) {}

    void operator()(unsigned tid, unsigned) {
      if (tid != 0)
        return;
      assert(0 && "fixme");
      // self->pGraph = &*self->graph;

      if (symmetric) {
        self->pGraph->structureFromFile(fname, symmetric);
      } else {
        self->pGraph->structureFromFile(fname, tname);
      }
    }

    typedef int tt_has_serialize;
    void serialize(SendBuffer& buf) const {
      gSerialize(buf, self, fname, tname, symmetric);
    }
    void deserialize(RecvBuffer& buf) {
      gDeserialize(buf, self, fname, tname, symmetric);
    }
  };

public:
  typedef int tt_has_serialize;
  typedef int tt_is_persistent;

  typedef typename Inner::GraphNode GraphNode;
  typedef typename Inner::edge_data_type edge_data_type;
  typedef typename Inner::node_data_type node_data_type;
  typedef typename Inner::edge_data_reference edge_data_reference;
  typedef typename Inner::node_data_reference node_data_reference;
  typedef typename Inner::edge_iterator edge_iterator;
  typedef typename Inner::in_edge_iterator in_edge_iterator;
  typedef typename Inner::iterator iterator;
  typedef typename Inner::const_iterator const_iterator;
  typedef typename Inner::local_iterator local_iterator;
  typedef typename Inner::const_local_iterator const_local_iterator;

  ReplicatedGraph() : pGraph(0) {
    graph = gptr<Inner>(new Inner);
    // runtime::allocatePerHost(this);
  }

  ReplicatedGraph(DeSerializeBuffer& s) { deserialize(s); }

  ~ReplicatedGraph() {
    // XXX cannot deallocate
    // runtime::deallocatePerHost(graph);
  }

  void serialize(SerializeBuffer& s) const { gSerialize(s, graph); }
  void deserialize(DeSerializeBuffer& s) { gDeserialize(s, graph); }

  node_data_reference getData(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    return pGraph->getData(N, mflag);
  }

  edge_data_reference getEdgeData(edge_iterator ni,
                                  MethodFlag mflag = MethodFlag::NONE) {
    return pGraph->getEdgeData(ni, mflag);
  }

  GraphNode getEdgeDst(edge_iterator ni) { return pGraph->getEdgeDst(ni); }

  uint64_t size() const { return pGraph->size(); }
  uint64_t sizeEdges() const { return pGraph->sizeEdges(); }

  iterator begin() const { return pGraph->begin(); }
  iterator end() const { return pGraph->end(); }

  local_iterator local_begin() const { return pGraph->local_begin(); }
  local_iterator local_end() const { return pGraph->local_end(); }

  edge_iterator edge_begin(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    return pGraph->edge_begin(N, mflag);
  }

  edge_iterator edge_end(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    return pGraph->edge_end(N, mflag);
  }

  EdgesIterator<ReplicatedGraph> out_edges(GraphNode N,
                                           MethodFlag mflag = MethodFlag::ALL) {
    return EdgesIterator<ReplicatedGraph>(*this, N, mflag);
  }

  /**
   * Sorts outgoing edges of a node. Comparison function is over EdgeTy.
   */
  template <typename CompTy>
  void sortEdgesByEdgeData(GraphNode N,
                           const CompTy& comp = std::less<EdgeTy>(),
                           MethodFlag mflag   = MethodFlag::ALL) {
    pGraph->sortEdgesByEdgeData(N, comp, mflag);
  }

  /**
   * Sorts outgoing edges of a node. Comparison function is over
   * <code>EdgeSortValue<EdgeTy></code>.
   */
  template <typename CompTy>
  void sortEdges(GraphNode N, const CompTy& comp,
                 MethodFlag mflag = MethodFlag::ALL) {
    pGraph->sortEdges(N, comp, mflag);
  }

  edge_data_reference getInEdgeData(in_edge_iterator ni,
                                    MethodFlag mflag = MethodFlag::NONE) {
    return pGraph->getInEdgeData(ni, mflag);
  }

  GraphNode getInEdgeDst(in_edge_iterator ni) {
    return pGraph->getInEdgeDst(ni);
  }

  in_edge_iterator in_edge_begin(GraphNode N,
                                 MethodFlag mflag = MethodFlag::ALL) {
    return pGraph->in_edge_begin(N, mflag);
  }

  in_edge_iterator in_edge_end(GraphNode N,
                               MethodFlag mflag = MethodFlag::ALL) {
    return pGraph->in_edge_end(N, mflag);
  }

  InEdgesIterator<ReplicatedGraph>
  in_edges(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    return InEdgesIterator<ReplicatedGraph>(*this, N, mflag);
  }

  /**
   * Sorts incoming edges of a node. Comparison function is over EdgeTy.
   */
  template <typename CompTy>
  void sortInEdgesByEdgeData(GraphNode N,
                             const CompTy& comp = std::less<EdgeTy>(),
                             MethodFlag mflag   = MethodFlag::ALL) {
    return pGraph->sortInEdgesByEdgeData(N, comp, mflag);
  }

  /**
   * Sorts incoming edges of a node. Comparison function is over
   * <code>EdgeSortValue<EdgeTy></code>.
   */
  template <typename CompTy>
  void sortInEdges(GraphNode N, const CompTy& comp,
                   MethodFlag mflag = MethodFlag::ALL) {
    return pGraph->sortInEdges(N, comp, mflag);
  }

  size_t idFromNode(GraphNode N) { return pGraph->idFromNode(N); }

  GraphNode nodeFromId(size_t N) { return pGraph->nodeFromId(N); }

  void structureFromFile(const std::string& fname, bool symmetric) {
    galois::on_each(
        StructureFromFile(gptr<ReplicatedGraph>(this), fname, symmetric));
  }

  void structureFromFile(const std::string& fname, const std::string& tname) {
    galois::on_each(
        StructureFromFile(gptr<ReplicatedGraph>(this), fname, tname));
  }
};

} // namespace graphs
} // namespace galois

#endif
