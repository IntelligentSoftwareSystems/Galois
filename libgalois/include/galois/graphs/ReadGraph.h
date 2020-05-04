/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
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

#ifndef GALOIS_GRAPHS_READGRAPH_H
#define GALOIS_GRAPHS_READGRAPH_H

#include "galois/Galois.h"
#include "galois/graphs/FileGraph.h"
#include "galois/graphs/Details.h"
#include "galois/Timer.h"

namespace galois {
namespace graphs {

/**
 * Allocates and constructs a graph from a file. Tries to balance
 * memory evenly across system. Cannot be called during parallel
 * execution.
 */
template <typename GraphTy, typename... Args>
void readGraph(GraphTy& graph, Args&&... args) {
  typename GraphTy::read_tag tag;
  readGraphDispatch(graph, tag, std::forward<Args>(args)...);
}

template <typename GraphTy>
void readGraphDispatch(GraphTy& graph, read_default_graph_tag tag,
                       const std::string& filename,
                       const bool readUnweighted = false) {
  FileGraph f;
  if (readUnweighted) {
    //! If user specifies that the input graph is unweighted,
    //! the file graph also should be aware of this.
    //! Note that the application still could use the edge data array.
    f.fromFileInterleaved<void>(filename);
  } else {
    f.fromFileInterleaved<typename GraphTy::file_edge_data_type>(filename);
  }
  readGraphDispatch(graph, tag, f, readUnweighted);
}

template <typename GraphTy>
struct ReadGraphConstructFrom {
  GraphTy& graph;
  FileGraph& f;
  bool readUnweighted = false;
  ReadGraphConstructFrom(GraphTy& g, FileGraph& _f) : graph(g), f(_f) {}
  ReadGraphConstructFrom(GraphTy& g, FileGraph& _f, bool _readUnweighted)
                         : graph(g), f(_f), readUnweighted(_readUnweighted) {}
  void operator()(unsigned tid, unsigned total) {
    graph.constructFrom(f, tid, total, readUnweighted);
  }
};

template <typename GraphTy>
void readGraphDispatch(GraphTy& graph, read_default_graph_tag,
                       FileGraph& f, const bool readUnweighted = false) {
  graph.allocateFrom(f);

  ReadGraphConstructFrom<GraphTy> reader(graph, f, readUnweighted);
  galois::on_each(reader);
}

template <typename GraphTy, typename Aux>
struct ReadGraphConstructNodesFrom {
  GraphTy& graph;
  FileGraph& f;
  Aux& aux;
  ReadGraphConstructNodesFrom(GraphTy& g, FileGraph& _f, Aux& a)
      : graph(g), f(_f), aux(a) {}
  void operator()(unsigned tid, unsigned total) {
    graph.constructNodesFrom(f, tid, total, aux);
  }
};

template <typename GraphTy, typename Aux>
struct ReadGraphConstructEdgesFrom {
  GraphTy& graph;
  FileGraph& f;
  Aux& aux;
  ReadGraphConstructEdgesFrom(GraphTy& g, FileGraph& _f, Aux& a)
      : graph(g), f(_f), aux(a) {}
  void operator()(unsigned tid, unsigned total) {
    graph.constructEdgesFrom(f, tid, total, aux);
  }
};

template <typename GraphTy>
void readGraphDispatch(GraphTy& graph, read_with_aux_graph_tag tag,
                       const std::string& filename) {
  FileGraph f;
  f.fromFileInterleaved<typename GraphTy::file_edge_data_type>(filename);
  readGraphDispatch(graph, tag, f);
}

template <typename GraphTy>
void readGraphDispatch(GraphTy& graph, read_with_aux_graph_tag, FileGraph& f) {
  typedef typename GraphTy::ReadGraphAuxData Aux;

  Aux aux;
  graph.allocateFrom(f, aux);

  ReadGraphConstructNodesFrom<GraphTy, Aux> nodeReader(graph, f, aux);
  galois::on_each(nodeReader);

  ReadGraphConstructEdgesFrom<GraphTy, Aux> edgeReader(graph, f, aux);
  galois::on_each(edgeReader);
}

template <typename GraphTy, typename Aux>
struct ReadGraphConstructOutEdgesFrom {
  GraphTy& graph;
  FileGraph& f;
  Aux& aux;
  ReadGraphConstructOutEdgesFrom(GraphTy& g, FileGraph& _f, Aux& a)
      : graph(g), f(_f), aux(a) {}
  void operator()(unsigned tid, unsigned total) {
    graph.constructOutEdgesFrom(f, tid, total, aux);
  }
};

template <typename GraphTy, typename Aux>
struct ReadGraphConstructInEdgesFrom {
  GraphTy& graph;
  FileGraph& f;
  Aux& aux;
  ReadGraphConstructInEdgesFrom(GraphTy& g, FileGraph& _f, Aux& a)
      : graph(g), f(_f), aux(a) {}
  void operator()(unsigned tid, unsigned total) {
    graph.constructInEdgesFrom(f, tid, total, aux);
  }
};

template <typename GraphTy>
void readGraphDispatch(GraphTy& graph, read_with_aux_first_graph_tag,
                       FileGraph& f) {
  typedef typename GraphTy::ReadGraphAuxData Aux;
  constexpr static const bool profile = false;

  galois::CondStatTimer<profile> TAlloc("AllocateAux");
  TAlloc.start();
  Aux* auxPtr = new Aux;
  graph.allocateFrom(f, *auxPtr);
  TAlloc.stop();

  galois::CondStatTimer<profile> TNode("ConstructNode");
  TNode.start();
  ReadGraphConstructNodesFrom<GraphTy, Aux> nodeReader(graph, f, *auxPtr);
  galois::on_each(nodeReader);
  TNode.stop();

  galois::CondStatTimer<profile> TOutEdge("ConstructOutEdge");
  TOutEdge.start();
  ReadGraphConstructOutEdgesFrom<GraphTy, Aux> outEdgeReader(graph, f, *auxPtr);
  galois::on_each(outEdgeReader);
  TOutEdge.stop();

  galois::CondStatTimer<profile> TInEdge("ConstructInEdge");
  TInEdge.start();
  ReadGraphConstructInEdgesFrom<GraphTy, Aux> inEdgeReader(graph, f, *auxPtr);
  galois::on_each(inEdgeReader);
  TInEdge.stop();

  galois::CondStatTimer<profile> TDestruct("DestructAux");
  TDestruct.start();
  delete auxPtr;
  TDestruct.stop();
}

template <typename GraphTy>
void readGraphDispatch(GraphTy& graph, read_with_aux_first_graph_tag tag,
                       const std::string& filename) {
  FileGraph f;
  f.fromFileInterleaved<typename GraphTy::file_edge_data_type>(filename);
  readGraphDispatch(graph, tag, f);
}

template <typename GraphTy>
void readGraphDispatch(GraphTy& graph, read_lc_inout_graph_tag,
                       const std::string& f1, const std::string& f2) {
  graph.createAsymmetric();

  typename GraphTy::out_graph_type::read_tag tag1;
  readGraphDispatch(graph, tag1, f1);

  typename GraphTy::in_graph_type::read_tag tag2;
  readGraphDispatch(graph.inGraph, tag2, f2);
}

template <typename GraphTy>
void readGraphDispatch(GraphTy& graph, read_lc_inout_graph_tag, FileGraph& f1,
                       FileGraph& f2) {
  graph.createAsymmetric();

  typename GraphTy::out_graph_type::read_tag tag1;
  readGraphDispatch(graph, tag1, f1);

  typename GraphTy::in_graph_type::read_tag tag2;
  readGraphDispatch(graph.inGraph, tag2, f2);
}

template <typename GraphTy>
void readGraphDispatch(GraphTy& graph, read_lc_inout_graph_tag, FileGraph& f1) {
  typename GraphTy::out_graph_type::read_tag tag1;
  readGraphDispatch(graph, tag1, f1);
}

template <typename GraphTy>
void readGraphDispatch(GraphTy& graph, read_lc_inout_graph_tag,
                       const std::string& f1) {
  typename GraphTy::out_graph_type::read_tag tag1;
  readGraphDispatch(graph, tag1, f1);
}

} // namespace graphs
} // namespace galois

#endif
