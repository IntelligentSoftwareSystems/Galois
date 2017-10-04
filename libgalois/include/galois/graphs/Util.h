/** Useful classes and methods for graphs  -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */

#ifndef GALOIS_GRAPH_UTIL_H
#define GALOIS_GRAPH_UTIL_H

#include "galois/Galois.h"
#include "galois/graphs/FileGraph.h"
#include "galois/graphs/Details.h"

namespace galois {
namespace graphs {

/**
 * Allocates and constructs a graph from a file. Tries to balance
 * memory evenly across system. Cannot be called during parallel
 * execution.
 */
template<typename GraphTy, typename... Args>
void readGraph(GraphTy& graph, Args&&... args) {
  typename GraphTy::read_tag tag;
  readGraphDispatch(graph, tag, std::forward<Args>(args)...);
}

template<typename GraphTy>
void readGraphDispatch(GraphTy& graph, read_default_graph_tag tag, const std::string& filename) {
  FileGraph f;
  f.fromFileInterleaved<typename GraphTy::file_edge_data_type>(filename);
  readGraphDispatch(graph, tag, f);
}

template<typename GraphTy>
struct ReadGraphConstructFrom {
  GraphTy& graph;
  FileGraph& f;
  ReadGraphConstructFrom(GraphTy& g, FileGraph& _f): graph(g), f(_f) { }
  void operator()(unsigned tid, unsigned total) {
    graph.constructFrom(f, tid, total);
  }
};

template<typename GraphTy>
void readGraphDispatch(GraphTy& graph, read_default_graph_tag, FileGraph& f) {
  graph.allocateFrom(f);

  ReadGraphConstructFrom<GraphTy> reader(graph, f);
  galois::on_each(reader);
}


template<typename GraphTy, typename Aux>
struct ReadGraphConstructNodesFrom {
  GraphTy& graph;
  FileGraph& f;
  Aux& aux;
  ReadGraphConstructNodesFrom(GraphTy& g, FileGraph& _f, Aux& a): graph(g), f(_f), aux(a) { }
  void operator()(unsigned tid, unsigned total) {
    graph.constructNodesFrom(f, tid, total, aux);
  }
};

template<typename GraphTy, typename Aux>
struct ReadGraphConstructEdgesFrom {
  GraphTy& graph;
  FileGraph& f;
  Aux& aux;
  ReadGraphConstructEdgesFrom(GraphTy& g, FileGraph& _f, Aux& a): graph(g), f(_f), aux(a) { }
  void operator()(unsigned tid, unsigned total) {
    graph.constructEdgesFrom(f, tid, total, aux);
  }
};

template<typename GraphTy>
void readGraphDispatch(GraphTy& graph, read_with_aux_graph_tag tag, const std::string& filename) {
  FileGraph f;
  f.fromFileInterleaved<typename GraphTy::file_edge_data_type>(filename);
  readGraphDispatch(graph, tag, f);
}

template<typename GraphTy>
void readGraphDispatch(GraphTy& graph, read_with_aux_graph_tag, FileGraph& f) {
  typedef typename GraphTy::ReadGraphAuxData Aux;

  Aux aux;
  graph.allocateFrom(f, aux);

  ReadGraphConstructNodesFrom<GraphTy, Aux> nodeReader(graph, f, aux);
  galois::on_each(nodeReader);

  ReadGraphConstructEdgesFrom<GraphTy, Aux> edgeReader(graph, f, aux);
  galois::on_each(edgeReader);
}


template<typename GraphTy, typename Aux>
struct ReadGraphConstructOutEdgesFrom {
  GraphTy& graph;
  FileGraph& f;
  Aux& aux;
  ReadGraphConstructOutEdgesFrom(GraphTy& g, FileGraph& _f, Aux& a): graph(g), f(_f), aux(a) { }
  void operator()(unsigned tid, unsigned total) {
    graph.constructOutEdgesFrom(f, tid, total, aux);
  }
};

template<typename GraphTy, typename Aux>
struct ReadGraphConstructInEdgesFrom {
  GraphTy& graph;
  FileGraph& f;
  Aux& aux;
  ReadGraphConstructInEdgesFrom(GraphTy& g, FileGraph& _f, Aux& a): graph(g), f(_f), aux(a) { }
  void operator()(unsigned tid, unsigned total) {
    graph.constructInEdgesFrom(f, tid, total, aux);
  }
};

template<typename GraphTy>
void readGraphDispatch(GraphTy& graph, read_with_aux_first_graph_tag, FileGraph& f) {
  typedef typename GraphTy::ReadGraphAuxData Aux;

  Aux aux;
  graph.allocateFrom(f, aux);

  ReadGraphConstructNodesFrom<GraphTy, Aux> nodeReader(graph, f, aux);
  galois::on_each(nodeReader);

  ReadGraphConstructOutEdgesFrom<GraphTy, Aux> outEdgeReader(graph, f, aux);
  galois::on_each(outEdgeReader);

  ReadGraphConstructInEdgesFrom<GraphTy, Aux> inEdgeReader(graph, f, aux);
  galois::on_each(inEdgeReader);
}

template<typename GraphTy>
void readGraphDispatch(GraphTy& graph, read_with_aux_first_graph_tag tag, const std::string& filename) {
  FileGraph f;
  f.fromFileInterleaved<typename GraphTy::file_edge_data_type>(filename);
  readGraphDispatch(graph, tag, f);
}

template<typename GraphTy>
void readGraphDispatch(GraphTy& graph, read_lc_inout_graph_tag, const std::string& f1, const std::string& f2) {
  graph.createAsymmetric();

  typename GraphTy::out_graph_type::read_tag tag1;
  readGraphDispatch(graph, tag1, f1);

  typename GraphTy::in_graph_type::read_tag tag2;
  readGraphDispatch(graph.inGraph, tag2, f2);
}

template<typename GraphTy>
void readGraphDispatch(GraphTy& graph, read_lc_inout_graph_tag, FileGraph& f1, FileGraph& f2) {
  graph.createAsymmetric();

  typename GraphTy::out_graph_type::read_tag tag1;
  readGraphDispatch(graph, tag1, f1);

  typename GraphTy::in_graph_type::read_tag tag2;
  readGraphDispatch(graph.inGraph, tag2, f2);
}

template<typename GraphTy>
void readGraphDispatch(GraphTy& graph, read_lc_inout_graph_tag, FileGraph& f1) {
  typename GraphTy::out_graph_type::read_tag tag1;
  readGraphDispatch(graph, tag1, f1);
}

template<typename GraphTy>
void readGraphDispatch(GraphTy& graph, read_lc_inout_graph_tag, const std::string& f1) {
  typename GraphTy::out_graph_type::read_tag tag1;
  readGraphDispatch(graph, tag1, f1);
}

} // end namespace
} // end namespace

#endif
