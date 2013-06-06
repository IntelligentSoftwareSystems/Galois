/** Useful classes and methods for graphs  -*- C++ -*-
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
#ifndef GALOIS_GRAPH_UTIL_H
#define GALOIS_GRAPH_UTIL_H

#include "Galois/Galois.h"
#include "Galois/Graph/LCGraph.h"

namespace Galois {
namespace Graph {

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
  f.structureFromFileInterleaved<typename GraphTy::edge_data_type>(filename);
  readGraphDispatch(graph, tag, f);
}

template<typename GraphTy>
struct ReadGraphDispatchHelper1 {
  GraphTy& graph;
  FileGraph& f;
  ReadGraphDispatchHelper1(GraphTy& g, FileGraph& _f): graph(g), f(_f) { }
  void operator()(unsigned tid, unsigned total) {
    graph.constructFrom(f, tid, total);
  }
};

template<typename GraphTy>
struct ReadGraphDispatchHelper2 {
  GraphTy& graph;
  FileGraph& f;
  ReadGraphDispatchHelper2(GraphTy& g, FileGraph& _f): graph(g), f(_f) { }
  void operator()(unsigned tid, unsigned total) {
    graph.constructNodesFrom(f, tid, total);
  }
};

template<typename GraphTy>
struct ReadGraphDispatchHelper3 {
  GraphTy& graph;
  FileGraph& f;
  ReadGraphDispatchHelper3(GraphTy& g, FileGraph& _f): graph(g), f(_f) { }
  void operator()(unsigned tid, unsigned total) {
    graph.constructEdgesFrom(f, tid, total);
  }
};

  
template<typename GraphTy>
void readGraphDispatch(GraphTy& graph, read_default_graph_tag, FileGraph& f) {
  graph.allocateFrom(f);

  Galois::on_each(ReadGraphDispatchHelper1<GraphTy>(graph, f));
}

template<typename GraphTy>
void readGraphDispatch(GraphTy& graph, read_lc_linear_graph_tag tag, const std::string& filename) {
  FileGraph f;
  f.structureFromFileInterleaved<typename GraphTy::edge_data_type>(filename);
  readGraphDispatch(graph, tag, f);
}

template<typename GraphTy>
void readGraphDispatch(GraphTy& graph, read_lc_linear_graph_tag, FileGraph& f) {
  graph.allocateFrom(f);

  Galois::on_each(ReadGraphDispatchHelper2<GraphTy>(graph, f));

  Galois::on_each(ReadGraphDispatchHelper3<GraphTy>(graph, f));
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
void readGraphDispatch(GraphTy& graph, read_lc_inout_graph_tag, const std::string& f1) { 
  typename GraphTy::out_graph_type::read_tag tag1;
  readGraphDispatch(graph, tag1, f1);
}

} // end namespace
} // end namespace

#endif
