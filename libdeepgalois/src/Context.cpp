/**
 * Based on common.hpp file of the Caffe deep learning library.
 */
#include "deepgalois/Context.h"
#include "deepgalois/utils.h"
#include "deepgalois/configs.h"
#include "galois/Galois.h"

namespace deepgalois {

Context::Context() : Context(false) {}

Context::~Context() {}

size_t Context::read_graph(bool selfloop) {
  std::string filename = path + dataset + ".csgr";
  std::string filetype = "gr";
  galois::StatTimer Tread("GraphReadingTime");
  Tread.start();
  if (filetype == "bin") {
    graph_cpu->readGraph(dataset);
  } else if (filetype == "gr") {
    graph_cpu            = new Graph();
    std::string filename = path + dataset + ".csgr";
    printf("Reading .gr file: %s\n", filename.c_str());
    if (selfloop) {
      galois::gWarn("SELF LOOPS NOT SUPPORTED AT THIS TIME");
      Graph graph_temp;
      // galois::graphs::readGraph(graph_temp, filename);
      graph_temp.readGraph(dataset);
      add_selfloop(graph_temp, *graph_cpu);
      is_selfloop_added = selfloop;
      //} else galois::graphs::readGraph(*graph_cpu, filename);
    } else {
      graph_cpu->readGraph(dataset);
      galois::gPrint("graph read size ", graph_cpu->size());
    }
    // TODO dist version of self loop
  } else {
    GALOIS_DIE("unknown file format for readgraph");
  }
  Tread.stop();

  auto g = getGraphPointer();
  galois::gPrint("num_vertices ", g->size(), " num_edges ", g->sizeEdges(),
                 "\n");
  return g->size();
}

void Context::add_selfloop(Graph& og, Graph& g) {
  // TODO not actually implemented yet
  g.allocateFrom(og.size(), og.size() + og.sizeEdges());
  g.constructNodes();
  // for (size_t src = 0; src < og.size(); src++) {
  //  //g.getData(src) = 1;
  //  auto begin = og.edge_begin(src);
  //  auto end = og.edge_end(src);
  //  g.fixEndEdge(src, end+src+1);
  //  bool self_inserted = false;
  //  if (begin == end) {
  //    new_edge_dst[begin+i] = i;
  //    continue;
  //  }
  //  for (auto e = begin; e != end; e++) {
  //    auto dst = og.getEdgeDst(e);
  //    if (!self_inserted) {
  //      if (dst > src) {
  //        g.constructEdge(e+src, src, 0);
  //        g.constructEdge(e+src+1, dst, 0);
  //        self_inserted = true;
  //      } else if (e+1 == end) {
  //        g.constructEdge(e+src+1, src, 0);
  //        g.constructEdge(e+src, dst, 0);
  //        self_inserted = true;
  //      } else g.constructEdge(e+src, dst, 0);
  //    } else g.constructEdge(e+src+1, dst, 0);
  //  }
  //}
}

// get current graph, also gets degrees of g
Graph* Context::getFullGraph() {
  Graph* g = getGraphPointer();
  g->degree_counting();
  return g;
}

} // namespace deepgalois
