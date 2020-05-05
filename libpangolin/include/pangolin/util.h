#ifndef UTIL_H
#define UTIL_H

#include "pangolin/scan.h"
#include "pangolin/mgraph.h"
#include "pangolin/res_man.h"

namespace util {

void print_graph(Graph& graph) {
  for (GNode n : graph) {
    std::cout << "vertex " << n << ": label = " << graph.getData(n)
              << ": degree = " << graph.get_degree(n) << " edgelist = [ ";
    for (auto e : graph.edges(n))
      std::cout << graph.getEdgeDst(e) << " ";
    std::cout << "]" << std::endl;
  }
}

void genGraph(MGraph& mg, Graph& g) {
  g.allocateFrom(mg.num_vertices(), mg.num_edges());
  g.constructNodes();
  for (size_t i = 0; i < mg.num_vertices(); i++) {
    g.getData(i)   = mg.get_label(i);
    auto row_begin = mg.get_offset(i);
    auto row_end   = mg.get_offset(i + 1);
    g.fixEndEdge(i, row_end);
    for (auto offset = row_begin; offset < row_end; offset++) {
      g.constructEdge(offset, mg.get_dest(offset), 0);
    }
  }
}
// relabel vertices by descending degree order (do not apply to weighted graphs)
void DegreeRanking(Graph& og, Graph& g) {
  std::cout << " Relabeling vertices by descending degree order\n";
  std::vector<IndexT> old_degrees(og.size(), 0);
  galois::do_all(
      galois::iterate(og.begin(), og.end()),
      [&](const auto& src) {
        old_degrees[src] = std::distance(og.edge_begin(src), og.edge_end(src));
      },
      galois::loopname("getOldDegrees"));

  size_t num_vertices = og.size();
  typedef std::pair<unsigned, IndexT> degree_node_p;
  std::vector<degree_node_p> degree_id_pairs(num_vertices);
  for (IndexT n = 0; n < num_vertices; n++)
    degree_id_pairs[n] = std::make_pair(old_degrees[n], n);
  std::sort(degree_id_pairs.begin(), degree_id_pairs.end(),
            std::greater<degree_node_p>());

  std::vector<IndexT> degrees(num_vertices, 0);
  std::vector<IndexT> new_ids(num_vertices);
  for (IndexT n = 0; n < num_vertices; n++) {
    degrees[n]                         = degree_id_pairs[n].first;
    new_ids[degree_id_pairs[n].second] = n;
  }
  std::vector<IndexT> offsets = PrefixSum(degrees);

  g.allocateFrom(og.size(), og.sizeEdges());
  g.constructNodes();
  galois::do_all(
      galois::iterate(og.begin(), og.end()),
      [&](const auto& src) {
        auto row_begin = offsets[src];
        g.fixEndEdge(src, row_begin + degrees[src]);
        IndexT offset = 0;
        for (auto e : og.edges(src)) {
          auto dst = og.getEdgeDst(e);
          g.constructEdge(row_begin + offset, new_ids[dst], 0);
          offset++;
        }
        assert(offset == degrees[src]);
      },
      galois::loopname("ConstructNewGraph"));
  g.sortAllEdgesByDst();
}

unsigned orientation(Graph& og, Graph& g) {
  galois::StatTimer Tdag("DAG");
  Tdag.start();
  std::cout << "Orientation enabled, using DAG\n";
  std::cout << "Assume the input graph is clean and symmetric (.csgr)\n";
  std::cout << "Before: num_vertices " << og.size() << " num_edges "
            << og.sizeEdges() << "\n";
  std::vector<IndexT> degrees(og.size(), 0);

  galois::do_all(
      galois::iterate(og.begin(), og.end()),
      [&](const auto& src) {
        degrees[src] = std::distance(og.edge_begin(src), og.edge_end(src));
      },
      galois::loopname("getOldDegrees"));

  unsigned max_degree = *(std::max_element(degrees.begin(), degrees.end()));
  std::vector<IndexT> new_degrees(og.size(), 0);

  galois::do_all(
      galois::iterate(og.begin(), og.end()),
      [&](const auto& src) {
        for (auto e : og.edges(src)) {
          auto dst = og.getEdgeDst(e);
          if (degrees[dst] > degrees[src] ||
              (degrees[dst] == degrees[src] && dst > src)) {
            new_degrees[src]++;
          }
        }
      },
      galois::loopname("getNewDegrees"));

  std::vector<IndexT> offsets = PrefixSum(new_degrees);
  assert(offsets[og.size()] == og.sizeEdges() / 2);

  g.allocateFrom(og.size(), og.sizeEdges() / 2);
  g.constructNodes();

  galois::do_all(
      galois::iterate(og.begin(), og.end()),
      [&](const auto& src) {
        g.getData(src) = 0;
        auto row_begin = offsets[src];
        g.fixEndEdge(src, row_begin + new_degrees[src]);
        IndexT offset = 0;
        for (auto e : og.edges(src)) {
          auto dst = og.getEdgeDst(e);
          if (degrees[dst] > degrees[src] ||
              (degrees[dst] == degrees[src] && dst > src)) {
            g.constructEdge(row_begin + offset, dst, 0);
            offset++;
          }
        }
        assert(offset == new_degrees[src]);
      },
      galois::loopname("ConstructNewGraph"));

  g.sortAllEdgesByDst();
  Tdag.stop();
  return max_degree;
}

// relabel is needed when we use DAG as input graph, and it is disabled when we
// use symmetrized graph
unsigned read_graph(Graph& graph, std::string filetype, std::string filename,
                    bool need_dag = false) {
  MGraph mgraph(need_dag);
  unsigned max_degree = 0;
  if (filetype == "txt") {
    printf("Reading .lg file: %s\n", filename.c_str());
    mgraph.read_txt(filename.c_str());
    genGraph(mgraph, graph);
  } else if (filetype == "adj") {
    printf("Reading .adj file: %s\n", filename.c_str());
    mgraph.read_adj(filename.c_str());
    genGraph(mgraph, graph);
  } else if (filetype == "mtx") {
    printf("Reading .mtx file: %s\n", filename.c_str());
    mgraph.read_mtx(filename.c_str(), true); // symmetrize
    genGraph(mgraph, graph);
  } else if (filetype == "gr") {
    printf("Reading .gr file: %s\n", filename.c_str());
    if (need_dag) {
      Graph g_temp;
      galois::graphs::readGraph(g_temp, filename);
      max_degree = orientation(g_temp, graph);
    } else {
      galois::graphs::readGraph(graph, filename);
      galois::do_all(
          galois::iterate(graph.begin(), graph.end()),
          [&](const auto& vid) {
            graph.getData(vid) = 1;
            // for (auto e : graph.edges(n)) graph.getEdgeData(e) = 1;
          },
          galois::loopname("assignVertexLabels"));
      std::vector<unsigned> degrees(graph.size());
      galois::do_all(
          galois::iterate(graph.begin(), graph.end()),
          [&](const auto& vid) {
            degrees[vid] =
                std::distance(graph.edge_begin(vid), graph.edge_end(vid));
          },
          galois::loopname("computeMaxDegree"));
      max_degree = *(std::max_element(degrees.begin(), degrees.end()));
    }
  } else {
    printf("Unkown file format\n");
    exit(1);
  }
  // print_graph(graph);
  if (filetype != "gr") {
    max_degree = mgraph.get_max_degree();
    mgraph.clean();
  }
  printf("max degree = %u\n", max_degree);
  return max_degree;
}

} // namespace util
#endif
