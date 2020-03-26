#include "Lonestar/mgraph.h"

void print_graph(Graph& graph) {
  for (GNode n : graph) {
    std::cout << "vertex " << n << ": label = " << graph.getData(n)
              << " edgelist = [ ";
    for (auto e : graph.edges(n))
      std::cout << graph.getEdgeDst(e) << " ";
    std::cout << "]" << std::endl;
  }
}

void genGraph(MGraph& mg, Graph& g) {
  g.allocateFrom(mg.num_vertices(), mg.num_edges());
  g.constructNodes();
  for (int i = 0; i < mg.num_vertices(); i++) {
    g.getData(i)  = mg.get_label(i);
    int row_begin = mg.get_offset(i);
    int row_end   = mg.get_offset(i + 1);
    // int num_neighbors = mg.out_degree(i);
    g.fixEndEdge(i, row_end);
    for (int offset = row_begin; offset < row_end; offset++) {
#ifdef ENABLE_LABEL
      // g.constructEdge(offset, mg.get_dest(offset), mg.get_weight(offset));
      g.constructEdge(offset, mg.get_dest(offset),
                      1); // do not consider edge labels currently
#else
      g.constructEdge(offset, mg.get_dest(offset), 0);
#endif
    }
  }
}

// relabel is needed when we use DAG as input graph, and it is disabled when we
// use symmetrized graph
int read_graph(Graph& graph, std::string filetype, std::string filename,
               bool need_relabel = false) {
  MGraph mgraph(need_relabel);
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
    if (need_relabel) {
      Graph g_temp;
      galois::graphs::readGraph(g_temp, filename);
      for (GNode n : g_temp)
        g_temp.getData(n) = 1;
      mgraph.read_gr(g_temp); // symmetrize
      genGraph(mgraph, graph);
    } else {
      galois::graphs::readGraph(graph, filename);
      for (GNode n : graph) {
#ifdef ENABLE_LABEL
        graph.getData(n) = rand() % 10 + 1;
        for (auto e : graph.edges(n))
          graph.getEdgeData(e) = 1;
#else
        graph.getData(n) = 1;
#endif
      }
    }
  } else {
    printf("Unkown file format\n");
    exit(1);
  }
  // print_graph(graph);
  int core = 0;
  if (need_relabel)
    core = mgraph.get_core();
  return core;
}

/*
// construct the edge-list for later use. May not be necessary if Galois has
this support void construct_edgelist(Graph& graph, std::vector<LabeledEdge>
&edgelist) { for (Graph::iterator it = graph.begin(); it != graph.end(); it ++)
{
        // for each vertex
        GNode src = *it;
        auto& src_label = graph.getData(src);
        Graph::edge_iterator first = graph.edge_begin(src,
galois::MethodFlag::UNPROTECTED); Graph::edge_iterator last =
graph.edge_end(src, galois::MethodFlag::UNPROTECTED);
        // foe each edge of this vertex
        for (auto e = first; e != last; ++ e) {
            GNode dst = graph.getEdgeDst(e);
            auto& dst_label = graph.getData(dst);
            LabeledEdge edge(src, dst, src_label, dst_label);
            edgelist.push_back(edge);
        }
    }
    assert(edgelist.size() == graph.sizeEdges());
}
*/
