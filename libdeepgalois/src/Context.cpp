/**
 * Based on common.hpp file of the Caffe deep learning library.
 */
#include "deepgalois/Context.h"
#include "deepgalois/utils.h"
#include "deepgalois/configs.h"
#include "galois/Galois.h"

namespace deepgalois {

Context::Context() : Context(false) {}

Context::~Context() {
  if (h_labels)
    delete[] h_labels;
  if (h_feats)
    delete[] h_feats;
  if (norm_factors)
    delete[] norm_factors;
  // if (h_feats_subg) delete[] h_feats_subg;
  // if (h_labels_subg) delete[] h_labels_subg;
  // if (norm_factors_subg) delete[] norm_factors_subg;
}

void Context::createSubgraphs(int num_subgraphs) {
  subgraphs_cpu.resize(num_subgraphs);
  for (int i = 0; i < num_subgraphs; i++)
    subgraphs_cpu[i] = new Graph();
}

// generate labels for the subgraph, m is subgraph size
void Context::gen_subgraph_labels(size_t m, const mask_t* masks) {
  // if (h_labels_subg == NULL) h_labels_subg = new label_t[m];
  if (is_single_class) {
    h_labels_subg.resize(m);
  } else {
    h_labels_subg.resize(m * num_classes);
  }
  size_t count = 0;
  for (size_t i = 0; i < n; i++) {
    if (masks[i] == 1) {
      if (is_single_class) {
        h_labels_subg[count] = h_labels[i];
      } else {
        std::copy(h_labels + i * num_classes, h_labels + (i + 1) * num_classes,
                  &h_labels_subg[count * num_classes]);
      }
      count++;
    }
  }
}

// generate input features for the subgraph, m is subgraph size
void Context::gen_subgraph_feats(size_t m, const mask_t* masks) {
  size_t count = 0;
  // if (h_feats_subg == NULL) h_feats_subg = new float_t[m*feat_len];
  h_feats_subg.resize(m * feat_len);
  for (size_t i = 0; i < n; i++) {
    if (masks[i] == 1) {
      std::copy(h_feats + i * feat_len, h_feats + (i + 1) * feat_len,
                &h_feats_subg[count * feat_len]);
      count++;
    }
  }
}

size_t Context::read_graph(bool selfloop) {
  std::string filename = path + dataset + ".csgr";
  std::string filetype = "gr";
  galois::StatTimer Tread("GraphReadingTime");
  Tread.start();
  if (filetype == "el") {
    filename = path + dataset + ".el";
    printf("Reading .el file: %s\n", filename.c_str());
    read_edgelist(filename.c_str(), true); // symmetrize
  } else if (filetype == "bin") {
    graph_cpu->readGraph(dataset);
  } else if (filetype == "gr") {
    graph_cpu            = new Graph();
    std::string filename = path + dataset + ".csgr";
    printf("Reading .gr file: %s\n", filename.c_str());
    if (selfloop) {
      Graph graph_temp;
      // galois::graphs::readGraph(graph_temp, filename);
      graph_temp.readGraph(dataset);
      add_selfloop(graph_temp, *graph_cpu);
      is_selfloop_added = selfloop;
      //} else galois::graphs::readGraph(*graph_cpu, filename);
    } else
      graph_cpu->readGraph(dataset);
    // TODO dist version of self loop
  } else {
    printf("Unkown file format\n");
    exit(1);
  }
  Tread.stop();
  auto g = getGraphPointer();
  std::cout << "num_vertices " << g->size() << " num_edges " << g->sizeEdges()
            << "\n";
  n = g->size();
  return n;
}

void Context::add_selfloop(Graph& og, Graph& g) {
  g.allocateFrom(og.size(), og.size() + og.sizeEdges());
  g.constructNodes();
  /*
  for (size_t src = 0; src < og.size(); src++) {
    //g.getData(src) = 1;
    auto begin = og.edge_begin(src);
    auto end = og.edge_end(src);
    g.fixEndEdge(src, end+src+1);
    bool self_inserted = false;
    if (begin == end) {
      new_edge_dst[begin+i] = i;
      continue;
    }
    for (auto e = begin; e != end; e++) {
      auto dst = og.getEdgeDst(e);
      if (!self_inserted) {
        if (dst > src) {
          g.constructEdge(e+src, src, 0);
          g.constructEdge(e+src+1, dst, 0);
          self_inserted = true;
        } else if (e+1 == end) {
          g.constructEdge(e+src+1, src, 0);
          g.constructEdge(e+src, dst, 0);
          self_inserted = true;
        } else g.constructEdge(e+src, dst, 0);
      } else g.constructEdge(e+src+1, dst, 0);
    }
  }
  //*/
}

void Context::alloc_norm_factor() {
  Graph* g = getGraphPointer();
  if (norm_factors == NULL)
#ifdef USE_MKL
    norm_factors = new float_t[g->sizeEdges()];
#else
    norm_factors = new float_t[g->size()];
#endif
}

void Context::alloc_subgraph_norm_factor(int subg_id) {
  Graph* g = getSubgraphPointer(subg_id);
  // if (norm_factors_subg == NULL)
#ifdef USE_MKL
  // norm_factors_subg = new float_t[g->sizeEdges()];
  norm_factors_subg.resize(g->sizeEdges());
#else
  norm_factors_subg.resize(g->size());
  // norm_factors_subg = new float_t[g->size()];
#endif
}

void Context::norm_factor_computing(bool is_subgraph, int subg_id) {
  Graph* g;
  float_t* constants;
  if (!is_subgraph) {
    g = getGraphPointer();
    alloc_norm_factor();
    constants = norm_factors;
  } else {
    g = getSubgraphPointer(subg_id);
    alloc_subgraph_norm_factor(subg_id);
    constants = get_norm_factors_subg_ptr();
  }
  auto g_size = g->size();
  g->degree_counting();
#ifdef USE_MKL
  galois::do_all(
      galois::iterate((size_t)0, g_size),
      [&](auto i) {
        float_t c_i = std::sqrt(float_t(g->get_degree(i)));
        for (auto e = g->edge_begin(i); e != g->edge_end(i); e++) {
          const auto j = g->getEdgeDst(e);
          float_t c_j  = std::sqrt(float_t(g->get_degree(j)));
          if (c_i == 0.0 || c_j == 0.0)
            constants[e] = 0.0;
          else
            constants[e] = 1.0 / (c_i * c_j);
        }
      },
      galois::loopname("NormCountingEdge"));
#else
  galois::do_all(
      galois::iterate((size_t)0, g_size),
      [&](auto v) {
        auto degree = g->get_degree(v);
        float_t temp = std::sqrt(float_t(degree));
        if (temp == 0.0)
          constants[v] = 0.0;
        else
          constants[v] = 1.0 / temp;
      },
      galois::loopname("NormCountingVertex"));
#endif
}

void Context::read_edgelist(const char* filename, bool symmetrize,
                            bool add_self_loop) {
  std::ifstream in;
  std::string line;
  in.open(filename, std::ios::in);
  size_t m, n;
  in >> m >> n >> std::ws;
  size_t num_vertices_ = m;
  size_t num_edges_    = 0;
  std::cout << "num_vertices " << num_vertices_ << "\n";
  std::vector<std::set<uint32_t>> vertices(m);
  for (size_t i = 0; i < n; i++) {
    std::set<uint32_t> neighbors;
    if (add_self_loop)
      neighbors.insert(i);
    vertices.push_back(neighbors);
  }
  while (std::getline(in, line)) {
    std::istringstream edge_stream(line);
    VertexID u, v;
    edge_stream >> u;
    edge_stream >> v;
    vertices[u].insert(v);
    if (symmetrize)
      vertices[v].insert(u);
  }
  in.close();
  for (size_t i = 0; i < n; i++)
    num_edges_ += vertices[i].size();
  std::cout << "num_edges " << num_edges_ << "\n";

  std::vector<uint32_t> degrees;
  degrees.resize(num_vertices_);
  std::fill(degrees.begin(), degrees.end(), 0);
  for (size_t i = 0; i < num_vertices_; i++)
    degrees[i] = vertices[i].size();
  std::vector<uint32_t> offsets(degrees.size() + 1);
  uint32_t total = 0;
  for (size_t n = 0; n < degrees.size(); n++) {
    offsets[n] = total;
    total += degrees[n];
  }
  offsets[degrees.size()] = total;
  degrees.clear();
  assert(num_edges_ == offsets[num_vertices_]);
  EdgeID* colidx_   = new EdgeID[num_edges_];
  VertexID* rowptr_ = new VertexID[num_vertices_ + 1];
  for (size_t i = 0; i < num_vertices_ + 1; i++)
    rowptr_[i] = offsets[i];
  for (size_t i = 0; i < num_vertices_; i++) {
    for (auto dst : vertices[i])
      colidx_[offsets[i]++] = dst;
  }

  auto g = getGraphPointer();
  g->allocateFrom(num_vertices_, num_edges_);
  g->constructNodes();
  for (size_t i = 0; i < num_vertices_; i++) {
    auto row_begin = rowptr_[i];
    auto row_end   = rowptr_[i + 1];
    g->fixEndEdge(i, row_end);
    for (auto offset = row_begin; offset < row_end; offset++)
      g->constructEdge(offset, colidx_[offset], 0);
  }
}

/*
inline void init_features(size_t dim, vec_t &x) {
    std::default_random_engine rng;
    std::uniform_real_distribution<feature_t> dist(0, 0.1);
    for (size_t i = 0; i < dim; ++i)
        x[i] = dist(rng);
}
*/

} // namespace deepgalois
