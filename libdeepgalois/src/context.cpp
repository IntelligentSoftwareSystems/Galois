/**
 * Based on common.hpp file of the Caffe deep learning library.
 */
#include "deepgalois/context.h"
#include "deepgalois/utils.h"
#include "deepgalois/configs.h"

namespace deepgalois {

#ifdef CPU_ONLY
Context::Context() : n(0), num_classes(0), 
  feat_len(0), is_single_class(true), 
  is_selfloop_added(false), use_subgraph(false),
  h_labels(NULL), h_labels_subg(NULL), 
  h_feats(NULL), h_feats_subg(NULL),
  d_labels(NULL), d_labels_subg(NULL),
  d_feats(NULL), d_feats_subg(NULL),
  norm_factor(NULL) {}

Context::~Context() {
  if (h_labels) delete h_labels;
  if (h_feats) delete h_feats;
  if (norm_factor) delete norm_factor;
}

size_t Context::read_graph(std::string dataset_str, bool selfloop) {
  n = read_graph_cpu(dataset_str, "gr", selfloop);
  return n;
}

void Context::createSubgraph() {
  subgraph_cpu = new Graph(); 
}

// generate labels for the subgraph, m is subgraph size
void Context::gen_subgraph_labels(size_t m, const mask_t *masks) {
  if (h_labels_subg == NULL) h_labels_subg = new label_t[m];
  size_t count = 0;
  for (size_t i = 0; i < n; i++) {
    if (masks[i] == 1) {
      if (is_single_class) {
        h_labels_subg[count] = h_labels[i];
      } else {
        std::copy(h_labels+i*num_classes, h_labels+(i+1)*num_classes, h_labels_subg+count*num_classes);
	  }
      count ++;
	}
  }
}

// generate input features for the subgraph, m is subgraph size
void Context::gen_subgraph_feats(size_t m, const mask_t *masks) {
  size_t count = 0;
  if (h_feats_subg == NULL) h_feats_subg = new float_t[m*feat_len];
  for (size_t i = 0; i < n; i++) {
    if (masks[i] == 1) {
      std::copy(h_feats+i*feat_len, h_feats+(i+1)*feat_len, h_feats_subg+count*feat_len);
      count ++;
	}
  }
}

size_t Context::read_graph_cpu(std::string dataset_str, std::string filetype, bool selfloop) {
  galois::StatTimer Tread("GraphReadingTime");
  Tread.start();
  graph_cpu = new Graph(); 
  if (filetype == "el") {
    std::string filename = path + dataset_str + ".el";
    printf("Reading .el file: %s\n", filename.c_str());
    read_edgelist(filename.c_str(), true); // symmetrize
  } else if (filetype == "gr") {
    std::string filename = path + dataset_str + ".csgr";
    printf("Reading .gr file: %s\n", filename.c_str());
    if (selfloop) {
      Graph graph_temp;
      galois::graphs::readGraph(graph_temp, filename);
      add_selfloop(graph_temp, *graph_cpu);
      is_selfloop_added = selfloop;
    } else galois::graphs::readGraph(*graph_cpu, filename);
// TODO dist version of self loop
  } else {
    printf("Unkown file format\n");
    exit(1);
  }
  Tread.stop();
  std::cout << "num_vertices " << graph_cpu->size() << " num_edges "
            << graph_cpu->sizeEdges() << "\n";
  return graph_cpu->size();
}

void Context::add_selfloop(Graph &og, Graph &g) {
  g.allocateFrom(og.size(), og.size()+og.sizeEdges());
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

void Context::norm_factor_counting(size_t g_size) {
  Graph *g = graph_cpu;
  if (use_subgraph) g = subgraph_cpu;
  if (norm_factor == NULL) norm_factor = new float_t[g_size];
  galois::do_all(galois::iterate((size_t)0, g_size), [&](auto v) {
    auto degree  = std::distance(g->edge_begin(v), g->edge_end(v));
    float_t temp = std::sqrt(float_t(degree));
    if (temp == 0.0) norm_factor[v] = 0.0;
    else norm_factor[v] = 1.0 / temp;
  }, galois::loopname("NormCounting"));
}

void Context::read_edgelist(const char* filename, bool symmetrize, bool add_self_loop) {
  std::ifstream in;
  std::string line;
  in.open(filename, std::ios::in);
  size_t m, n;
  in >> m >> n >> std::ws;
  size_t num_vertices_ = m;
  size_t num_edges_    = 0;
  std::cout << "num_vertices " << num_vertices_ << "\n";
  std::vector<std::set<uint32_t> > vertices(m);
  for (size_t i = 0; i < n; i++) {
    std::set<uint32_t> neighbors;
    if (add_self_loop) neighbors.insert(i);
    vertices.push_back(neighbors);
  }
  while (std::getline(in, line)) {
    std::istringstream edge_stream(line);
    VertexID u, v;
    edge_stream >> u;
    edge_stream >> v;
    vertices[u].insert(v);
    if (symmetrize) vertices[v].insert(u);
  }
  in.close();
  for (size_t i = 0; i < n; i++) num_edges_ += vertices[i].size();
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
  EdgeID *colidx_ = new EdgeID[num_edges_];
  VertexID *rowptr_ = new VertexID[num_vertices_ + 1];
  for (size_t i = 0; i < num_vertices_ + 1; i++)
    rowptr_[i] = offsets[i];
  for (size_t i = 0; i < num_vertices_; i++) {
    for (auto dst : vertices[i])
        colidx_[offsets[i]++] = dst;
  }

  graph_cpu->allocateFrom(num_vertices_, num_edges_);
  graph_cpu->constructNodes();
  for (size_t i = 0; i < num_vertices_; i++) {
    auto row_begin = rowptr_[i];
    auto row_end = rowptr_[i+1];
    graph_cpu->fixEndEdge(i, row_end);
    for (auto offset = row_begin; offset < row_end; offset++)
      graph_cpu->constructEdge(offset, colidx_[offset], 0);
  }
}

#endif

// labels contain the ground truth (e.g. vertex classes) for each example
// (num_examples x 1). Note that labels is not one-hot encoded vector and it can
// be computed as y.argmax(axis=1) from one-hot encoded vector (y) of labels if
// required.
size_t Context::read_labels(std::string dataset_str) {
  std::cout << "Reading labels ... ";
  Timer t_read;
  t_read.Start();
  std::string filename = path + dataset_str + "-labels.txt";
  std::ifstream in;
  std::string line;
  in.open(filename, std::ios::in);
  size_t m; // m: number of samples
  in >> m >> num_classes >> std::ws;
  assert(m == n);
  if (is_single_class) {
    std::cout << "Using single-class (one-hot) labels\n";
    h_labels = new label_t[m]; // single-class (one-hot) label for each vertex: N x 1
  } else {
    std::cout << "Using multi-class labels\n";
   h_labels = new label_t[m*num_classes]; // multi-class label for each vertex: N x E
  }
  unsigned v = 0;
  while (std::getline(in, line)) {
    std::istringstream label_stream(line);
    unsigned x;
    for (size_t idx = 0; idx < num_classes; ++idx) {
      label_stream >> x;
      if (is_single_class) {
        if (x != 0) {
          h_labels[v] = idx;
          break;
        }
      } else {
        h_labels[v*num_classes+idx] = x;
      }
    }
    v++;
  }
  in.close();
  t_read.Stop();
  // print the number of vertex classes
  std::cout << "Done, unique label counts: " << num_classes
            << ", time: " << t_read.Millisecs() << " ms\n";
  //for (auto i = 0; i < 10; i ++) std::cout << "labels[" << i << "] = " << unsigned(labels[i]) << "\n";
  return num_classes;
}

//! Read features, return the length of a feature vector
//! Features are stored in the Context class
size_t Context::read_features(std::string dataset_str, std::string filetype) {
  //filetype = "txt";
  std::cout << "Reading features ... ";
  Timer t_read;
  t_read.Start();
  size_t m; // m = number of vertices
  std::string filename = path + dataset_str + ".ft";
  std::ifstream in;

  if (filetype == "bin") {
    std::string file_dims = path + dataset_str + "-dims.txt";
    std::ifstream ifs;
    ifs.open(file_dims, std::ios::in);
    ifs >> m >> feat_len >> std::ws;
    ifs.close();
  } else {
    in.open(filename, std::ios::in);
    in >> m >> feat_len >> std::ws;
  }
  std::cout << "N x D: " << m << " x " << feat_len << "\n";
  h_feats = new float_t[m * feat_len];
  if (filetype == "bin") {
    filename = path + dataset_str + "-feats.bin";
    in.open(filename, std::ios::binary|std::ios::in);
    in.read((char*)h_feats, sizeof(float_t) * m * feat_len);
  } else {
    std::string line;
    while (std::getline(in, line)) {
      std::istringstream edge_stream(line);
      unsigned u, v;
      float_t w;
      edge_stream >> u;
      edge_stream >> v;
      edge_stream >> w;
      h_feats[u * feat_len + v] = w;
    }
  }
  in.close();
  t_read.Stop();
  std::cout << "Done, feature length: " << feat_len
            << ", time: " << t_read.Millisecs() << " ms\n";
  //for (auto i = 0; i < 6; i ++) 
    //for (auto j = 0; j < 6; j ++) 
      //std::cout << "feats[" << i << "][" << j << "] = " << h_feats[i*feat_len+j] << "\n";
  return feat_len;
}

//! Get masks from datafile where first line tells range of
//! set to create mask from
size_t Context::read_masks(std::string dataset_str, std::string mask_type,
                  size_t n, size_t& begin, size_t& end, mask_t* masks) {
  bool dataset_found = false;
  for (int i = 0; i < NUM_DATASETS; i++) {
    if (dataset_str == dataset_names[i]) {
      dataset_found = true;
      break;
    }
  }
  if (!dataset_found) {
    std::cout << "Dataset currently not supported\n";
    exit(1);
  }
  size_t i             = 0;
  size_t sample_count  = 0;
  std::string filename = path + dataset_str + "-" + mask_type + "_mask.txt";
  // std::cout << "Reading " << filename << "\n";
  std::ifstream in;
  std::string line;
  in.open(filename, std::ios::in);
  in >> begin >> end >> std::ws;
  while (std::getline(in, line)) {
    std::istringstream mask_stream(line);
    if (i >= begin && i < end) {
      unsigned mask = 0;
      mask_stream >> mask;
      if (mask == 1) {
        masks[i] = 1;
        sample_count++;
      }
    }
    i++;
  }
  std::cout << mask_type + "_mask range: [" << begin << ", " << end
    << ") Number of valid samples: " << sample_count << " (" 
    << (float)sample_count/(float)n*(float)100 << "\%)\n";
  in.close();
  return sample_count;
}

/*
inline void init_features(size_t dim, vec_t &x) {
    std::default_random_engine rng;
    std::uniform_real_distribution<feature_t> dist(0, 0.1);
    for (size_t i = 0; i < dim; ++i)
        x[i] = dist(rng);
}
*/
} // end deepgalois namespace
