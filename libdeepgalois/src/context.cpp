/**
 * Based on common.hpp file of the Caffe deep learning library.
 */

#include "deepgalois/context.h"

namespace deepgalois {

#ifdef CPU_ONLY
Context::Context() {}
Context::~Context() {
  if (labels) delete labels;
  if (h_feats) delete h_feats;
  if (norm_factor) delete norm_factor;
}

size_t Context::read_graph(std::string dataset_str, bool selfloop) {
  n = read_graph_cpu(dataset_str, "gr", selfloop);
  return n;
}

size_t Context::read_graph_cpu(std::string dataset_str, std::string filetype, bool selfloop) {
  galois::StatTimer Tread("GraphReadingTime");
  Tread.start();
  graph_cpu = new Graph(); 
  if (filetype == "el") {
    std::string filename = path + dataset_str + ".el";
    printf("Reading .el file: %s\n", filename.c_str());
    LGraph lgraph;
    lgraph.read_edgelist(filename.c_str(), true); // symmetrize
    genGraph(lgraph, *graph_cpu);
    lgraph.clean();
  } else if (filetype == "gr") {
    std::string filename = path + dataset_str + ".csgr";
    printf("Reading .gr file: %s\n", filename.c_str());
    if (selfloop) {
      Graph graph_temp;
      galois::graphs::readGraph(graph_temp, filename);
      add_selfloop(graph_temp, *graph_cpu);
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

void Context::genGraph(LGraph& lg, Graph& g) {
  g.allocateFrom(lg.num_vertices(), lg.num_edges());
  g.constructNodes();
  for (size_t i = 0; i < lg.num_vertices(); i++) {
    g.getData(i)   = 1;
    auto row_begin = lg.get_offset(i);
    auto row_end   = lg.get_offset(i + 1);
    g.fixEndEdge(i, row_end);
    for (auto offset = row_begin; offset < row_end; offset++)
      g.constructEdge(offset, lg.get_dest(offset), 0);
  }
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

Graph* Context::getCpuGraphPointer() {
  return Context::graph_cpu;
}

float_t* Context::get_in_ptr() { return h_feats; }

void Context::norm_factor_counting() {
  norm_factor = new float_t[n];
  galois::do_all(galois::iterate((size_t)0, n),
    [&](auto v) {
      auto degree  = std::distance(graph_cpu->edge_begin(v), graph_cpu->edge_end(v));
      float_t temp = std::sqrt(float_t(degree));
      if (temp == 0.0) norm_factor[v] = 0.0;
      else norm_factor[v] = 1.0 / temp;
    }, galois::loopname("NormCounting"));
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
    labels = new label_t[m]; // single-class (one-hot) label for each vertex: N x 1
  } else {
    std::cout << "Using multi-class labels\n";
    labels = new label_t[m*num_classes]; // multi-class label for each vertex: N x E
  }
  unsigned v = 0;
  while (std::getline(in, line)) {
    std::istringstream label_stream(line);
    unsigned x;
    for (size_t idx = 0; idx < num_classes; ++idx) {
      label_stream >> x;
      if (is_single_class) {
        if (x != 0) {
          labels[v] = idx;
          break;
        }
      } else {
        labels[v*num_classes+idx] = x;
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

/*
inline void init_features(size_t dim, vec_t &x) {
    std::default_random_engine rng;
    std::uniform_real_distribution<feature_t> dist(0, 0.1);
    for (size_t i = 0; i < dim; ++i)
        x[i] = dist(rng);
}
*/
} // end deepgalois namespace
