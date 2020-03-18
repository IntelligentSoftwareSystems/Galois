#include "deepgalois/DistContext.h"

namespace deepgalois {
DistContext::DistContext() {}
DistContext::~DistContext() {}

void DistContext::saveGraph(Graph* dGraph) {
  graph_cpu = dGraph;

  localVertices = graph_cpu->size();
}

size_t DistContext::read_labels(std::string dataset_str) {
  Graph* dGraph = DistContext::graph_cpu;
  unsigned myID = galois::runtime::getSystemNetworkInterface().ID;
  galois::gPrint("[", myID, "] Reading labels from disk...\n");

  std::string filename = path + dataset_str + "-labels.txt";
  std::ifstream in;
  std::string line;
  in.open(filename, std::ios::in);
  size_t m;
  // read file header
  in >> m >> num_classes >> std::ws;
  assert(m == dGraph->globalSize());
  // size of labels should be # local nodes
  labels.resize(dGraph->size(), 0);

  uint32_t foundVertices = 0;
  unsigned v = 0;
  // each line contains a set of 0s and 1s
  while (std::getline(in, line)) {
    // only bother if local node
    if (dGraph->isLocal(v)) {
      std::istringstream label_stream(line);
      unsigned x;
      // for each class
      for (size_t idx = 0; idx < num_classes; ++idx) {
        // check if that class is labeled
        label_stream >> x;
        if (x != 0) {
          // set local id
          labels[dGraph->getLID(v)] = idx;
          foundVertices++;
          break;
        }
      }
    }
    // always increment v
    v++;
  }

  in.close();

  // print the number of vertex classes
  galois::gPrint("[", myID, "] Done with labels, unique label counts: ",
                 num_classes, "; set ", foundVertices, " nodes\n");

  return num_classes;
}

size_t DistContext::read_features(std::string dataset_str) {
  Graph* dGraph = DistContext::graph_cpu;
  unsigned myID = galois::runtime::getSystemNetworkInterface().ID;
  galois::gPrint("[", myID, "] Reading features from disk...\n");

  std::string filename = path + dataset_str + ".ft";
  std::ifstream in;
  std::string line;

  in.open(filename, std::ios::in);
  size_t m; // m = number of global vertices

  // header read
  in >> m >> feat_len >> std::ws;
  // use local size, not global size
  h_feats.resize(dGraph->size() * feat_len, 0);

  // loop through all features
  while (std::getline(in, line)) {
    std::istringstream edge_stream(line);
    unsigned u, v;
    float_t w;
    // vertex to set feature for
    edge_stream >> u;
    // only set if local
    if (dGraph->isLocal(u)) {
      // feature index
      edge_stream >> v;
      // actual feature
      edge_stream >> w;

      h_feats[dGraph->getLID(u) * feat_len + v] = w;
    }
  }
  in.close();

  galois::gPrint("[", myID, "] Done with features, feature length: ",
                 feat_len, "\n");

  return feat_len;
}

float_t* DistContext::get_in_ptr() {
  return &h_feats[0];
}

void DistContext::norm_factor_counting() {
  // TODO: this is a distributed operation

  // create for now, TODO need to actually fill it in
  norm_factor = new float_t[localVertices];
  galois::do_all(galois::iterate((size_t)0, localVertices),
    [&](auto v) {
      norm_factor[v] = 0.01;
    }, galois::loopname("NormCounting"));

  //galois::do_all(galois::iterate((size_t)0, localVertices),
  //  [&](auto v) {
  //    auto degree  = std::distance(graph_cpu->edge_begin(v), graph_cpu->edge_end(v));
  //    float_t temp = std::sqrt(float_t(degree));
  //    if (temp == 0.0) norm_factor[v] = 0.0;
  //    else norm_factor[v] = 1.0 / temp;
  //  }, galois::loopname("NormCounting"));

  return;
}

void DistContext::initializeSyncSubstrate() {
  DistContext::syncSubstrate =
    new galois::graphs::GluonSubstrate<Graph>(
      *DistContext::graph_cpu,
      galois::runtime::getSystemNetworkInterface().ID,
      galois::runtime::getSystemNetworkInterface().Num,
      false
    );
}

galois::graphs::GluonSubstrate<Graph>* DistContext::getSyncSubstrate() {
  return DistContext::syncSubstrate;
};

}  // deepgalois
