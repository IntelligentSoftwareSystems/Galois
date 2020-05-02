#include "deepgalois/DistContext.h"
#include "deepgalois/utils.h"
#include "deepgalois/configs.h"

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
  h_labels = new label_t[dGraph->size()]; // single-class (one-hot) label for each vertex: N x 1

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
          h_labels[dGraph->getLID(v)] = idx;
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
  h_feats = new float_t[dGraph->size() * feat_len];

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

size_t DistContext::read_masks(std::string dataset_str, std::string mask_type,
                               size_t n, size_t& begin, size_t& end,
                               mask_t* masks, Graph* dGraph) {
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
        // only bother if it's local
        if (dGraph->isLocal(i)) {
          masks[dGraph->getLID(i)] = 1;
          sample_count++;
        }
      }
    }
    i++;
  }
  std::cout << mask_type + "_mask range: [" << begin << ", " << end
    << ") Number of valid samples: " << sample_count << "("
    << (float)sample_count/(float)n*(float)100 << "\%)\n";
  in.close();
  return sample_count;
}

float_t* DistContext::get_in_ptr() {
  return &h_feats[0];
}

void DistContext::norm_factor_computing(size_t g_size) {
  // TODO: this is a distributed operation

  // create for now, TODO need to actually fill it in
  norm_factor = new float_t[localVertices];
  galois::do_all(galois::iterate((size_t)0, localVertices),
    [&](auto v) {
      norm_factor[v] = 1;
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
