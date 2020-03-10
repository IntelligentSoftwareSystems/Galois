#include "deepgalois/DistContext.h"

namespace deepgalois {
DistContext::DistContext() {}
DistContext::~DistContext() {}

void DistContext::saveGraph(Graph* dGraph) {
  graph_cpu = dGraph;
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
  // TODO
  return 0;
}

float_t* DistContext::get_in_ptr() {
  // TODO
  return nullptr;
}

void DistContext::norm_factor_counting() {
  // TODO
}

}  // deepgalois
