// Graph Neural Networks
// Xuhao Chen <cxh@utexas.edu>
#include "lonestargnn.h"
#include "DistributedGraphLoader.h"

const char* name = "Graph Isomorphism Network (GIN)";
const char* desc = "Graph isomorphism neural networks on an undirected graph";
const char* url  = 0;
static cll::opt<unsigned>learn_eps("le", cll::desc("whether to learn the parameter epsilon (default value false)"), cll::init(0));
static cll::opt<std::string>agg_type("at", cll::desc("Aggregator Type"), cll::init("sum"));

template <>
class graph_conv_layer<agg_type> {
public:
  FV apply_edge(VertexID src, VertexID dst, FV2D in_data) {
    return in_data[dst];
  }
  FV apply_vertex(VertexID src, FV2D in_data) {
    FV a = deepgalois::matmul(deepgalois::accum, deepgalois::W);
    FV b = deepgalois::scale(in_data[src], 1.0 + self.eps);
    return deepgalois::vadd(a, b);
  }
};

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarGnnStart(argc, argv, name, desc, url);
  deepgalois::Net network; // the neural network to train

  graph_conv_layer<agg_type> layer0;
  return 0;
}

