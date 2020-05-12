#pragma once
#include <string>
#include <cassert>
#include "deepgalois/types.h"
#include "deepgalois/reader.h"
#include "deepgalois/configs.h"
#include "deepgalois/GraphTypes.h"

namespace deepgalois {

class Context {
  bool is_device;         // is this on device or host
  bool is_selfloop_added; // whether selfloop is added to the input graph
  std::string dataset;
  Reader reader;

public:
  GraphCPU* graph_cpu; // the input graph, |V| = N
  GraphCPU* getGraphPointer() { return graph_cpu; }
  Context() : Context(false) {}
  //! initializer for gpu; goes ahead and sets a few things
  Context(bool use_gpu) : is_device(use_gpu), is_selfloop_added(false) {}
  ~Context() {}
  void set_dataset(std::string dataset_str) {
    dataset = dataset_str;
    reader.init(dataset);
  }
  size_t read_masks(std::string mask_type, size_t n, 
                    size_t& begin, size_t& end, mask_t* masks) {
    return reader.read_masks(mask_type, n, begin, end, masks);
  }
  size_t read_graph(bool selfloop) {
    graph_cpu            = new GraphCPU();
    graph_cpu->readGraph(dataset, selfloop);
    is_selfloop_added = selfloop;
    std::cout << "num_vertices " << graph_cpu->size() 
              << " num_edges " << graph_cpu->sizeEdges() << "\n";
    return graph_cpu->size();
  }

  //! Checks if subgraph being used, sets currenet graph, then calls degreex
  //! counting
  GraphCPU* getFullGraph() {
    graph_cpu->degree_counting(); // TODO: why is it here? should be in read_graph
    return graph_cpu;
  }
};

} // namespace deepgalois
