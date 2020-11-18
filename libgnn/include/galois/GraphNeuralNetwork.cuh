#ifndef GALOIS_GNN_GPU_CLASS
#define GALOIS_GNN_GPU_CLASS

#include "galois/GNNTypes.h"
#include "galois/graphs/GNNGraph.cuh"

namespace galois {

//! Helper class for a GNN: holds GPU arguments. In its own class so that the
//! compiler used for it can differ from the main CPU code
class GraphNeuralNetworkGPU {
public:
  //! Gets accuracy of a prediction given pointers to the data on the GPU
  float
  GetGlobalAccuracyGPU(const galois::graphs::GNNGraphGPUAllocations& gpu_graph,
                       galois::GNNPhase phase,
                       const galois::PointerWithSize<GNNFloat> predictions);
};

} // namespace galois

#endif
