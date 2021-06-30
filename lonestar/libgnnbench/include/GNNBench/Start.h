#pragma once

#include "galois/Galois.h"
#include "galois/Version.h"
#include "GNNBench/Input.h"
#ifdef GALOIS_ENABLE_GPU
#include "galois/CUDAUtilHostDecls.h"
#endif

////////////////////////////////////////////////////////////////////////////////
// CLI
////////////////////////////////////////////////////////////////////////////////

extern llvm::cl::opt<unsigned> num_threads;
extern llvm::cl::opt<unsigned> num_epochs;
extern llvm::cl::opt<unsigned> layer_size;
extern llvm::cl::opt<galois::GNNLayerType> cl_layer_type;
extern llvm::cl::opt<unsigned> train_minibatch_size;
extern llvm::cl::opt<unsigned> test_minibatch_size;
extern llvm::cl::opt<bool> do_graph_sampling;
extern llvm::cl::opt<float> learning_rate;

#ifdef GALOIS_ENABLE_GPU
std::string personality_str(DevicePersonality p);
extern llvm::cl::opt<int> num_nodes;
extern llvm::cl::opt<std::string> personality_set;

namespace internal {
void heteroSetup();
};
#endif

const char* GNNLayerToString(galois::GNNLayerType s);

////////////////////////////////////////////////////////////////////////////////
// Init functions
////////////////////////////////////////////////////////////////////////////////

//! Parses command line + setup some stats
void GNNBenchStart(int argc, char** argv, const char* app);
void GNNBenchStart(int argc, char** argv, const char* app, const char* desc,
                   const char* url);
