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

#ifdef GALOIS_ENABLE_GPU
std::string personality_str(DevicePersonality p);
extern llvm::cl::opt<int> num_nodes;
extern llvm::cl::opt<std::string> personality_set;

namespace internal {
void heteroSetup();
};
#endif

////////////////////////////////////////////////////////////////////////////////
// Init functions
////////////////////////////////////////////////////////////////////////////////

//! Parses command line + setup some stats
void GNNBenchStart(int argc, char** argv, const char* app);
void GNNBenchStart(int argc, char** argv, const char* app, const char* desc,
                   const char* url);
