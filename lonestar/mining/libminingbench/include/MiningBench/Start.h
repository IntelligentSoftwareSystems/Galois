#pragma once
#include "llvm/Support/CommandLine.h"

namespace cll = llvm::cl;
extern cll::opt<std::string> inputFile;
extern cll::opt<std::string> filetype;
extern cll::opt<unsigned> num_trials;
extern cll::opt<unsigned> nblocks;
extern cll::opt<std::string> pattern_filename;
extern cll::opt<std::string> morder_filename;
extern cll::opt<unsigned> fv;
extern cll::opt<unsigned> k;
extern cll::opt<unsigned> show;
extern cll::opt<unsigned> debug;
extern cll::opt<unsigned> minsup;
extern cll::opt<std::string> preset_filename;

extern cll::opt<bool> simpleGraph;

// note these may come from uplevel liblonestar or libdistbench
extern cll::opt<int> numThreads; // locally defined for gpu apps (necessary?)
extern cll::opt<bool> verify;    // TODO use skipVerify from liblonestar
#ifndef GALOIS_ENABLE_GPU
extern cll::opt<std::string> statFile;
#endif
extern cll::opt<bool> symmetricGraph; // locally defined for gpu apps

void LonestarMineStart(int argc, char** argv, const char* app, const char* desc,
                       const char* url);
