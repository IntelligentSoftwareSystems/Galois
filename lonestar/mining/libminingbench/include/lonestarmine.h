#pragma once
#include "llvm/Support/CommandLine.h"

namespace cll = llvm::cl;
extern cll::opt<std::string> filename;
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
extern cll::opt<int> numThreads;
extern cll::opt<std::string> preset_filename;
extern cll::opt<bool> verify;

extern cll::opt<bool> simpleGraph;

extern cll::opt<bool> symmetricGraph;

#ifndef GALOIS_ENABLE_GPU
extern cll::opt<std::string> statFile;
#endif

void LonestarMineStart(int argc, char** argv, const char* app, const char* desc,
                       const char* url);
