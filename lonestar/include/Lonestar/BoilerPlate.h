#ifndef LONESTAR_BOILERPLATE_H
#define LONESTAR_BOILERPLATE_H

#include "galois/Galois.h"
#include "galois/Version.h"
#include "llvm/Support/CommandLine.h"

//! standard global options to the benchmarks
extern llvm::cl::opt<bool> skipVerify;
extern llvm::cl::opt<int> numThreads;
extern llvm::cl::opt<std::string> statFile;

//! initialize lonestar benchmark
void LonestarStart(int argc, char** argv, const char* app, 
                   const char* desc = nullptr, const char* url = nullptr);
#endif
