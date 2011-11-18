/** Common command line processing for benchmarks -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef LONESTAR_BOILERPLATE_H
#define LONESTAR_BOILERPLATE_H

#include "llvm/Support/CommandLine.h"
#include "Galois/Galois.h"

//! standard global options to the benchmarks
static llvm::cl::opt<bool> skipVerify("noverify", llvm::cl::desc("Skip verification step"), llvm::cl::init(false));
static llvm::cl::opt<int> numThreads("t", llvm::cl::desc("Number of threads"), llvm::cl::init(1));

//! initialize lonestar benchmark
template<typename OS>
void LonestarStart(int argc, char** argv, OS& out, const char* app, const char* desc = 0, const char* url = 0) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  out << "\nLonestar Benchmark Suite v3.0 (C++)\n"
      << "Copyright (C) 2011 The University of Texas at Austin\n"
      << "http://iss.ices.utexas.edu/lonestar/\n"
      << "\n"
      << "application: " << app << "\n";
  if (desc)
    out << desc << "\n";
  if (url) {
    out 
      << "http://iss.ices.utexas.edu/?p=projects/galois/benchmarks/" 
      << url << "\n";
  }
  Galois::setMaxThreads(numThreads);
  out << "\n";
}

#endif
