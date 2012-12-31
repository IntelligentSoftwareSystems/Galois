/** Common command line processing for benchmarks -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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

#ifdef GALOIS_USE_SVNVERSION
#include "Galois/svnversion.h"
#else
#define SVNVERSION 0
#endif
#include "Galois/Version.h"
#include "Galois/Galois.h"
#include "Galois/Runtime/ll/gio.h"
#include "llvm/Support/CommandLine.h"

#include <sstream>

//! standard global options to the benchmarks
static llvm::cl::opt<bool> skipVerify("noverify", llvm::cl::desc("Skip verification step"), llvm::cl::init(false));
static llvm::cl::opt<int> numThreads("t", llvm::cl::desc("Number of threads"), llvm::cl::init(1));

//! initialize lonestar benchmark
void LonestarStart(int argc, char** argv, const char* app, const char* desc = 0, const char* url = 0) {
  using namespace GaloisRuntime::LL;

  gPrint("Galois Benchmark Suite v" GALOIS_VERSION_STRING);
  gPrint(" (r%d)\n", SVNVERSION);
  gPrint("Copyright (C) " GALOIS_COPYRIGHT_YEAR_STRING " The University of Texas at Austin\n");
  gPrint("http://iss.ices.utexas.edu/galois/\n\n");
  gPrint("application: %s\n", app);
  gPrint("%s\n", desc ? desc : "");
  if (url) {
    gPrint("http://iss.ices.utexas.edu/?p=projects/galois/benchmarks/%s\n", url);
  }

  std::ostringstream cmdout;
  for (int i = 0; i < argc; ++i) {
    cmdout << argv[i];
    if (i != argc - 1)
      cmdout << " ";
  }
  gInfo("CommandLine %s", cmdout.str().c_str());
  char name[256];
  gethostname(name, 256);
  gInfo("Hostname %s", name);
  gFlush();

  llvm::cl::ParseCommandLineOptions(argc, argv);
  numThreads = Galois::setActiveThreads(numThreads); 

  // gInfo ("Using %d threads\n", numThreads.getValue());
  GaloisRuntime::reportStat(0, "Threads", numThreads);
}

#endif
