#include <string>
#include <sstream>
#include <iostream>
#ifndef GALOIS_ENABLE_GPU
#include "galois/Galois.h"
#endif
#include "galois/gIO.h"
#include "MiningBench/Start.h"

namespace cll = llvm::cl;
cll::opt<std::string> filetype("ft", cll::desc("<filetype: txt,adj,mtx,gr>"),
                               cll::init("gr"));
cll::opt<unsigned> num_trials("n",
                              cll::desc("perform n trials (default value 1)"),
                              cll::init(1));
cll::opt<unsigned>
    nblocks("b", cll::desc("edge blocking to b blocks (default value 1)"),
            cll::init(1));
cll::opt<std::string>
    pattern_filename("p",
                     cll::desc("<pattern graph filename: symmetrized graph>"),
                     cll::init(""));
cll::opt<std::string>
    morder_filename("mo", cll::desc("<filename: pre-defined matching order>"),
                    cll::init(""));
cll::opt<unsigned> fv("fv", cll::desc("first vertex is special"), cll::init(0));
cll::opt<unsigned>
    k("k", cll::desc("max number of vertices in k-clique (default value 3)"),
      cll::init(3));
cll::opt<unsigned> show("s", cll::desc("print out the details"), cll::init(0));
cll::opt<unsigned>
    debug("d", cll::desc("print out the frequent patterns for debugging"),
          cll::init(0));
cll::opt<unsigned> minsup("ms",
                          cll::desc("minimum support (default value 300)"),
                          cll::init(300));
cll::opt<std::string>
    preset_filename("pf", cll::desc("<filename: preset matching order>"),
                    cll::init(""));
// TODO use skipVerify from liblonestar
cll::opt<bool>
    verify("v", llvm::cl::desc("do verification step (default value false)"),
           llvm::cl::init(false));

cll::opt<bool>
    simpleGraph("simpleGraph",
                cll::desc("Specify that the input graph is "
                          "simple (has no multiple edges or self-loops)"),
                cll::init(false));

#ifdef GALOIS_ENABLE_GPU
// TODO is numThreads necessary for gpu apps? remove it if not.
cll::opt<int> numThreads("t",
                         llvm::cl::desc("Number of threads (default value 1)"),
                         llvm::cl::init(1));
cll::opt<bool>
    symmetricGraph("symmetricGraph",
                   cll::desc("Specify that the input graph is symmetric"),
                   cll::init(false));
#endif

// TODO merge LonestarStart for cpu apps
void LonestarMineStart(int argc, char** argv, const char* app, const char* desc,
                       const char* url) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  if (!simpleGraph || !symmetricGraph) {
    GALOIS_DIE("This application requires a symmetric simple graph input "
               " which is symmetric and has no multiple edges or self-loops;"
               " please use both -symmetricGraph and -simpleGraph flag "
               " to indicate the input is a symmetric simple graph");
  }

#ifndef GALOIS_ENABLE_GPU
  numThreads = galois::setActiveThreads(numThreads);
  galois::runtime::setStatFile(statFile);
#endif
  std::cout << "Copyright (C) 2020 The University of Texas at Austin\n";
  std::cout << "http://iss.ices.utexas.edu/galois/\n\n";
  std::cout << "application: " << (app ? app : "unspecified") << "\n";
  if (desc)
    std::cout << desc << "\n";
  if (url)
    std::cout << "http://iss.ices.utexas.edu/?p=projects/galois/benchmarks/"
              << url << "\n";
  std::cout << "\n";
  std::ostringstream cmdout;
  for (int i = 0; i < argc; ++i) {
    cmdout << argv[i];
    if (i != argc - 1)
      cmdout << " ";
  }
#ifndef GALOIS_ENABLE_GPU
  galois::runtime::reportParam("(NULL)", "CommandLine", cmdout.str());
  galois::runtime::reportParam("(NULL)", "Threads", numThreads);
  galois::runtime::reportParam("(NULL)", "Runs", num_trials);
  galois::runtime::reportParam("(NULL)", "Input", inputFile);
  galois::runtime::reportParam("(NULL)", "Hosts", 1);
#endif
}
