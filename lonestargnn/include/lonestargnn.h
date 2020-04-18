#pragma once

#include <sstream>
#include <iostream>
#include "galois/Timer.h"
#include "galois/Galois.h"
#include "galois/Version.h"
#include "galois/Reduction.h"
#include "galois/ParallelSTL.h"
#include "galois/runtime/Profile.h"
#include "llvm/Support/CommandLine.h"
#include <boost/iterator/transform_iterator.hpp>
#ifdef GALOIS_USE_DIST
#include "galois/DistGalois.h"
#include "galois/runtime/Network.h"
#endif

namespace cll = llvm::cl;
static cll::opt<std::string>
    dataset(cll::Positional, cll::desc("<dataset name>"),
            cll::Required); // 'cora', 'citeseer', 'pubmed'
static cll::opt<std::string>
    filetype(cll::Positional, cll::desc("<filetype: el,gr>"),
             cll::init("gr")); // file format of the input graph
static cll::opt<std::string>
    model("m", cll::desc("Model string"),
          cll::init("gcn")); // 'gcn', 'gcn_cheby', 'dense'
static cll::opt<float>
    learning_rate("lr", cll::desc("Initial learning rate (default value 0.01)"),
                  cll::init(0.01));
static cll::opt<unsigned>
    epochs("k", cll::desc("number of epoch, i.e. iterations (default value 1)"),
           cll::init(1));
static cll::opt<unsigned>
    hidden1("h",
            cll::desc("Number of units in hidden layer 1 (default value 16)"),
            cll::init(16));
static cll::opt<float> dropout_rate(
    "d", cll::desc("Dropout rate (1 - keep probability) (default value 0.5)"),
    cll::init(0.5));
static cll::opt<float> weight_decay(
    "wd",
    cll::desc("Weight for L2 loss on embedding matrix (default value 5e-4)"),
    cll::init(5e-4));
static cll::opt<float> early_stopping(
    "es",
    cll::desc("Tolerance for early stopping (# of epochs) (default value 10)"),
    cll::init(10));
static cll::opt<unsigned> max_degree(
    "md", cll::desc("Maximum Chebyshev polynomial degree (default value 3)"),
    cll::init(3));
static cll::opt<unsigned> do_validate("dv", cll::desc("enable validation"),
                                      cll::init(1));
static cll::opt<unsigned> do_test("dt", cll::desc("enable test"), cll::init(1));
static cll::opt<bool> add_selfloop("sl", cll::desc("add selfloop"), cll::init(0));
static cll::opt<bool> is_single_class("sc", 
    cll::desc("single-class or multi-class label (default single)"), cll::init(1));

//! standard global options to the benchmarks
extern llvm::cl::opt<bool> skipVerify;
extern llvm::cl::opt<int> numThreads;
extern llvm::cl::opt<std::string> statFile;

//! standard global options to the benchmarks
llvm::cl::opt<bool>
    skipVerify("noverify",
               llvm::cl::desc("Skip verification step (default value false)"),
               llvm::cl::init(false));
llvm::cl::opt<int>
    numThreads("t", llvm::cl::desc("Number of threads (default value 1)"),
               llvm::cl::init(1));
llvm::cl::opt<std::string> statFile(
    "statFile",
    llvm::cl::desc("ouput file to print stats to (default value empty)"),
    llvm::cl::init(""));

static void LonestarGnnPrintVersion() {
  std::cout << "LoneStarGNN Benchmark Suite v" << galois::getVersion() << " ("
            << galois::getRevision() << ")\n";
}

//! initialize lonestargnn benchmark
void LonestarGnnStart(int argc, char** argv, const char* app, const char* desc,
                      const char* url) {
  llvm::cl::SetVersionPrinter(LonestarGnnPrintVersion);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  numThreads = galois::setActiveThreads(numThreads);
  galois::runtime::setStatFile(statFile);

#ifdef GALOIS_USE_DIST
  auto& net = galois::runtime::getSystemNetworkInterface();
  if (net.ID == 0) {
#endif
  LonestarGnnPrintVersion();
  std::cout << "Copyright (C) " << galois::getCopyrightYear()
            << " The University of Texas at Austin\n";
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
  galois::runtime::reportParam("(NULL)", "CommandLine", cmdout.str());
  galois::runtime::reportParam("(NULL)", "Threads", numThreads);
#ifdef GALOIS_USE_DIST
  }
#endif

  char name[256];
  gethostname(name, 256);
  galois::runtime::reportParam("(NULL)", "Hostname", name);
}

#include "deepgalois/types.h"
#include "deepgalois/utils.h"
#include "deepgalois/net.h"
