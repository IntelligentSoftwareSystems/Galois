// Execution engine
#include <iostream>
#include <sstream>
#ifdef GALOIS_USE_DIST
#include "DistributedGraphLoader.h"
#include "galois/DistGalois.h"
#include "galois/runtime/Network.h"
#endif
#include "galois/Galois.h"
#include "galois/Version.h"
#include "galois/Timer.h"
#include "deepgalois/Net.h"

static void LonestarGnnPrintVersion(llvm::raw_ostream& out) {
  out << "LoneStarGNN Benchmark Suite v" << galois::getVersion()
      << " (" << galois::getRevision() << ")\n";
  out.flush();
}

//! initialize lonestargnn benchmark
void LonestarGnnStart(int argc, char** argv, const char* app, const char* desc,
                      const char* url) {
  llvm::cl::SetVersionPrinter(LonestarGnnPrintVersion);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  galois::runtime::setStatFile(statFile);

#ifndef __GALOIS_HET_CUDA__
  numThreads = galois::setActiveThreads(numThreads); // number of threads on CPU
#endif

#ifdef GALOIS_USE_DIST
  auto& net = galois::runtime::getSystemNetworkInterface();
  if (net.ID == 0) {
#endif
  LonestarGnnPrintVersion(llvm::outs());
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

int main(int argc, char** argv) {
#ifdef GALOIS_USE_DIST
  galois::DistMemSys G;
#else
  galois::SharedMemSys G;
#endif
  LonestarGnnStart(argc, argv, name, desc, url);

  // Get a partitioned graph first
  std::vector<unsigned> dummyVec;
  deepgalois::DGraph* dGraph = NULL;
#ifdef GALOIS_USE_DIST
  dGraph = galois::graphs::constructSymmetricGraph<char, void>(dummyVec);
#endif

  // initialize network + whole context on CPU
  // read network, features, ground truth, initialize metadata
  // default setting for now; can be customized by the user
  deepgalois::Net network(dataset, numThreads, num_conv_layers, epochs, hidden1,
                          learning_rate, dropout_rate, weight_decay,
                          add_selfloop, is_single_class, add_l2norm, add_dense,
                          neighbor_sample_sz, subgraph_sample_sz, val_interval);

  // initialize distributed context
  network.partitionInit(dGraph, dataset, is_single_class);

  // construct layers from distributed context
  network.construct_layers();
  network.print_layers_info();
  deepgalois::ResourceManager rm; // tracks peak memory usage

  // the optimizer used to update parameters,
  // see optimizer.h for more details
  // optimizer *opt = new gradient_descent();
  // optimizer *opt = new adagrad();
  deepgalois::optimizer* opt = new deepgalois::adam();
  galois::StatTimer Ttrain("TrainAndVal");
  Ttrain.start();
  network.train(opt, do_validate); // do training using training samples
  Ttrain.stop();

  if (do_test) {
    // test using test samples
    galois::gPrint("\n");
    network.read_test_masks(dataset);
    galois::StatTimer Ttest("Test");
    Ttest.start();
    acc_t test_loss = 0.0, test_acc = 0.0;
    double test_time = network.evaluate("test", test_loss, test_acc);
    galois::gPrint("Testing: test_loss = ", test_loss, " test_acc = ", test_acc,
                   " test_time = ", test_time, "\n");
    Ttest.stop();
  }
  galois::gPrint("\n", rm.get_peak_memory(), "\n\n");
  return 0;
}
