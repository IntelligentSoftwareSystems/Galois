// Execution engine
#include <iostream>
#include <sstream>
#ifdef GALOIS_ENABLE_GPU
#include "galois/Galois.h"
#else
#include "DistributedGraphLoader.h"
#include "galois/DistGalois.h"
#include "galois/runtime/Network.h"
#endif
#include "galois/Version.h"
#include "galois/Timer.h"
#include "deepgalois/Net.h"

static void LonestarGnnPrintVersion(llvm::raw_ostream& out) {
  out << "LoneStarGNN Benchmark Suite v" << galois::getVersion() << " ("
      << galois::getRevision() << ")\n";
  out.flush();
}

//! initialize lonestargnn benchmark
void LonestarGnnStart(int argc, char** argv, const char* app, const char* desc,
                      const char* url) {
  llvm::cl::SetVersionPrinter(LonestarGnnPrintVersion);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  galois::runtime::setStatFile(statFile);

  unsigned hostID = 0;
#ifndef GALOIS_ENABLE_GPU
  numThreads = galois::setActiveThreads(numThreads); // number of threads on CPU
  hostID = galois::runtime::getSystemNetworkInterface().ID;
#endif

  if (hostID == 0) {
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
  }

  char name[256];
  gethostname(name, 256);
  galois::runtime::reportParam("(NULL)", "Hostname", name);
}

int main(int argc, char** argv) {
#ifdef GALOIS_ENABLE_GPU
  galois::SharedMemSys G;
#else
  galois::DistMemSys G;
#endif
  LonestarGnnStart(argc, argv, name, desc, url);

  // Get a partitioned graph first
  std::vector<unsigned> dummyVec;
  std::unique_ptr<deepgalois::DGraph> dGraph;
#ifndef GALOIS_ENABLE_GPU
  dGraph = galois::graphs::constructSymmetricGraph<char, void>(dummyVec);
#endif
  // initialize network + whole context on CPU
  // read network, initialize metadata
  // default setting for now; can be customized by the user
  deepgalois::Net network(dataset, numThreads, num_conv_layers, epochs, hidden1,
                          learning_rate, dropout_rate, weight_decay,
                          add_selfloop, is_single_class, add_l2norm, add_dense,
                          neighbor_sample_sz, subgraph_sample_sz, val_interval);

  // initialize distributed context
  network.partitionInit(dGraph.get(), dataset, is_single_class);

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
    network.read_test_masks(dataset);
    galois::StatTimer Ttest("Test");
    Ttest.start();
    acc_t test_loss = 0.0, test_acc = 0.0;
    double test_time = network.evaluate("test", test_loss, test_acc);
#ifndef GALOIS_ENABLE_GPU
    if (galois::runtime::getSystemNetworkInterface().ID == 0) {
      galois::gPrint("test_loss = ", test_loss, " test_acc = ", test_acc,
                     " test_time = ", test_time, "\n");
    }
#else
    galois::gPrint("Testing: test_loss = ", test_loss, " test_acc = ", test_acc,
                   " test_time = ", test_time, "\n");
#endif
    Ttest.stop();
  }
  galois::gInfo(rm.get_peak_memory());
  return 0;
}
