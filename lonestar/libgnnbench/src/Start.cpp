#include "GNNBench/Start.h"

namespace cll = llvm::cl;

cll::opt<unsigned> num_threads("t", cll::desc("Number of threads (default 1)"),
                               cll::init(1));
cll::opt<unsigned> num_runs("runs", cll::desc("Number of runs (default 1)"),
                            cll::init(1));
cll::opt<std::string>
    stat_file("statFile", cll::desc("Optional output file to print stats to"));

////////////////////////////////////////////////////////////////////////////////

static void PrintVersion(llvm::raw_ostream& out) {
  out << "D-Galois Benchmark Suite v" << galois::getVersion() << " ("
      << galois::getRevision() << ")\n";
  out.flush();
}

////////////////////////////////////////////////////////////////////////////////

void GNNBenchStart(int argc, char** argv, const char* app) {
  GNNBenchStart(argc, argv, app, nullptr, nullptr);
}

void GNNBenchStart(int argc, char** argv, const char* app, const char* desc,
                   const char* url) {
  llvm::cl::SetVersionPrinter(PrintVersion);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  num_threads = galois::setActiveThreads(num_threads);
  galois::runtime::setStatFile(stat_file);

  auto& net = galois::runtime::getSystemNetworkInterface();

  if (net.ID == 0) {
    PrintVersion(llvm::outs());
    llvm::outs() << "Copyright (C) " << galois::getCopyrightYear()
                 << " The University of Texas at Austin\n";
    llvm::outs() << "http://iss.ices.utexas.edu/galois/\n\n";
    llvm::outs() << "application: " << (app ? app : "unspecified") << "\n";

    if (desc) {
      llvm::outs() << desc << "\n";
    }
    if (url) {
      llvm::outs()
          << "http://iss.ices.utexas.edu/?p=projects/galois/benchmarks/" << url
          << "\n";
    }
    llvm::outs() << "\n";
    llvm::outs().flush();

    std::ostringstream cmdout;

    for (int i = 0; i < argc; ++i) {
      cmdout << argv[i];
      if (i != argc - 1)
        cmdout << " ";
    }

    galois::runtime::reportParam("GNNBench", "CommandLine", cmdout.str());
    galois::runtime::reportParam("GNNBench", "Threads", num_threads);
    galois::runtime::reportParam("GNNBench", "Hosts", net.Num);
    galois::runtime::reportParam("GNNBench", "Runs", num_runs);
    galois::runtime::reportParam("GNNBench", "Run_UUID",
                                 galois::runtime::getRandUUID());
    galois::runtime::reportParam("GNNBench", "InputDirectory", input_directory);
    galois::runtime::reportParam("GNNBench", "Input", input_name);
    galois::runtime::reportParam("GNNBench", "PartitionScheme",
                                 GNNPartitionToString(partition_scheme));
    // XXX report the rest of the command line options
  }

  char name[256];
  gethostname(name, 256);
  galois::runtime::reportParam("GNNBench", "Hostname", name);
}
