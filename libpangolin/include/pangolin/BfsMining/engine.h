#include "galois/Galois.h"
#include "pangolin/res_man.h"
#include "pangolin/BfsMining/embedding_list.h"
#include "llvm/Support/CommandLine.h"

llvm::cl::opt<std::string>
  statFile("statFile", llvm::cl::desc("Optional output file to print stats to"));

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarMineStart(argc, argv, name, desc, url);

  galois::runtime::setStatFile(statFile);
  AppMiner miner(k, numThreads);
  galois::StatTimer Tinitial("GraphReadingTime");
  Tinitial.start();
  miner.read_graph(filetype, filename);
  Tinitial.stop();
  ResourceManager rm;
  for (unsigned nt = 0; nt < num_trials; nt++) {
    std::cout << "\nStart running trial " << nt + 1 << ": ";
    galois::StatTimer Tinitemb("EmbInitTime");
    Tinitemb.start();
    miner.initialize(pattern_filename);
    Tinitemb.stop();

    galois::StatTimer Tcomp("Compute");
    Tcomp.start();
#ifdef TRIANGLE
    miner.tc_solver();
#else
    miner.solver();
#endif // TRIANGLE
    Tcomp.stop();
    miner.print_output();
    miner.clean();
  }
  std::cout << "\n\t" << rm.get_peak_memory() << "\n\n";
  return 0;
}
