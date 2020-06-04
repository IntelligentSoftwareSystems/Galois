#include "galois/Galois.h"
#include "pangolin/res_man.h"
#include "pangolin/BfsMining/embedding_list.h"

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarMineStart(argc, argv, name, desc, url);

  galois::StatTimer totalTime("TimerTotal");
  totalTime.start();

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

    galois::StatTimer execTime("Timer_0");
    execTime.start();
#ifdef TRIANGLE
    miner.tc_solver();
#else
    miner.solver();
#endif // TRIANGLE
    execTime.stop();
    miner.print_output();
    miner.clean();
  }
  std::cout << "\n\t" << rm.get_peak_memory() << "\n\n";

  totalTime.stop();

  return 0;
}
