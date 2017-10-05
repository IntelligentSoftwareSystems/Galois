#include "galois/Galois.h"
#include "galois/Reduction.h"
#include "galois/Timer.h"
#include "Lonestar/BoilerPlate.h"
#include "llvm/Support/CommandLine.h"

#include <boost/iterator/counting_iterator.hpp>

#include <cmath>
#include <iostream>
#include <vector>

typedef galois::GAccumulator<double> AccumDouble;

namespace cll = llvm::cl;

static cll::opt<int> size("size", cll::desc ("length of vectors"), cll::init(1000));
static cll::opt<int> rounds("rounds", cll::desc ("number of rounds"), cll::init(10000));
static cll::opt<int> trials("trials", cll::desc ("number of trials"), cll::init(1));

void runDoAllBurn(int num) {
  galois::substrate::getThreadPool().burnPower(galois::getActiveThreads());

  for (int r = 0; r < rounds; ++r) {
    galois::do_all(galois::iterate(0, num),
                   [&](int i) { asm volatile("" ::: "memory"); });
  }

  galois::substrate::getThreadPool().beKind();
}

void runDoAll(int num) {
  for (int r = 0; r < rounds; ++r) {
    galois::do_all(galois::iterate(0, num),
        [&](int i) { asm volatile("" ::: "memory"); });
  }
}

void runExplicitThread(int num) {
  galois::substrate::Barrier& barrier = galois::runtime::getBarrier(galois::runtime::activeThreads);
  
  galois::on_each([&](unsigned tid, unsigned total) {
      auto range = galois::block_range(
                                       boost::counting_iterator<int>(0), boost::counting_iterator<int>(num), 
                                       tid, total);
      for (int r = 0; r < rounds; ++r) {
        for (auto ii = range.first, ei = range.second; ii != ei; ++ii) {
          asm volatile("" ::: "memory");
        }
        barrier();
      }
  });
}

void run(std::function<void(int)> fn, std::string name) {
  galois::Timer t;
  t.start();
  fn(size);
  t.stop();
  std::cout << name << " time: " << t.get() << "\n";
}

std::atomic<int> EXIT;
#include <chrono>

int main(int argc, char* argv[]) {
  LonestarStart(argc, argv, 0, 0, 0);
  galois::setActiveThreads(std::max(galois::getActiveThreads(), 2U));

  EXIT = 0;
  std::function<void(void)> f = [] () { while (!EXIT) { std::cerr << "."; std::this_thread::sleep_for(std::chrono::milliseconds(100)); } };
  galois::substrate::getThreadPool().runDedicated(f);

  std::cout
    << "threads: " << galois::getActiveThreads() 
    << " rounds: " << rounds
    << " size: " << size << "\n";

  for (int t = 0; t < trials; ++t) {
    run(runDoAll, "DoAll");
    run(runDoAllBurn, "DoAllBurn");
    run(runExplicitThread, "ExplicitThread");
  }
  EXIT=1;
  return 0;
}
