#include "Galois/Galois.h"
#include "Galois/Accumulator.h"
#include "Galois/Statistic.h"
#include "Lonestar/BoilerPlate.h"
#include "llvm/Support/CommandLine.h"

#include <boost/iterator/counting_iterator.hpp>

#include <cmath>
#include <iostream>
#include <vector>

typedef Galois::GAccumulator<double> AccumDouble;

namespace cll = llvm::cl;

static cll::opt<int> size("size", cll::desc ("length of vectors"), cll::init(1000 * 1000));
static cll::opt<int> rounds("rounds", cll::desc ("number of rounds"), cll::init(100));
static cll::opt<int> trials("trials", cll::desc ("number of trials"), cll::init(3));

double runDoAllBurn(const std::vector<double>& vecA, const std::vector<double>& vecB) {
  Galois::Runtime::getSystemThreadPool().burnPower(Galois::getActiveThreads());
  AccumDouble result;

  for (int r = 0; r < rounds; ++r) {
    Galois::do_all(boost::counting_iterator<int>(0), boost::counting_iterator<int>(vecA.size()),
        [&](int i) { result += vecA[i] * vecB[i]; });
  }

  Galois::Runtime::getSystemThreadPool().beKind();

  return result.reduce();
}

double runDoAll(const std::vector<double>& vecA, const std::vector<double>& vecB) {
  AccumDouble result;

  for (int r = 0; r < rounds; ++r) {
    Galois::do_all(boost::counting_iterator<int>(0), boost::counting_iterator<int>(vecA.size()),
        [&](int i) { result += vecA[i] * vecB[i]; });
  }

  return result.reduce();
}

double runExplicitThread(const std::vector<double>& vecA, const std::vector<double>& vecB) {
  Galois::Runtime::Barrier& barrier = Galois::Runtime::getSystemBarrier();
  AccumDouble result;
  
  Galois::on_each([&](unsigned tid, unsigned total) {
    for (int r = 0; r < rounds; ++r) {
      auto range = Galois::block_range(
        boost::counting_iterator<int>(0), boost::counting_iterator<int>(vecA.size()), 
        tid, total);
      double sum = 0;
      for (auto ii = range.first, ei = range.second; ii != ei; ++ii) {
        sum += vecA[*ii] * vecB[*ii];
      }
      result += sum;
      barrier();
    }
  });

  return result.reduce();
}

void run(
    const std::vector<double>& vecA,
    const std::vector<double>& vecB,
    std::function<double(const std::vector<double>&, const std::vector<double>&)> fn,
    std::string name) {
  Galois::Timer t;
  t.start();
  double r = fn(vecA, vecB);
  t.stop();
  std::cout << name << " result: " << r << " time: " << t.get() << "\n";
}

int main(int argc, char* argv[]) {
  LonestarStart(argc, argv, 0, 0, 0);
  Galois::setActiveThreads(std::max(Galois::getActiveThreads(), 2U));

  std::cout
    << "threads: " << Galois::getActiveThreads() 
    << " rounds: " << rounds
    << " size: " << size << "\n";

  std::vector<double> vecA(size);
  std::vector<double> vecB(size);

  Galois::do_all(boost::counting_iterator<int>(0), boost::counting_iterator<int>(size),
      [&vecA, &vecB](int i) {
        vecA[i] = acos(-1.0);
        vecB[i] = asin(-1.0);
      }, Galois::loopname("init_loop"));

  for (int t = 0; t < trials; ++t) {
    run(vecA, vecB, runDoAll, "DoAll");
    run(vecA, vecB, runDoAllBurn, "DoAllBurn");
    run(vecA, vecB, runExplicitThread, "ExplicitThread");
  }
  return 0;
}
