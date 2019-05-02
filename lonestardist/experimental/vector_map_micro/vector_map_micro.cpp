#include <random>
#include <unordered_set>

#include "galois/Galois.h"
#include "galois/gstl.h"
#include "galois/runtime/Profile.h"
#include "Lonestar/BoilerPlate.h"

#include "llvm/Support/CommandLine.h"

namespace cll = llvm::cl;
static cll::opt<unsigned> metasize("metasize", cll::desc("metasize to use"), 
                               cll::init(30000));
static cll::opt<unsigned> sizet("size", cll::desc("Size to use"), 
                               cll::init(100));
static cll::opt<float> per("percent", cll::desc("Percent to access"), 
                           cll::init(1));

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, "dummy", "dummY", "dumy");

  galois::gstl::Vector<galois::gstl::Vector<unsigned>> test;
  test.resize(metasize);

  galois::do_all(galois::iterate(0u, (unsigned)metasize), [&] (unsigned j) {
    test[j].resize(sizet);
    for (unsigned i = 0; i < sizet; i++) {
      test[j][i] = i;
    }
  });

  galois::gstl::Vector<galois::gstl::Map<unsigned, unsigned>> test2;
  test2.resize(metasize);
  galois::do_all(galois::iterate(0u, (unsigned)metasize), [&] (unsigned j) {
    for (unsigned i = 0; i < sizet; i++) {
      test2[j][i] = i;
    }
  });

  //galois::gstl::Vector<galois::gstl::UnorderedMap<unsigned, unsigned>> test3;
  //test3.resize(metasize);
  //galois::do_all(galois::iterate(0u, (unsigned)metasize), [&] (unsigned j) {
  //  for (unsigned i = 0; i < sizet; i++) {
  //    test3[j][i] = i;
  //  }
  //});

  // get random sources to check
  std::minstd_rand0 r_generator;
  r_generator.seed(100);
  std::uniform_int_distribution<uint64_t> r_dist(0, sizet - 1);
  
  unsigned numberOfSources = per * sizet;
  std::unordered_set<unsigned> randomSources;
  while (randomSources.size() < numberOfSources) {
    randomSources.insert(r_dist(r_generator));
  }
  galois::gPrint("random ready\n");

  //galois::StatTimer a("t1");
  //a.start();
  // search for numbers in backwards order
  galois::GAccumulator<unsigned> found;
  found.reset();

  //galois::runtime::profilePapi([&] () {
  galois::do_all(galois::iterate(0u, (unsigned)metasize), [&] (unsigned k) {
    for (unsigned i : randomSources) {
      for (unsigned j = 0; j < sizet; j++) {
        if (test[k][j] == i) {
          found += 1;
          break;
        }
      }
    }
  }, galois::loopname("vector access"));
  //}, "vector access");

  galois::gPrint("vector done\n");

  //galois::runtime::profilePapi([&] () {
  galois::do_all(galois::iterate(0u, (unsigned)metasize), [&] (unsigned k) {
    for (unsigned i : randomSources) {
      if (test2[k][i] == i) {
        found += 1;
      }
    }
  },galois::loopname("map access"));
  //}, "map access");

  galois::gPrint("map done\n");

  //galois::runtime::profilePapi([&] () {
  //galois::do_all(galois::iterate(0u, (unsigned)metasize), [&] (unsigned k) {
  //  for (unsigned i : randomSources) {
  //    if (test3[k][i] == i) {
  //      found += 1;
  //    }
  //  }
  //});
  //}, "umap access");

  galois::gPrint("found is ", found.reduce(), "\n");
}
