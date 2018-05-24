#include "galois/substrate/HWTopo.h"
#include "galois/gIO.h"

#include <vector>
#include <sched.h>

using namespace galois::substrate;

namespace {

static bool bindToProcessor(int proc) {
  cpu_set_t mask;
  /* CPU_ZERO initializes all the bits in the mask to zero. */
  CPU_ZERO(&mask);
  
  /* CPU_SET sets only the bit corresponding to cpu. */
  // void to cancel unused result warning
  (void)CPU_SET(proc, &mask);
  
  /* sched_setaffinity returns 0 in success */
  if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
    gWarn("Could not set CPU affinity for thread ", proc, "(", strerror(errno), ")");
    return false;
  }
  return true;
}

//! Flat machine with the correct number of threads and binding
struct Policy {
  std::vector<int> procmap; //Galois id -> cpu id

  unsigned numThreads, numCores, numPackages;

  Policy() {
    for (int i = 0; i < 16; ++i) {
      for (int j = 0; j < 4; ++j) {
        procmap.push_back(j*16 + i);
      }
    }
    numThreads = procmap.size();
    numCores = procmap.size();
    numPackages = 1;
  }
};

static Policy& getPolicy() {
  static Policy A;
  return A;
}

} //namespace

bool galois::runtime::LL::bindThreadToProcessor(int id) {
  return bindToProcessor(getPolicy().procmap[id]);
}

unsigned galois::runtime::LL::getProcessorForThread(int id) {
  assert(size_t(id) < procmap.size ());
  return getPolicy().procmap[id];
}

unsigned galois::runtime::LL::getMaxThreads() {
  return getPolicy().numThreads;
}

unsigned galois::runtime::LL::getMaxCores() {
  return getPolicy().numCores;
}

unsigned galois::runtime::LL::getMaxPackages() {
  return getPolicy().numPackages;
}

unsigned galois::runtime::LL::getMaxPackageForThread(int id) {
  return getPolicy().numPackages - 1;
}

unsigned galois::runtime::LL::getPackageForThread(int id) {
  return 0;
}

bool galois::runtime::LL::isPackageLeader(int id) {
  return id == 0;
}

unsigned galois::runtime::LL::getLeaderForThread(int id) {
  return 0;
}

unsigned galois::runtime::LL::getLeaderForPackage(int id) {
  return 0;
}
