#include "galois/substrate/HWTopo.h"
#include "galois/gIO.h"
#include <sys/types.h>
#include <sys/sysctl.h>

using namespace galois::substrate;

namespace {

struct Policy {
  //number of "real" processors
  uint32_t numCpus;

  Policy() {
    size_t bufSize = sizeof(numCpus);
    if (sysctlbyname("hw.activecpu", &numCpus, &bufSize, NULL, 0) == -1) {
      GALOIS_SYS_DIE("Error querying number of cpus");
    } else if (bufSize != sizeof(numCpus)) {
      GALOIS_SYS_DIE("Error querying number of cpus");
    }
  }
};

static Policy& getPolicy() {
  static Policy A;
  return A;
}

} //namespace

bool galois::runtime::LL::bindThreadToProcessor(int id) {
  return false;
}

unsigned galois::runtime::LL::getProcessorForThread(int id) {
  return id;
}

unsigned galois::runtime::LL::getMaxThreads() {
  return getPolicy().numCpus;
}

unsigned galois::runtime::LL::getMaxCores() {
  return getPolicy().numCpus;
}

unsigned galois::runtime::LL::getMaxPackages() {
  return getPolicy().numCpus;
}

unsigned galois::runtime::LL::getPackageForThread(int id) {
  return id;
}

unsigned galois::runtime::LL::getMaxPackageForThread(int id) {
  return id;
}

bool galois::runtime::LL::isPackageLeader(int id) {
  return true;
}

unsigned galois::runtime::LL::getLeaderForThread(int id) {
  return id;
}

unsigned galois::runtime::LL::getLeaderForPackage(int id) {
  return id;
}
