/** Machine Descriptions on Darwin -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2014, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS SOFTWARE
 * AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY
 * PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY
 * WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF TRADE.
 * NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO THE USE OF THE
 * SOFTWARE OR DOCUMENTATION. Under no circumstances shall University be liable
 * for incidental, special, indirect, direct or consequential damages or loss of
 * profits, interruption of business, or related expenses which may arise from use
 * of Software or Documentation, including but not limited to those resulting from
 * defects in Software and/or Documentation, or loss or inaccuracy of data of any
 * kind.
 *
 * @section Description
 *
 * See HWTopoLinux.cpp.
 * 
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#include "Galois/Runtime/ll/HWTopo.h"
#include "Galois/Runtime/ll/gio.h"
#include <sys/types.h>
#include <sys/sysctl.h>

using namespace Galois::Runtime::LL;

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

bool Galois::Runtime::LL::bindThreadToProcessor(int id) {
  return false;
}

unsigned Galois::Runtime::LL::getProcessorForThread(int id) {
  return id;
}

unsigned Galois::Runtime::LL::getMaxThreads() {
  return getPolicy().numCpus;
}

unsigned Galois::Runtime::LL::getMaxCores() {
  return getPolicy().numCpus;
}

unsigned Galois::Runtime::LL::getMaxPackages() {
  return getPolicy().numCpus;
}

unsigned Galois::Runtime::LL::getPackageForThread(int id) {
  return id;
}

unsigned Galois::Runtime::LL::getMaxPackageForThread(int id) {
  return id;
}

bool Galois::Runtime::LL::isPackageLeader(int id) {
  return true;
}

unsigned Galois::Runtime::LL::getLeaderForThread(int id) {
  return id;
}

unsigned Galois::Runtime::LL::getLeaderForPackage(int id) {
  return id;
}
