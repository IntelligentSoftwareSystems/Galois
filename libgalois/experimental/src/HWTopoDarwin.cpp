/** Machine Descriptions on Darwin -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * See HWTopoLinux.cpp.
 * 
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#include "Galois/Substrate/HWTopo.h"
#include "Galois/gIO.h"
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
