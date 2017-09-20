/** Machine Descriptions on BlueGeneQ -*- C++ -*-
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

#include <vector>
#include <sched.h>

using namespace galois::Substrate;

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

bool galois::Runtime::LL::bindThreadToProcessor(int id) {
  return bindToProcessor(getPolicy().procmap[id]);
}

unsigned galois::Runtime::LL::getProcessorForThread(int id) {
  assert(size_t(id) < procmap.size ());
  return getPolicy().procmap[id];
}

unsigned galois::Runtime::LL::getMaxThreads() {
  return getPolicy().numThreads;
}

unsigned galois::Runtime::LL::getMaxCores() {
  return getPolicy().numCores;
}

unsigned galois::Runtime::LL::getMaxPackages() {
  return getPolicy().numPackages;
}

unsigned galois::Runtime::LL::getMaxPackageForThread(int id) {
  return getPolicy().numPackages - 1;
}

unsigned galois::Runtime::LL::getPackageForThread(int id) {
  return 0;
}

bool galois::Runtime::LL::isPackageLeader(int id) {
  return id == 0;
}

unsigned galois::Runtime::LL::getLeaderForThread(int id) {
  return 0;
}

unsigned galois::Runtime::LL::getLeaderForPackage(int id) {
  return 0;
}
