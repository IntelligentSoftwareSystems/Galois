/** Machine Descriptions on Sun -*- C++ -*-
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
*/
#include "Galois/Runtime/ll/HWTopo.h"

#include <vector>

#include <unistd.h>
#include <stdio.h>
#include <thread.h>
#include <sys/types.h>
#include <sys/processor.h>
#include <sys/procset.h>

using namespace galois::Runtime::LL;

namespace {

static bool sunBindToProcessor(int proc) {
  if (processor_bind(P_LWPID,  thr_self(), proc, 0) == -1) {
    gWarn("Could not set CPU affinity for thread ", proc, "(", strerror(errno), ")");
    return false;
  }
  return true;
}

//Flat machine with the correct number of threads and binding
struct Policy {
  std::vector<int> procmap; //Galois id -> solaris id

  unsigned numThreads, numCores, numPackages;

  Policy() {
    processorid_t i, cpuid_max;
    cpuid_max = sysconf(_SC_CPUID_MAX);
    for (i = 0; i <= cpuid_max; i++) {
      if (p_online(i, P_STATUS) != -1) {
	procmap.push_back(i);
	//printf("processor %d present\n", i);
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
  assert(size_t(id) < procmap.size ());
  return sunBindToProcessor(getPolicy().procmap[id]);
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
