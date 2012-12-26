/** Machine Descriptions on Sun -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
*/
#if defined(sun) || defined(__sun)

#include "Galois/Runtime/ll/HWTopo.h"

#include <vector>

#include <unistd.h>
#include <stdio.h>
#include <thread.h>
#include <sys/types.h>
#include <sys/processor.h>
#include <sys/procset.h>

using namespace GaloisRuntime;

namespace {

static bool sunBindToProcessor(int proc) {
  if (processor_bind(P_LWPID,  thr_self(), proc, 0) == -1) {
    perror("Error");
    return false;
    //reportWarning("Could not set CPU Affinity for thread", (unsigned)proc);
  }
  return true;
}

//Flat machine with the correct number of threads and binding
struct AutoSunPolicy {
  
  std::vector<int> procmap; //Galoid id -> solaris id

  unsigned numThreads, numCores, numPackages;

  AutoSunPolicy() {

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
AutoSunPolicy A;

} //namespace



bool GaloisRuntime::LL::bindThreadToProcessor(int id) {
  return sunBindToProcessor(A.procmap[id]);
}

unsigned GaloisRuntime::LL::getMaxThreads() {
  return A.numThreads;
}

unsigned GaloisRuntime::LL::getMaxCores() {
  return A.numCores;
}

unsigned GaloisRuntime::LL::getMaxPackages() {
  return A.numPackages;
}

unsigned GaloisRuntime::LL::getMaxPackageForThread(int id) {
  return A.numPackages - 1;
}

unsigned GaloisRuntime::LL::getPackageForThread(int id) {
  return 0;
}

bool GaloisRuntime::LL::isPackageLeader(int id) {
  return id == 0;
}

#endif //sun
