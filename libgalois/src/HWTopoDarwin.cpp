/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#include "galois/substrate/EnvCheck.h"
#include "galois/substrate/HWTopo.h"
#include "galois/substrate/SimpleLock.h"
#include "galois/gIO.h"

#include <mach/mach_interface.h>
#include <mach/thread_policy.h>
#include <sys/types.h>
#include <sys/sysctl.h>
#include <algorithm>
#include <memory>
#include <mutex>
#include <pthread.h>

using namespace galois::substrate;

namespace {

int getIntValue(const char* name) {
  int value;
  size_t len = sizeof(value);

  if (sysctlbyname(name, &value, &len, nullptr, 0) == -1) {
    GALOIS_SYS_DIE("could not get sysctl value for ", name, ": ",
                   strerror(errno));
  }

  return value;
}

HWTopoInfo makeHWTopo() {
  MachineTopoInfo mti;
  mti.maxSockets   = getIntValue("hw.packages");
  mti.maxThreads   = getIntValue("hw.logicalcpu_max");
  mti.maxCores     = getIntValue("hw.physicalcpu_max");
  mti.maxNumaNodes = mti.maxSockets;

  std::vector<ThreadTopoInfo> tti;
  tti.reserve(mti.maxThreads);

  // Darwin doesn't expose more fine-grained topology information,
  // so assume a dense configuration:
  // thread 0 +
  //          |- core 0 +
  // thread 1 +         |
  //                    |- socket 0
  // thread 2 +         |
  //          |- core 1 +
  // thread 3 +

  const unsigned threadsPerSocket =
      (mti.maxThreads + mti.maxThreads - 1) / mti.maxSockets;

  // Describe dense configuration first; then, sort logical threads to the
  // back.
  for (unsigned i = 0; i < mti.maxThreads; ++i) {
    unsigned socket = i / threadsPerSocket;
    unsigned leader = socket * threadsPerSocket;
    tti.push_back(ThreadTopoInfo{
        .socketLeader = leader,
        .socket       = socket,
        .numaNode     = socket,
        .osContext    = i,
        .osNumaNode   = socket,
    });
  }

  const unsigned logicalPerPhysical =
      (mti.maxThreads + mti.maxThreads - 1) / mti.maxCores;

  std::sort(tti.begin(), tti.end(),
            [&](const ThreadTopoInfo& a, const ThreadTopoInfo& b) {
              int smtA = a.osContext % logicalPerPhysical;
              int smtB = b.osContext % logicalPerPhysical;
              if (smtA == smtB) {
                return a.osContext < b.osContext;
              }
              return smtA < smtB;
            });

  for (unsigned i = 0, m = 0; i < mti.maxThreads; ++i) {
    m                          = std::max(m, tti[i].socket);
    tti[i].tid                 = i;
    tti[i].cumulativeMaxSocket = m;
  }

  return {
      .machineTopoInfo = mti,
      .threadTopoInfo  = tti,
  };
}

} // namespace

//! binds current thread to OS HW context "proc"
bool galois::substrate::bindThreadSelf(unsigned osContext) {
  pthread_t thread              = pthread_self();
  thread_affinity_policy policy = {int(osContext)};
  thread_t machThread           = pthread_mach_thread_np(thread);
  if (thread_policy_set(machThread, THREAD_AFFINITY_POLICY,
                        thread_policy_t(&policy),
                        THREAD_AFFINITY_POLICY_COUNT)) {
    galois::gWarn("Could not set CPU affinity to ", osContext, " (",
                  strerror(errno), ")");
    return false;
  }

  return true;
}

HWTopoInfo galois::substrate::getHWTopo() {
  static SimpleLock lock;
  static std::unique_ptr<HWTopoInfo> data;

  std::lock_guard<SimpleLock> guard(lock);
  if (!data) {
    data = std::make_unique<HWTopoInfo>(makeHWTopo());
  }
  return *data;
}
