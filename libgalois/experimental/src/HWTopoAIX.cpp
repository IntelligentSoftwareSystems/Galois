/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
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

#include "galois/substrate/HWTopo.h"
#include "galois/substrate/EnvCheck.h"
#include "galois/gIO.h"

#include <vector>
#include <cerrno>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <boost/iterator/counting_iterator.hpp>
#include <string.h>
#include <sys/rset.h>
#include <sys/processor.h>
#include <sys/thread.h>

using namespace galois::substrate;

namespace {

static bool bindToProcessor(int proc) {
  tid_t tid = thread_self();
  if (bindprocessor(BINDTHREAD, tid, proc)) {
    gWarn("Could not set CPU affinity for thread ", proc, "(", strerror(errno), ")");
    return false;
  }
  return true;
}

struct Policy {
  //! number of hardware threads
  int numThreads;
  //! number of "real" processors
  int numCores;
  //! number of sockets
  int numSockets;

  std::vector<int> sockets;
  std::vector<int> maxSocket;
  std::vector<int> virtmap;
  std::vector<int> leaders;

  /**
   * Raw data from rs_info is a set of mappings for various levels of
   * of processor tree. For example, for two dual-core processors with
   * 2-way SMT (note that processor 3 is not mapped):
   * <pre>
   * levels[0] = [ 0, 1, 2, 4, 5, 6, 7 ]  "AIX logical processor"
   * levels[2] = [ 0, 0, 1, 2, 2, 3, 3 ]  "Core"
   * levels[4] = [ 0, 0, 0, 1, 1, 1, 1 ]  "Socket"
   *
   * levels[1] = [ 0, 1, 0, 0, 1, 0, 1 ]  "Core siblings" (i.e., SMT)
   * levels[3] = [ 0, 1, 2, 0, 1, 2, 3 ]  "Socket siblings"
   * </pre>
   *
   * The sibling levels are just an alternate indexing to simplify mapping operations.
   */
  typedef std::vector<std::vector<int>> Levels;
  Levels levels;

  struct SortedByThread {
    Levels& l;
    SortedByThread(Levels& l): l(l) { }

    bool operator()(int a, int b) {
      // Only use SMT threads after all the cores have been used
      if (l[1][a] != l[1][b])
        return l[1][a] < l[1][b];
      if (l[2][a] != l[2][b])
        return l[2][a] < l[2][b];
      return l[0][a] < l[0][b];
    }
  };

  void parse() {
    int maxProcs = rs_getinfo(NULL, R_MAXPROCS, 0);
    for (int i = 0; i < 5; ++i)
      levels.emplace_back(maxProcs, -1);   

    // SMP level
    int sdl;
    if ((sdl = rs_getinfo(NULL, R_SMPSDL, 0)) == 0)
      GALOIS_DIE("rs_getinfo failed");
    numThreads = populate(sdl, levels[0], false);
    for (int i = 0; i < maxProcs; ++i) {
      if (levels[0][i] != -1)
        levels[0][i] = i;
    }

    // NUMA level
    if ((sdl = rs_getinfo(NULL, R_MCMSDL, 0)) == 0)
      return;
    populate(sdl, levels[4], false);
    populate(sdl, levels[3], true);
    numSockets = *std::max_element(levels[4].begin(), levels[4].end()) + 1;

    // Search for SDL for core because it doesn't have an explicit name
    int maxSdl = rs_getinfo(NULL, R_MAXSDL, 0);
    for (sdl = 0; sdl <= maxSdl; ++sdl) {
      if (sdl == rs_getinfo(NULL, R_SMPSDL, 0))
        continue;
      if (sdl == rs_getinfo(NULL, R_MCMSDL, 0))
        continue;
      populate(sdl, levels[2], false);
      int v = *std::max_element(levels[2].begin(), levels[2].end()) + 1;
      if (numSockets < v && v < numThreads) {
        numCores = v;
        populate(sdl, levels[1], true);
        break;
      }
    }
  }

  //! Fill in level vector, returns the number of entries set 
  int populate(int sdl, std::vector<int>& l, bool siblingsHaveDifferentNames) {
    // Get resources for current logical partition
    rsethandle_t rset = rs_alloc(RS_PARTITION);
    rsethandle_t rad = rs_alloc(RS_EMPTY);

    int sumNumProcs = 0;
    int numRads = rs_numrads(rset, sdl, 0);
    for (int rindex = 0; rindex < numRads; ++rindex) {
      if (rs_getrad(rset, rad, sdl, rindex, 0))
        GALOIS_SYS_DIE("rs_getrad() failed");
      sumNumProcs += rs_getinfo(rad, R_NUMPROCS, 0);
      int maxCpus = rs_getinfo(rad, R_MAXPROCS, 0);
      int id = 0;
      for (int i = 0; i < maxCpus; ++i) {
        if (rs_op(RS_TESTRESOURCE, rad, NULL, R_PROCS, i)) {
          if (siblingsHaveDifferentNames)
            l[i] = id++;
          else
	    l[i] = rindex;
        }
      }
    }
    rs_free(rad);
    rs_free(rset);

    return sumNumProcs;
  }

  void generateUniformLevel(int l) {
    int id = 0;
    for (unsigned i = 0; i < levels[0].size(); ++i) {
      if (levels[0][i] >= 0) {
        levels[l][i] = id++;
        levels[l+1][i] = 0;
      }
    }
  }
 
  void generate() {
    if (numCores == -1) {
      // Assume each thread is a core
      generateUniformLevel(1);
      numCores = numThreads;
    }
    if (numSockets == -1) {
      // Assume one socket
      generateUniformLevel(3);
      numSockets = 1;
    }

    // Remove unmapped processors
    for (unsigned i = 0; i < levels.size(); ++i) {
      auto isMapped = [](int x) { return x >= 0; };
      auto it = std::stable_partition(levels[i].begin(), levels[i].end(), isMapped);
      levels[i].resize(std::distance(levels[i].begin(), it));
    }
    
    // Generate mappings
    std::vector<int> lthreads(numThreads);
    std::copy(boost::counting_iterator<int>(0), boost::counting_iterator<int>(numThreads), lthreads.begin());
    std::sort(lthreads.begin(), lthreads.end(), SortedByThread(levels));

    virtmap.resize(numThreads);
    leaders.resize(numSockets);
    sockets.resize(numThreads);
    for (int gid = 0; gid < numThreads; ++gid) {
      int lid = lthreads[gid];
      virtmap[gid] = levels[0][lid];
      sockets[gid] = levels[4][lid];
      if (levels[3][lid] == 0) 
        leaders[sockets[gid]] = gid;
    }
    maxSocket.resize(numThreads);
    std::partial_sum(sockets.begin(), sockets.end(), maxSocket.begin(), [](int a, int b) { return std::max(a, b); });
  }

  Policy(): numThreads(-1), numCores(-1), numSockets(-1) {
    parse();
    generate();
    if (EnvCheck("GALOIS_DEBUG_TOPO"))
      printConfiguration();
    levels.clear();
  }

  void printConfiguration() {
    for (unsigned level = 0; level < levels.size(); ++level) {
      gPrint("levels[", level, "] = [ ");
      for (int i = 0; i < numThreads; ++i) {
        gPrint(levels[level][i], " ");
      }
      gPrint("]\n");
    }

    gPrint("Threads: ", numThreads, "\n");
    gPrint("Cores: ", numCores, "\n");
    gPrint("Sockets: ", numSockets, "\n");

    for (int i = 0; i < (int) virtmap.size(); ++i) {
      gPrint(
          "T ", i, 
          " P ", sockets[i],
          " Tr ", virtmap[i], 
          " L? ", (i == leaders[sockets[i]] ? 1 : 0));
      if (i >= numCores)
        gPrint(" HT");
      gPrint("\n");
    }
  }
};

static Policy& getPolicy() {
  static Policy A;
  return A;
}

} //namespace

bool galois::runtime::LL::bindThreadToProcessor(int id) {
  assert(size_t(id) < getPolicy().virtmap.size());
  return bindToProcessor(getPolicy().virtmap[id]);
}

unsigned galois::runtime::LL::getProcessorForThread(int id) {
  assert(size_t(id) < getPolicy().virtmap.size());
  return getPolicy().virtmap[id];
}

unsigned galois::runtime::LL::getMaxThreads() {
  return getPolicy().numThreads;
}

unsigned galois::runtime::LL::getMaxCores() {
  return getPolicy().numCores;
}

unsigned galois::runtime::LL::getMaxSockets() {
  return getPolicy().numSockets;
}

unsigned galois::runtime::LL::getSocketForThread(int id) {
  assert(size_t(id) < getPolicy().sockets.size());
  return getPolicy().sockets[id];
}

unsigned galois::runtime::LL::getMaxSocketForThread(int id) {
  assert(size_t(id) < getPolicy().maxSocket.size());
  return getPolicy().maxSocket[id];
}

bool galois::runtime::LL::isSocketLeader(int id) {
  assert(size_t(id) < getPolicy().sockets.size());
  return getPolicy().leaders[getPolicy().sockets[id]] == id;
}

unsigned galois::runtime::LL::getLeaderForThread(int id) {
  assert(size_t(id) < getPolicy().sockets.size());
  return getPolicy().leaders[getPolicy().sockets[id]];
}

unsigned galois::runtime::LL::getLeaderForSocket(int id) {
  assert(size_t(id) < getPolicy().leaders.size());
  return getPolicy().leaders[id];
}
