/** Machine Descriptions on Linux -*- C++ -*-
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
 * This contains descriptions of machine topologies.  These describes levels in
 * the machine.  The lowest level is a package.
 *
 * This also matches OS cpu numbering to galois thread numbering and binds threads
 * to processors.  Threads are assigned densly in each package before the next 
 * package.  SMT hardware contexts are bound after all real cores (int x86).
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
*/
#include "Galois/Substrate/HWTopo.h"
#include "Galois/Substrate/EnvCheck.h"
#include "Galois/gIO.h"

#include <vector>
#include <functional>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <cerrno>

#include <sched.h>

using namespace galois::substrate;

namespace {

struct cpuinfo {
  int proc;
  int physid;
  int sib;
  int coreid;
  int cpucores;
};

static const char* sProcInfo = "/proc/cpuinfo";
static const char* sCPUSet   = "/proc/self/cpuset";

static bool linuxBindToProcessor(int proc) {
  // cpu_set_t mask;
  // /* CPU_ZERO initializes all the bits in the mask to zero. */
  // CPU_ZERO( &mask );
  
  // /* CPU_SET sets only the bit corresponding to cpu. */
  // // void to cancel unused result warning
  // (void)CPU_SET( proc, &mask );
  
  // /* sched_setaffinity returns 0 in success */
  // if( sched_setaffinity( 0, sizeof(mask), &mask ) == -1 ) {
  //   gWarn("Could not set CPU affinity for thread ", proc, "(", strerror(errno), ")");
  //   return false;
  // }
  return false;
}

//! Parse /proc/cpuinfo
static std::vector<cpuinfo> parseCPUInfo() {
  std::vector<cpuinfo> vals;
  vals.reserve(64);

  FILE* f = fopen(sProcInfo, "r");
  if (!f) {
    GALOIS_SYS_DIE("failed opening ", sProcInfo);
    return vals; //Shouldn't get here
  }

  const int len = 1024;
  char* line = (char*)malloc(len);
  int cur = -1;

  while (fgets(line, len, f)) {
    int num;
    if (sscanf(line, "processor : %d", &num) == 1) {
      assert(cur < num);
      cur = num;
      vals.resize(cur + 1);
      vals.at(cur).proc = num;
    } else if (sscanf(line, "physical id : %d", &num) == 1) {
      vals.at(cur).physid = num;
    } else if (sscanf(line, "siblings : %d", &num) == 1) {
      vals.at(cur).sib = num;
    } else if (sscanf(line, "core id : %d", &num) == 1) {
      vals.at(cur).coreid = num;
    } else if (sscanf(line, "cpu cores : %d", &num) == 1) {
      vals.at(cur).cpucores = num;
    }
  }

  free(line);
  fclose(f);

  return vals;
}

//! Returns physical ids in current cpuset
std::vector<int> parseCPUSet() {
  std::vector<int> vals;
  vals.reserve(64);

  //PARSE: /proc/self/cpuset
  FILE* f = fopen(sCPUSet, "r");
  if (!f) {
    return vals;
  }

  const int len = 1024;
  char* path = (char*)malloc(len);
  path[0] = '/';
  path[1] = '\0';
  if (!fgets(path, len, f)) {
    fclose(f);
    return vals;
  }
  fclose(f);

  if(char* t = index(path, '\n'))
    *t = '\0';

  if (strlen(path) == 1) {
    free(path);
    return vals;
  }

  char* path2 = (char*)malloc(len);
  strcpy(path2, "/dev/cpuset");
  strcat(path2, path);
  strcat(path2, "/cpus");

  f = fopen(path2, "r");
  if (!f) {
    free(path2);
    free(path);
    GALOIS_SYS_DIE("failed opening ", path2);
    return vals; //Shouldn't get here
  }

  //reuse path
  char* np = path;
  if (!fgets(np, len, f)) {
    fclose(f);
    return vals;
  }
  while (np && strlen(np)) {
    char* c = index(np, ',');  
    if (c) { //slice string at comma (np is old string, c is next string
      *c = '\0';
      ++c;
    }
    
    char* d = index(np, '-');
    if (d) { //range
      *d = '\0';
      ++d;
      int b = atoi(np);
      int e = atoi(d);
      while (b <= e)
	vals.push_back(b++);
    } else { //singleton
      vals.push_back(atoi(np));
    }
    np = c;
  };
  
  fclose(f);
  free(path2);
  free(path);
  return vals;
}

struct AutoLinuxPolicy {
  //number of hw supported threads
  unsigned numThreads, numThreadsRaw;
  
  //number of "real" processors
  unsigned numCores, numCoresRaw;

  //number of packages
  unsigned numPackages, numPackagesRaw;

  std::vector<int> packages;
  std::vector<int> maxPackage;
  std::vector<int> virtmap;
  std::vector<int> leaders;

  //! Sort in package-dense manner
  struct DensePackageLessThan: public std::binary_function<int,int,bool> {
    const std::vector<cpuinfo>& vals;
    DensePackageLessThan(const std::vector<cpuinfo>& v): vals(v) { }
    bool operator()(int a, int b) const {
      if (vals[a].physid < vals[b].physid) {
        return true;
      } else if (vals[a].physid == vals[b].physid) {
        if (vals[a].coreid < vals[b].coreid) {
          return true;
        } else if (vals[a].coreid == vals[b].coreid) {
          return vals[a].proc < vals[b].proc;
        } else {
          return false;
        }
      } else {
        return false;
      }
    }
  };

  struct DensePackageEqual: public std::binary_function<int,int,bool> {
    const std::vector<cpuinfo>& vals;
    DensePackageEqual(const std::vector<cpuinfo>& v): vals(v) { }
    bool operator()(int a, int b) const {
      return vals[a].physid == vals[b].physid && vals[a].coreid == vals[b].coreid;
    }
  };

  AutoLinuxPolicy() {
    std::vector<cpuinfo> vals = parseCPUInfo();
    virtmap = parseCPUSet();

    if (virtmap.empty()) {
      //1-1 mapping for non-cpuset using systems
      for (unsigned i = 0; i < vals.size(); ++i)
	virtmap.push_back(i);
    }

    if (EnvCheck("GALOIS_DEBUG_TOPO"))
      printRawConfiguration(vals);

    //Get thread count
    numThreadsRaw = vals.size();
    numThreads = virtmap.size();

    //Get package level stuff
    int maxrawpackage = generateRawPackageData(vals);
    generatePackageData(vals);
    
    //Sort by package to get package-dense mapping
    std::sort(virtmap.begin(), virtmap.end(), DensePackageLessThan(vals));
    generateHyperthreads(vals);

    //Finally renumber for virtual processor numbers
    finalizePackageData(vals, maxrawpackage);

    //Get core count
    numCores = generateCoreData(vals);
 
    //Compute cummulative max package
    int p = 0;
    maxPackage.resize(packages.size());
    for (int i = 0; i < (int)packages.size(); ++i) {
      p = std::max(packages[i],p);
      maxPackage[i] = p;
    }

    //Compute first thread in package
    leaders.resize(numPackages, -1);
    for (int i = 0; i < (int) packages.size(); ++i)
      if (leaders[packages[i]] == -1)
	leaders[packages[i]] = i;

    if (EnvCheck("GALOIS_DEBUG_TOPO"))
      printFinalConfiguration(); 
  }

  void printRawConfiguration(const std::vector<cpuinfo>& vals) {
    for (unsigned i = 0; i < vals.size(); ++i) {
      const cpuinfo& p = vals[i];
      gPrint("proc ", p.proc, ", physid ", p.physid, ", sib ", p.sib, ", coreid ", p.coreid, ", cpucores ", p.cpucores, "\n");
    }
    for (unsigned i = 0; i < virtmap.size(); ++i)
      gPrint(", ", virtmap[i]);
    gPrint("\n");
  }

  void printFinalConfiguration() {
    //DEBUG: PRINT Stuff
    gPrint("Threads: ", numThreads, ", ", numThreadsRaw, " (raw)\n");
    gPrint("Cores: ", numCores, ", ", numCoresRaw, " (raw)\n");
    gPrint("Packages: ", numPackages, ", ", numPackagesRaw, " (raw)\n");

    for (unsigned i = 0; i < virtmap.size(); ++i) {
      gPrint(
          "T ", i, 
          " P ", packages[i],
          " Tr ", virtmap[i], 
          " L? ", ((int)i == leaders[packages[i]] ? 1 : 0));
      if (i >= numCores)
	gPrint(" HT");
      gPrint("\n");
    }
  }

  void finalizePackageData(const std::vector<cpuinfo>& vals, int maxrawpackage) {
    std::vector<int> mapping(maxrawpackage+1);
    int nextval = 1;
    for (int i = 0; i < (int)virtmap.size(); ++i) {
      int x = vals[virtmap[i]].physid;
      if (!mapping[x])
        mapping[x] = nextval++;
      packages.push_back(mapping[x]-1);
    }
  }

  unsigned generateCoreData(const std::vector<cpuinfo>& vals) {
    std::vector<std::pair<int, int> > cores;
    //first get the raw numbers
    for (unsigned i = 0; i < vals.size(); ++i)
      cores.push_back(std::make_pair(vals[i].physid, vals[i].coreid));
    std::sort(cores.begin(), cores.end());
    std::vector<std::pair<int,int> >::iterator it = std::unique(cores.begin(), cores.end());
    numCoresRaw = std::distance(cores.begin(), it);
    cores.clear();

    for (unsigned i = 0; i < virtmap.size(); ++i)
      cores.push_back(std::make_pair(vals[virtmap[i]].physid, vals[virtmap[i]].coreid));
    std::sort(cores.begin(), cores.end());
    it = std::unique(cores.begin(), cores.end());
    return std::distance(cores.begin(), it);
  }

  void generateHyperthreads(const std::vector<cpuinfo>& vals) {
    //Find duplicates, which are hyperthreads, and place them at the end
    // annoyingly, values after tempi are unspecified for std::unique, so copy in and out instead
    std::vector<int> dense(numThreads);
    std::vector<int>::iterator it = std::unique_copy(virtmap.begin(), virtmap.end(), 
        dense.begin(), DensePackageEqual(vals));
    std::vector<bool> mask(numThreadsRaw);
    for (std::vector<int>::iterator ii = dense.begin(); ii < it; ++ii)
      mask[*ii] = true;
    for (std::vector<int>::iterator ii = virtmap.begin(), ei = virtmap.end(); ii < ei; ++ii) {
      if (!mask[*ii])
        *it++ = *ii;
    }
    virtmap = dense;
  }
  
  void generatePackageData(const std::vector<cpuinfo>& vals) {
    std::vector<int> p;
    for (unsigned i = 0; i < virtmap.size(); ++i)
      p.push_back(vals[virtmap[i]].physid);
    std::sort(p.begin(), p.end());
    std::vector<int>::iterator it = std::unique(p.begin(), p.end());
    numPackages = std::distance(p.begin(), it);
  }

  int generateRawPackageData(const std::vector<cpuinfo>& vals) {
    std::vector<int> p;
    for (unsigned i = 0; i < vals.size(); ++i)
      p.push_back(vals[i].physid);

    int retval = *std::max_element(p.begin(), p.end());
    std::sort(p.begin(), p.end());
    std::vector<int>::iterator it = std::unique(p.begin(), p.end());
    numPackagesRaw = std::distance(p.begin(), it);
    return retval;
  }
};

AutoLinuxPolicy& getPolicy() {
  static AutoLinuxPolicy A;
  return A;
}

} //namespace

bool galois::runtime::LL::bindThreadToProcessor(int id) {
  assert(size_t(id) < getPolicy().virtmap.size());
  return linuxBindToProcessor(getPolicy().virtmap[id]);
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

unsigned galois::runtime::LL::getMaxPackages() {
  return getPolicy().numPackages;
}

unsigned galois::runtime::LL::getPackageForThread(int id) {
  assert(size_t(id) < getPolicy().packages.size());
  return getPolicy().packages[id];
}

unsigned galois::runtime::LL::getMaxPackageForThread(int id) {
  assert(size_t(id) < getPolicy().maxPackage.size());
  return getPolicy().maxPackage[id];
}

bool galois::runtime::LL::isPackageLeader(int id) {
  assert(size_t(id) < getPolicy().packages.size());
  return getPolicy().leaders[getPolicy().packages[id]] == id;
}

unsigned galois::runtime::LL::getLeaderForThread(int id) {
  assert(size_t(id) < getPolicy().packages.size());
  return getPolicy().leaders[getPolicy().packages[id]];
}

unsigned galois::runtime::LL::getLeaderForPackage(int id) {
  assert(size_t(id) < getPolicy().leaders.size());
  return getPolicy().leaders[id];
}
