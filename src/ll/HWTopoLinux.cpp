/** Machine Descriptions on Linux -*- C++ -*-
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
 * This contains descriptions of machine topologies.  These describes levels in
 * the machine.  The lowest level is a package.
 *
 * This also matches OS cpu numbering to galois thread numbering and binds threads
 * to processors.  Threads are assigned densly in each package before the next 
 * package.  SMT hardware contexts are bound after all real cores (int x86).
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
*/
#ifdef __linux__

#include "Galois/Runtime/ll/HWTopo.h"
#include "Galois/Runtime/ll/gio.h"

#define DEBUG_HWTOPOLINUX 0

#include <stdio.h>
#include <sched.h>
#include <errno.h>
#include <string.h>
#include <assert.h>

#include <vector>
#include <algorithm>

using namespace GaloisRuntime::LL;

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

#if DEBUG_HWTOPOLINUX
void printCPUINFO(const cpuinfo& p) {
  gPrint("(proc %d, physid %d, sib %d, coreid %d, cpucores %d\n",
	 p.proc, p.physid, p.sib, p.coreid, p.cpucores);
}
#endif

static bool linuxBindToProcessor(int proc) {
  cpu_set_t mask;
  /* CPU_ZERO initializes all the bits in the mask to zero. */
  CPU_ZERO( &mask );
  
  /* CPU_SET sets only the bit corresponding to cpu. */
  // void to cancel unused result warning
  (void)CPU_SET( proc, &mask );
  
  /* sched_setaffinity returns 0 in success */
  if( sched_setaffinity( 0, sizeof(mask), &mask ) == -1 ) {
    gWarn("Could not set CPU affinity for thread %d (%s)", proc, strerror(errno));
    return false;
  }
  return true;
}

static void openFailed(const char* s) {
  gError(true, "opening %s failed with %s\n", s, strerror(errno));
}

static std::vector<cpuinfo> parseCPUInfo() {
  //PARSE: /proc/cpuinfo

  std::vector<cpuinfo> vals;
  vals.reserve(64);

  FILE* f = fopen(sProcInfo, "r");
  if (!f) {
    openFailed(sProcInfo);
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

std::vector<int> parseCPUSet() {
  std::vector<int> vals;
  vals.reserve(64);

  //PARSE: /proc/self/cpuset
  FILE* f = fopen(sCPUSet, "r");
  if (!f) {
    fclose(f);
    return vals;
  }

  const int len = 1024;
  char* path = (char*)malloc(len);
  path[0] = '/';
  path[1] = '\0';
  fgets(path, len, f);
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
    openFailed(path2);
    return vals; //Shouldn't get here
  }

  //reuse path
  char* np = path;
  fgets(np, len, f);
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
  int numThreads, numThreadsRaw;
  
  //number of "real" processors
  int numCores, numCoresRaw;

  //number of packages
  int numPackages, numPackagesRaw;

  std::vector<int> packages;
  std::vector<int> maxPackage;
  std::vector<int> virtmap;

  AutoLinuxPolicy() {

    std::vector<cpuinfo> vals = parseCPUInfo();
    virtmap = parseCPUSet();

    std::vector<int>::iterator tempi;

    if (virtmap.empty()) {
      //1-1 mapping for non-cpuset using systems
      for (int i = 0; i < (int)vals.size(); ++i)
	virtmap.push_back(i);
    }

#if DEBUG_HWTOPOLINUX
    //DEBUG:
    for (int i = 0; i < vals.size(); ++i)
      printCPUINFO(vals[i]);
    for (int i = 0; i < virtmap.size(); ++i)
      gPrint("%d, ", virtmap[i]);
    gPrint("\n");
    //End DEBUG
#endif

    //Get thread count
    numThreadsRaw = vals.size();
    numThreads = virtmap.size();

    //Get package level stuff
    int maxrawpackage;
    //First get raw info
    for (int i = 0; i < (int)vals.size(); ++i)
      packages.push_back(vals[i].physid);
    maxrawpackage = *std::max_element(packages.begin(), packages.end());
    std::sort(packages.begin(), packages.end());
    tempi = std::unique(packages.begin(), packages.end());
    numPackagesRaw = std::distance(packages.begin(), tempi);
    packages.clear();
    //Second get cpuset info
    for (int i = 0; i < (int)virtmap.size(); ++i)
      packages.push_back(vals[virtmap[i]].physid);
    std::sort(packages.begin(), packages.end());
    tempi = std::unique(packages.begin(), packages.end());
    numPackages = std::distance(packages.begin(), tempi);
    packages.clear();
    //finally renumber for virtual processor numbers
    {
      std::vector<int> mapping(maxrawpackage+1);
      int nextval = 1;
      for (int i = 0; i < (int)virtmap.size(); ++i) {
	int x = vals[virtmap[i]].physid;
	if (!mapping[x])
	  mapping[x] = nextval++;
	packages.push_back(mapping[x]-1);
      }
    }

    //Get core count
    std::vector<std::pair<int, int> > cores;
    //but first get the raw numbers
    for (int i = 0; i < (int)vals.size(); ++i)
      cores.push_back(std::make_pair(vals[i].physid, vals[i].coreid));
    std::sort(cores.begin(), cores.end());
    std::vector<std::pair<int,int> >::iterator core_u = std::unique(cores.begin(), cores.end());
    numCoresRaw = std::distance(cores.begin(), core_u);
    cores.clear();
    for (int i = 0; i < (int)virtmap.size(); ++i)
      cores.push_back(std::make_pair(vals[virtmap[i]].physid, vals[virtmap[i]].coreid));
    std::sort(cores.begin(), cores.end());
    core_u = std::unique(cores.begin(), cores.end());
    numCores = std::distance(cores.begin(), core_u);
 
    //Compute cummulative max package
    int p = 0;
    maxPackage.resize(packages.size());
    for (int i = 0; i < (int)packages.size(); ++i) {
      p = std::max(packages[i],p);
      maxPackage[i] = p;
    }

#if DEBUG_HWTOPOLINUX
    //DEBUG: PRINT Stuff
    gPrint("Threads: %d, %d (raw)\n", numThreads, numThreadsRaw);
    gPrint("Cores: %d, %d (raw)\n", numCores, numCoresRaw);
    gPrint("Packages: %d, %d (raw)\n", numPackages, numPackagesRaw);

    for (int i = 0; i < virtmap.size(); ++i) {
      gPrint("T %3d P %3d Tr %3d", i, packages[i], virtmap[i]);
      if (i >= numCores)
	gPrint(" HT");
      gPrint("\n");
    }
#endif
  }

};

AutoLinuxPolicy A;

} //namespace



bool GaloisRuntime::LL::bindThreadToProcessor(int id) {
  assert(id < (int)A.virtmap.size());
  return linuxBindToProcessor(A.virtmap[id]);
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

unsigned GaloisRuntime::LL::getPackageForThread(int id) {
  assert(id < (int)A.packages.size());
  return A.packages[id];
}

unsigned GaloisRuntime::LL::getMaxPackageForThread(int id) {
  assert(id < (int)A.maxPackage.size());
  return A.maxPackage[id];
}

#endif //__linux__
