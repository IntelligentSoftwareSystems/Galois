/** Machine Descriptions -*- C++ -*-
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
#include "Galois/Runtime/Threads.h"
#include "Galois/Runtime/Support.h"

#ifdef __linux__
#include <sched.h>
#endif

#if defined(sun) || defined(__sun)
#include <thread.h>
#include <sys/types.h>
#include <sys/processor.h>
#include <sys/procset.h>
#endif

#include <fstream>
#include <cstdio>
#include <cassert>
#include <map>
#include <algorithm>

using namespace GaloisRuntime;

namespace {

static void genericBindToProcessor(int proc) {
#ifdef __linux__
  cpu_set_t mask;
  /* CPU_ZERO initializes all the bits in the mask to zero. */
  CPU_ZERO( &mask );
  
  /* CPU_SET sets only the bit corresponding to cpu. */
  // void to cancel unused result warning
  (void)CPU_SET( proc, &mask );
  
  /* sched_setaffinity returns 0 in success */
  if( sched_setaffinity( 0, sizeof(mask), &mask ) == -1 ) {
    perror("Error");
    reportWarning("Could not set CPU Affinity for thread", (unsigned)proc);
  }
  return;
#endif
#if defined(sun) || defined(__sun)
  if (processor_bind(P_LWPID,  thr_self(), proc, 0) == -1) {
    perror("Error");
    reportWarning("Could not set CPU Affinity for thread", (unsigned)proc);
  }
  return;
#endif

  reportWarning("Don't know how to bind thread to cpu on this platform");
  return;
}

#ifdef __linux__
struct cpuinfo {
  int proc;
  int physid;
  int sib;
  int coreid;
  int cpucores;
};

bool cmp_procinv(const cpuinfo& m1, const cpuinfo& m2) {
  return m1.proc > m2.proc;
}

struct AutoLinuxPolicy : public ThreadPolicy {
    std::vector<int> mapping; //index (galois) -> proc (OS)

  virtual void bindThreadToProcessor() {
    genericBindToProcessor(mapping[ThreadPool::getMyID()]);
  }

  AutoLinuxPolicy() {

    name = "AutoLinux";

    std::vector<cpuinfo> vals;
    int cur = -1;
    
    //PARSE:
    
    std::ifstream fcpuinfo("/proc/cpuinfo");
    char line[1024];
    while (fcpuinfo.getline(line, sizeof(line))) {
      //std::cout << "*" << line << "*\n";
      int num;
      if (sscanf(line, "processor : %d", &num) == 1) {
	assert(cur < num);
	cur = num;
	vals.resize(num + 1);
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
    fcpuinfo.close();
    
    //GROUP:
    typedef std::vector<cpuinfo> L;
    typedef std::map<int, std::map<int, L > > M;
    M byPhysCoreID;
    htRatio = 0;
    for (cur = 0; (unsigned)cur < vals.size(); ++cur) {
      cpuinfo& x = vals[cur];
      byPhysCoreID[x.physid][x.coreid].push_back(x);
      if (htRatio) {
	assert(htRatio == x.sib / x.cpucores);
      } else {
	htRatio = x.sib / x.cpucores;
      }
    }
    
    numThreads = vals.size();
    numCores = byPhysCoreID.size() * byPhysCoreID.begin()->second.begin()->second.begin()->cpucores;
    numPackages = byPhysCoreID.size();

    levelSize.push_back(1);
    levelSize.push_back(numCores / numPackages);
    
    // GERNATE MAPPING:
    for (int ht = 0; ht < htRatio; ++ht) {
      for (M::iterator ii = byPhysCoreID.begin(), ee = byPhysCoreID.end();
	   ii != ee; ++ii) {
	std::map<int, L>& coreids = ii->second;
	for (std::map<int, L>::iterator cii = coreids.begin(), cee = coreids.end(); cii != cee; ++cii) {
	  //assign galois thread num to lowest proc id for this core, phys
	  std::sort(cii->second.begin(), cii->second.end(), cmp_procinv);
	  mapping.push_back(cii->second.back().proc);
	  cii->second.pop_back();
	}
      }
    }
    
    // // PRINT MAPPING:
    // //    printf("htRatio, numPackages, numCores, numThreads: %d, %d, %d, %d\n", htRatio, numPackages, numCores, numThreads);
    // for( unsigned i = 0; i < mapping.size(); ++i)
    //   printf("(%d,%d) ", i, mapping[i]);
    // printf("\n");
    
    // // PRINT LEVELMAP for levels
    // for (int x = 0; x < getNumLevels(); ++x) {
    //   printf("Level %d\n", x);
    //   for (int y = 0; y < getNumThreads(); ++y) {
    // 	printf(" (%d,%d)", y, indexLevelMap(x,y));
    //   }
    //   printf("\n");
    // }

  }

};
#endif

#if defined(sun) || defined(__sun)
//Flat machine with the correct number of threads and binding
struct AutoSunPolicy : public ThreadPolicy {
  
  std::vector<int> procmap; //Galoid id -> solaris id

  virtual void bindThreadToProcessor() {
    genericBindToProcessor(procmap[ThreadPool::getMyID()]);
  }

  AutoSunPolicy() {

    name = "AutoSunPolicy";

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
    htRatio = 1;

    levelSize.push_back(1);
  }
};
#endif

struct DummyPolicy : public ThreadPolicy {

  virtual void bindThreadToProcessor() {
    genericBindToProcessor(ThreadPool::getMyID());
  }

  DummyPolicy() {
    htRatio = 1;

    name = "Dummy";

    reportWarning("Unknown number of processors (assuming 128)");
    numThreads = 128;
    numCores = numThreads;
    levelSize.push_back(1);
  }
};

}

static ThreadPolicy* TP = 0;

ThreadPolicy& GaloisRuntime::getSystemThreadPolicy() {
  bool printthing = !TP;
#ifdef __linux__
  if (!TP) TP = new AutoLinuxPolicy();
#endif
#if defined(sun) || defined(__sun)
  if (!TP) TP = new AutoSunPolicy();
#endif
  if (!TP) TP = new DummyPolicy();
  
  if (printthing)
    reportInfo("Thread Policy", TP->getName());

  return *TP;
}
