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

#include <boost/algorithm/string/predicate.hpp>

#include <sched.h>
#include <string.h>
#include <sstream>
#include <fstream>
#include <cstdio>
#include <map>
#include <iostream>


using namespace GaloisRuntime;

static void genericBindToProcessor(int proc) {
#ifdef __linux__
  cpu_set_t mask;
  /* CPU_ZERO initializes all the bits in the mask to zero. */
  CPU_ZERO( &mask );
  
  /* CPU_SET sets only the bit corresponding to cpu. */
  // void to cancel unused result warning
  (void)CPU_SET( proc, &mask );
  
  /* sched_setaffinity returns 0 in success */
  if( sched_setaffinity( 0, sizeof(mask), &mask ) == -1 )
    reportWarning("Could not set CPU Affinity for thread", (unsigned)proc);
  
  return;
#endif      
}


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
    
    // PRINT MAPPING:
    //    printf("htRatio, numPackages, numCores, numThreads: %d, %d, %d, %d\n", htRatio, numPackages, numCores, numThreads);
    for( unsigned i = 0; i < mapping.size(); ++i)
      printf("(%d,%d) ", i, mapping[i]);
    printf("\n");
    
    // PRINT LEVELMAP for levels
    for (int x = 0; x < getNumLevels(); ++x) {
      printf("Level %d\n", x);
      for (int y = 0; y < getNumThreads(); ++y) {
	printf(" (%d,%d)", y, indexLevelMap(x,y));
      }
      printf("\n");
    }

  }

};

struct DummyPolicy : public ThreadPolicy {

  virtual void bindThreadToProcessor() {
    genericBindToProcessor(ThreadPool::getMyID());
  }

  DummyPolicy() {
    htRatio = 1;

    name = "Dummy";

    reportWarning("Don't know how to bind thread to cpu on this platform");
    reportWarning("Unknown number of processors (assuming 64)");
    numThreads = 128;
    numCores = numThreads;
    levelSize.push_back(1);
  }
};

static ThreadPolicy* TP = 0;

static bool hostnameMatches(std::string hostname, std::string m) {
  return hostname == m || boost::starts_with(hostname, m + ".");
}

static void setSystemThreadPolicy(std::string hostname) {
  ThreadPolicy* newPolicy = 0;
#ifdef __linux__
  newPolicy = new AutoLinuxPolicy();
#endif

  if (newPolicy) {
    std::ostringstream out;
    out << "Using " << newPolicy->getName() << " for thread assignment policy";
    reportInfo("ThreadPool", out.str().c_str());
  } else {
    newPolicy = new DummyPolicy();
    reportWarning("using default thread assignment policy");
  }

  if (TP)
    delete TP;
  TP = newPolicy;
}

ThreadPolicy& GaloisRuntime::getSystemThreadPolicy() {
  if (!TP) {
    //try autodetecting policy from hostname
    char name[256];
    int r = gethostname(name, 256);
    if (!r)
      setSystemThreadPolicy(name);
    else 
      TP = new DummyPolicy();
  }
  return *TP;
}
