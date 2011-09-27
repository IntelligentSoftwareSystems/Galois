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

//6 Cores per package, 4 packages
struct FaradayPolicy : public ThreadPolicy {

  virtual void bindThreadToProcessor(int id) {
    //schedule inside each package first
    int carry = 0;
    if (id > 23) {
      id -= 24;
      carry = 24;
    }
    genericBindToProcessor(carry + ((id % 6) * 4) + (id / 6));
  }

  FaradayPolicy() {
    numLevels = 1;
    numThreads = 48;
    numCores = 24;
    levelSize.push_back(4);
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < 4; ++x)
	for (int i = 0; i < 6; ++i)
	  levelMap.push_back(x);
  }
};

//4 Cores per package, 2 packages per blade, 4 blades
struct VoltaPolicy : public ThreadPolicy {

  virtual void bindThreadToProcessor(int id) {
    //1-1 works on volota
    genericBindToProcessor(id);
  }

  VoltaPolicy() {
    numLevels = 2;
    numThreads = 64;
    numCores = 32;

    //NUMA Nodes
    levelSize.push_back(4);
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < 4; ++x)
	for (int i = 0; i < 8; ++i)
	  levelMap.push_back(x);

    //Packages
    levelSize.push_back(8);
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < 8; ++x)
	for (int i = 0; i < 4; ++i)
	  levelMap.push_back(x);
  }
};

//4 Cores per package, 2 packages
struct MaxwellPolicy : public ThreadPolicy {

  virtual void bindThreadToProcessor(int id) {
    //schedule inside each package first
    int carry = 0;
    if (id > 7) {
      id -= 8;
      carry = 8;
    }
    genericBindToProcessor(carry + (id * 2) - ((id /4) * 7));
  }

  MaxwellPolicy() {
    numLevels = 1;
    numThreads = 16;
    numCores = 8;
    levelSize.push_back(2);
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < 2; ++x)
	for (int i = 0; i < 4; ++i)
	  levelMap.push_back(x);
  }
};

//4 Cores per package, 2 packages
struct DelaunayPolicy : public ThreadPolicy {

  virtual void bindThreadToProcessor(int id) {
    genericBindToProcessor(id);
  }

  DelaunayPolicy() {
    numLevels = 1;
    numThreads = 128;
    numCores = 128;
    levelSize.push_back(2);
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < 64; ++x)
	for (int i = 0; i < 64; ++i)
	  levelMap.push_back(x);
  }
};

// 4 Cores per package, 4 packages
struct GaloisPolicy : public ThreadPolicy {

  virtual void bindThreadToProcessor(int id) {
    //1-1 works on galois
    genericBindToProcessor(id);
  }

  GaloisPolicy() {
    numLevels = 1;
    numThreads = 16;
    numCores = 16;
    levelSize.push_back(4);
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < 4; ++x)
        for (int i = 0; i < 4; ++i)
          levelMap.push_back(x);
  }
};



struct DummyPolicy : public ThreadPolicy {

  virtual void bindThreadToProcessor(int id) {
    genericBindToProcessor(id);
  }

  DummyPolicy() {
    numLevels = 1;

#ifdef __linux__
    numThreads = sysconf(_SC_NPROCESSORS_CONF);
#else
    reportWarning("Don't know how to bind thread to cpu on this platform");
    reportWarning("Unknown number of processors (assuming 64)");
    numThreads = 64;
#endif

    numCores = numThreads;
    levelSize.push_back(1);
    for (int x = 0; x < numThreads; ++x)
      levelMap.push_back(0);
  }
};

static ThreadPolicy* TP = 0;

static bool hostnameMatches(std::string hostname, std::string m) {
  return hostname == m || boost::starts_with(hostname, m + ".");
}

static void setSystemThreadPolicy(std::string hostname) {
  ThreadPolicy* newPolicy = 0;
  if (hostnameMatches(hostname, "faraday"))
    newPolicy = new FaradayPolicy();
  else if (hostnameMatches(hostname, "volta"))
    newPolicy = new VoltaPolicy();
  else if (hostnameMatches(hostname, "maxwell"))
    newPolicy = new MaxwellPolicy();
  else if (hostnameMatches(hostname, "galois"))
    newPolicy = new GaloisPolicy();
  else if (hostnameMatches(hostname, "delaunay"))
    newPolicy = new DelaunayPolicy();

  if (newPolicy) {
    std::ostringstream out;
    out << "Using " << hostname << " for thread assignment policy";
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
