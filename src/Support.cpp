/** Support functions -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#include "Galois/Statistic.h"
#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Runtime/Support.h"
#include "Galois/Runtime/ll/StaticInstance.h"
#include "Galois/Runtime/ll/gio.h"
#include "Galois/Runtime/mm/Mem.h"

#include <set>
#include <map>
#include <vector>
#include <string>
#include <cmath>

using Galois::Runtime::LL::gPrint;

namespace {

class StatManager {
  typedef std::pair<std::string, std::string> KeyTy;

  Galois::Runtime::PerThreadStorage<std::map<KeyTy, unsigned long> > Stats;

  volatile unsigned maxID;

  void updateMax(unsigned n) {
    unsigned c;
    while (n > (c = maxID))
      __sync_bool_compare_and_swap(&maxID, c, n);
  }

  KeyTy mkKey(const std::string& loop, const std::string& category) {
    return std::make_pair(loop,category);
  }

  void gather(const std::string& s1, const std::string& s2, unsigned m, 
	      std::vector<unsigned long>& v) {
    for (unsigned x = 0; x < m; ++x)
      v.push_back((*Stats.getRemote(x))[mkKey(s1,s2)]);
  }

  unsigned long getSum(std::vector<unsigned long>& Values, unsigned maxThreadID) {
    unsigned long R = 0;
    for (unsigned x = 0; x < maxThreadID; ++x)
      R += Values[x];
    return R;
  }

public:
  StatManager() :maxID(0) {}

  void addToStat(const std::string& loop, const std::string& category, size_t value) {
    (*Stats.getLocal())[mkKey(loop, category)] += value;
    updateMax(Galois::Runtime::activeThreads);
  }

  void addToStat(Galois::Statistic* value) {
    for (unsigned x = 0; x < Galois::Runtime::activeThreads; ++x)
      (*Stats.getRemote(x))[mkKey(value->getLoopname(), value->getStatname())] += value->getValue(x);
    updateMax(Galois::Runtime::activeThreads);
  }

  void addPageAllocToStat(const std::string& loop, const std::string& category) {
    for (unsigned x = 0; x < Galois::Runtime::activeThreads; ++x)
      (*Stats.getRemote(x))[mkKey(loop, category)] += Galois::Runtime::MM::numPageAllocForThread(x);
    updateMax(Galois::Runtime::activeThreads);
  }

  void addNumaAllocToStat(const std::string& loop, const std::string& category) {
    int nodes = Galois::Runtime::MM::numNumaNodes();
    for (int x = 0; x < nodes; ++x)
      (*Stats.getRemote(x))[mkKey(loop, category)] += Galois::Runtime::MM::numNumaAllocForNode(x);
    updateMax(nodes);
  }

  //Assume called serially
  void printStats() {
    std::set<KeyTy> LKs;
    unsigned maxThreadID = maxID;
    //Find all loops and keys
    for (unsigned x = 0; x < maxThreadID; ++x) {
      std::map<KeyTy, unsigned long>& M = *Stats.getRemote(x);
      for (std::map<KeyTy, unsigned long>::iterator ii = M.begin(), ee = M.end();
	   ii != ee; ++ii) {
	LKs.insert(ii->first);
      }
    }
    //print header
    gPrint("STATTYPE,LOOP,CATEGORY,n,sum");
    for (unsigned x = 0; x < maxThreadID; ++x)
      gPrint(",T", x);
    gPrint("\n");
    //print all values
    for (std::set<KeyTy>::iterator ii = LKs.begin(), ee = LKs.end(); ii != ee; ++ii) {
      std::vector<unsigned long> Values;
      gather(ii->first, ii->second, maxThreadID, Values);
      gPrint("STAT,",
	     ii->first.c_str(), ",",
	     ii->second.c_str(), ",",
	     maxThreadID, ",",
	     getSum(Values, maxThreadID)
	     );
      for (unsigned x = 0; x < maxThreadID; ++x) {
	gPrint(",", Values[x]);
      }
      gPrint("\n");
    }
  }
};

static Galois::Runtime::LL::StaticInstance<StatManager> SM;

}

bool Galois::Runtime::inGaloisForEach = false;

void Galois::Runtime::reportStat(const char* loopname, const char* category, unsigned long value) {
  SM.get()->addToStat(std::string(loopname ? loopname : "(NULL)"), 
		      std::string(category ? category : "(NULL)"),
		      value);
}

void Galois::Runtime::reportStat(const std::string& loopname, const std::string& category, unsigned long value) {
  SM.get()->addToStat(loopname, category, value);
}

void Galois::Runtime::reportStat(Galois::Statistic* value) {
  SM.get()->addToStat(value);
}

void Galois::Runtime::printStats() {
  SM.get()->printStats();
}

void Galois::Runtime::reportPageAlloc(const char* category) {
  SM.get()->addPageAllocToStat(std::string("(NULL)"), std::string(category ? category : "(NULL)"));
}

void Galois::Runtime::reportNumaAlloc(const char* category) {
  SM.get()->addNumaAllocToStat(std::string("(NULL)"), std::string(category ? category : "(NULL)"));
}
