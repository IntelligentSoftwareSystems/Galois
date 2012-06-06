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
#include "Galois/Runtime/ll/StaticInstance.h"
#include "Galois/Runtime/ll/gio.h"
#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Runtime/Support.h"
#include "Galois/Threads.h"
#include "Galois/Statistic.h"

#include <set>
#include <map>
#include <vector>
#include <string>
#include <cmath>

using GaloisRuntime::LL::gPrint;

namespace {

class StatManager {

  typedef std::pair<std::string, std::string> KeyTy;

  GaloisRuntime::PerThreadStorage<std::map<KeyTy, unsigned long> > Stats;

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

  double getAvg(std::vector<unsigned long>& Values, unsigned maxThreadID) {
    double R = 0.0;
    for (unsigned x = 0; x < maxThreadID; ++x)
      R += (double)Values[x];
    return R / (double)maxThreadID;
  }

  unsigned long getMin(std::vector<unsigned long>& Values, unsigned maxThreadID) {
    unsigned long R = Values[0];
    for (unsigned x = 1; x < maxThreadID; ++x)
      R = std::min(R, Values[x]);
    return R;
  }

  unsigned long getMax(std::vector<unsigned long>& Values, unsigned maxThreadID) {
    unsigned long R = Values[0];
    for (unsigned x = 1; x < maxThreadID; ++x)
      R = std::max(R, Values[x]);
    return R;
  }

  double getStddev(std::vector<unsigned long>& Values, unsigned maxThreadID) {
    double avg = getAvg(Values, maxThreadID);
    double Diff = 0.0;
    for (unsigned x = 0; x < maxThreadID; ++x) {
      double R = avg - (double)Values[x];
      Diff += R*R;
    }
    return std::sqrt(Diff / (double)maxThreadID);
  }
  
public:
  void addToStat(const std::string& loop, const std::string& category, size_t value) {
    (*Stats.getLocal())[mkKey(loop, category)] += value;
  }

  void addToStat(Galois::Statistic* value) {
    for (unsigned x = 0; x < Galois::getActiveThreads(); ++x)
      (*Stats.getRemote(x))[mkKey(value->getLoopname(), value->getStatname())] += value->getValue(x);
  }

  //Assume called serially
  void printStats() {
    std::set<std::string> Loops;
    std::set<std::string> Keys;
    unsigned maxThreadID = Galois::getActiveThreads();
    //Find all loops and keys
    for (unsigned x = 0; x < maxThreadID; ++x) {
      std::map<KeyTy, unsigned long>& M = *Stats.getRemote(x);
      for (std::map<KeyTy, unsigned long>::iterator ii = M.begin(), ee = M.end();
	   ii != ee; ++ii) {
	Loops.insert(ii->first.first);
	Keys.insert(ii->first.second);
      }
    }
    //print header
    gPrint("STATTYPE,LOOP,CATEGORY,n,sum");
    for (unsigned x = 0; x < maxThreadID; ++x)
      gPrint(",T%d", x);
    gPrint("\n");
    //print all values
    for(std::set<std::string>::iterator iiL = Loops.begin(), eeL = Loops.end();
	iiL != eeL; ++iiL) {
      for (std::set<std::string>::iterator iiK = Keys.begin(), eeK = Keys.end();
	   iiK != eeK; ++iiK) {
	std::vector<unsigned long> Values;
	gather(*iiL, *iiK, maxThreadID, Values);
	gPrint("STAT,%s,%s,%u,%lu", 
	       iiL->c_str(), 
	       iiK->c_str(),
	       maxThreadID,
               getSum(Values, maxThreadID)
	       );
	for (unsigned x = 0; x < maxThreadID; ++x) {
	  gPrint(",%ld", Values[x]);
	}
	gPrint("\n");
      }
    }
  }
};

static GaloisRuntime::LL::StaticInstance<StatManager> SM;

}


bool GaloisRuntime::inGaloisForEach = false;

void GaloisRuntime::reportStat(const char* loopname, const char* category, size_t value) {
  SM.get()->addToStat(std::string(loopname ? loopname : "(NULL)"), 
		      std::string(category ? category : "(NULL)"),
		      value);
}

void GaloisRuntime::reportStat(const std::string& loopname, const std::string& category, size_t value) {
  SM.get()->addToStat(loopname, category, value);
}

void GaloisRuntime::reportStat(Galois::Statistic* value) {
  SM.get()->addToStat(value);
}

void GaloisRuntime::printStats() {
  SM.get()->printStats();
}


