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
#include "Galois/Runtime/ll/SimpleLock.h"
#include "Galois/Runtime/Support.h"
#include "llvm/ADT/SmallVector.h"
#include <map>
#include <vector>
#include <cstdio>

static GaloisRuntime::LL::SimpleLock<int, true> lock;

namespace {
class PrintStats {
  typedef std::pair<const char*, const char*> strPair;
  typedef std::map<strPair, unsigned long> StatsMap;
  typedef std::map<int, GaloisRuntime::OnlineStatistics> DistStatsValue;
  typedef std::map<strPair, DistStatsValue> DistStatsMap;
  StatsMap stats;
  DistStatsMap distStats;
  int gcounter;

  void summarizeList(int iternum, const char* first, const char* second,
      const GaloisRuntime::OnlineStatistics& x) {
    printf("STAT DISTRIBUTION %d %s %s n: %zu ave: %.1f "
      "min: %.0f max: %.0f stdev: %.1f\n",
      iternum, first, second, x.n(), x.mean(), x.min(), x.max(), sqrt(x.sample_variance()));
  }

public:
  PrintStats() : gcounter(0) { }
  ~PrintStats() {
    for (StatsMap::iterator ii = stats.begin(), ee = stats.end();
        ii != ee; ++ii) {
      printf("STAT SINGLE %s %s %ld\n",
          ii->first.first, 
          ii->first.second ? ii->first.second : "(null)",
          ii->second);
    }
    for (DistStatsMap::iterator ii = distStats.begin(), ee = distStats.end();
        ii != ee; ++ii) {
      for (DistStatsValue::iterator i = ii->second.begin(), e = ii->second.end();
          i != e; ++i) {
        summarizeList(i->first,
            ii->first.first ? ii->first.first : "(null)",
            ii->first.second ? ii->first.second : "(null)",
            i->second);
      } 
    }
  }

  void reportStatAvg(const char* text, double val, const char* loopname) {
    distStats[std::make_pair(text,loopname)][gcounter].push(val);
  }

  void reportStatSum(const char* text, unsigned long val, const char* loopname) {
    stats[std::make_pair(text,loopname)] += val;
  }

  void incIteration() {
    ++gcounter;
  }
};
}

std::ostream& operator<<(std::ostream& os, const GaloisRuntime::OnlineStatistics& x) {
  os << "n: " << x.n()
    << " ave: " << x.mean()
    << " min: " << x.min()
    << " max: " << x.max()
    << " stdev: " << x.sample_variance();
  return os;
}

static PrintStats& getPS() {
  static PrintStats P;
  return P;
}

void GaloisRuntime::reportStatSum(const char* text, unsigned long val, const char* loopname) {
  getPS().reportStatSum(text, val, loopname);
}

void GaloisRuntime::reportStatAvg(const char* text, unsigned long val, const char* loopname) {
  getPS().reportStatAvg(text, val, loopname);
}

void GaloisRuntime::statDone() {
  getPS().incIteration();
}

void GaloisRuntime::reportFlush() {
  fflush(stdout);
}
