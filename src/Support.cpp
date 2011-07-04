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
#include "Galois/Runtime/SimpleLock.h"
#include "Galois/Runtime/Support.h"
#include "LLVM/SmallVector.h"
#include <map>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdio>

static GaloisRuntime::SimpleLock<int, true> lock;

namespace {
class PrintStats {
  typedef std::pair<const char*, const char*> strPair;
  typedef std::map<strPair, unsigned long> StatsMap;
  typedef std::map<int, std::vector<unsigned long> > DistStatsValue;
  typedef std::map<strPair, DistStatsValue> DistStatsMap;
  typedef std::vector<unsigned long>::iterator long_iterator;
  StatsMap stats;
  DistStatsMap distStats;
  int gcounter;

  void summarizeList(int iternum, const char* first, const char* second,
      long_iterator b, long_iterator e) {
    long size = std::distance(b,e);
    long min = *std::min_element(b, e);
    long max = *std::max_element(b, e);
    double ave = std::accumulate(b, e, 0.0) / size;
   
    double acc = 0.0;
    for (long_iterator it = b; it != e; ++it) {
      acc += (*it - ave) * (*it - ave);
    }

    double stdev = 0.0;
    if (size > 1) {
      stdev = sqrt(acc / (size - 1));
    }

    printf("STAT DISTRIBUTION %d %s %s n: %ld ave: %.1f "
        "min: %ld max: %ld stdev: %.1f\n",
           iternum, first, second, size, ave, min, max, stdev);
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
            i->second.begin(),
            i->second.end());
      } 
    }
  }

  void reportStatAvg(const char* text, unsigned long val, const char* loopname) {
    distStats[std::make_pair(text,loopname)][gcounter].push_back(val);
  }

  void reportStatSum(const char* text, unsigned long val, const char* loopname) {
    stats[std::make_pair(text,loopname)] += val;
  }

  void incIteration() {
    ++gcounter;
  }
};
}
PrintStats P;

void GaloisRuntime::reportStatSum(const char* text, unsigned long val, const char* loopname) {
  P.reportStatSum(text, val, loopname);
}

void GaloisRuntime::reportStatAvg(const char* text, unsigned long val, const char* loopname) {
  P.reportStatAvg(text, val, loopname);
}

void GaloisRuntime::statDone() {
  P.incIteration();
}

static void genericReport(bool error, const char* text1,
    const char* text2, const char* val) {
  lock.lock();
  FILE *out = error ? stderr : stdout;
  fprintf(out, "%s %s %s\n", text1, text2, val);
  lock.unlock();
}

//Report Warnings
void GaloisRuntime::reportWarning(const char* text) {
  genericReport(true, "WARNING:", text, "");
}

void GaloisRuntime::reportWarning(const char* text, unsigned int val) {
  char buf[128];
  snprintf(buf, 128, "%u", val);
  genericReport(true, "WARNING:", text, buf);
}

void GaloisRuntime::reportWarning(const char* text, unsigned long val) {
  char buf[128];
  snprintf(buf, 128, "%lu", val);
  genericReport(true, "WARNING:", text, buf);
}

void GaloisRuntime::reportWarning(const char* text, const char* val) {
  genericReport(true, "WARNING:", text, val);
}

//Report Info
void GaloisRuntime::reportInfo(const char* text) {
  genericReport(false, "INFO:", text, "");
}

void GaloisRuntime::reportInfo(const char* text, unsigned int val) {
  char buf[128];
  snprintf(buf, 128, "%u", val);
  genericReport(false, "INFO:", text, buf);
}

void GaloisRuntime::reportInfo(const char* text, unsigned long val) {
  char buf[128];
  snprintf(buf, 128, "%lu", val);
  genericReport(false, "INFO:", text, buf);
}

void GaloisRuntime::reportInfo(const char* text, const char* val) {
  genericReport(false, "INFO:", text, val);
}

/// grow_pod - This is an implementation of the grow() method which only works
/// on POD-like datatypes and is out of line to reduce code duplication.
void llvm::SmallVectorBase::grow_pod(size_t MinSizeInBytes, size_t TSize) {
  size_t CurSizeBytes = size_in_bytes();
  size_t NewCapacityInBytes = 2 * capacity_in_bytes() + TSize; // Always grow.
  if (NewCapacityInBytes < MinSizeInBytes)
    NewCapacityInBytes = MinSizeInBytes;

  void *NewElts;
  if (this->isSmall()) {
    NewElts = malloc(NewCapacityInBytes);

    // Copy the elements over.  No need to run dtors on PODs.
    memcpy(NewElts, this->BeginX, CurSizeBytes);
  } else {
    // If this wasn't grown from the inline copy, grow the allocated space.
    NewElts = realloc(this->BeginX, NewCapacityInBytes);
  }

  this->EndX = (char*)NewElts+CurSizeBytes;
  this->BeginX = NewElts;
  this->CapacityX = (char*)this->BeginX + NewCapacityInBytes;
}
