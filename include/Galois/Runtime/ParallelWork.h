// simple galois scheduler and runtime -*- C++ -*-
/*
Galois, a framework to exploit amorphous data-parallelism in irregular
programs.

Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS SOFTWARE
AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY
PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY
WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF TRADE.
NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO THE USE OF THE
SOFTWARE OR DOCUMENTATION. Under no circumstances shall University be liable
for incidental, special, indirect, direct or consequential damages or loss of
profits, interruption of business, or related expenses which may arise from use
of Software or Documentation, including but not limited to those resulting from
defects in Software and/or Documentation, or loss or inaccuracy of data of any
kind.
*/

#ifndef __PARALLELWORK_H_
#define __PARALLELWORK_H_
#include <algorithm>
#include <numeric>
#include <sstream>
#include <math.h>
#include "Galois/Executable.h"
#include "Galois/PerIterMem.h"

#include "Galois/Runtime/Support.h"
#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/Timer.h"
#include "Galois/Runtime/Threads.h"
#include "Galois/Runtime/PerCPU.h"
#include "Galois/Runtime/WorkList.h"
#include "Galois/Runtime/DebugWorkList.h"
#include "Galois/Runtime/Termination.h"

#ifdef GALOIS_VTUNE
#include "ittnotify.h"
#endif

namespace GaloisRuntime {

template<class WorkListTy>
class ParallelThreadContext
  : public SimpleRuntimeContext, public Galois::PerIterMem
{
  typedef typename WorkListTy::value_type value_type;
  typedef GaloisRuntime::WorkList::MP_SC_FIFO<value_type> AbortedListTy;

  unsigned long conflicts;
  unsigned long iterations;
  WorkListTy* wl;
  AbortedListTy* aborted;
  
  using SimpleRuntimeContext::start_iteration;
  using SimpleRuntimeContext::cancel_iteration;
  using SimpleRuntimeContext::commit_iteration;
  using PerIterMem::__resetAlloc;

public:
  ParallelThreadContext()
    :conflicts(0), iterations(0), wl(0), aborted(0)
  {}
  
  virtual ~ParallelThreadContext() {}

  unsigned long getIterations() { return iterations; }
  void setWl(WorkListTy* WL) { wl = WL; }
  void setAborted(AbortedListTy* WL) { aborted = WL; }

  void report() const {
    reportStat("Conflicts", conflicts);
    reportStat("Iterations", iterations);
  }

  bool drainAborted() {
    bool retval = false;
    while (true) {
      std::pair<bool, value_type> p = aborted->pop();

      if (p.first) {
        retval = true;
        wl->push(p.second);
      } else
        break;
    }
    return retval;
  }

  template<bool is_leader, typename Function>
  bool doProcess(value_type val, Function& f) {
    ++iterations;
    if (is_leader && (iterations & 1023) == 0) {
      drainAborted();
    }
    
    start_iteration();
    try {
      f(val, *this);
    } catch (int a) {
      cancel_iteration();
      ++conflicts;
      //wl->aborted(val);
      aborted->push(val);
      return false;
    }
    commit_iteration();
    __resetAlloc();
    return true;
  }

  void push(value_type val) {
    wl->push(val);
  }

  void finish() { }
  
  void suspendWith(Galois::Executable* E) {
    abort();
  }

  static void merge(ParallelThreadContext& lhs, ParallelThreadContext& rhs) {
    lhs.conflicts += rhs.conflicts;
    lhs.iterations += rhs.iterations;
  }
};

static void summarizeList(const char* name, const std::vector<long>& list) {
  long min = *std::min_element(list.begin(), list.end());
  long max = *std::max_element(list.begin(), list.end());
  double ave = std::accumulate(list.begin(), list.end(), 0.0) / list.size();
 
  double acc = 0.0;
  for (std::vector<long>::const_iterator it = list.begin(), end = list.end(); it != end; ++it) {
    acc += (*it - ave) * (*it - ave);
  }

  double stdev = 0.0;
  if (list.size() > 1) {
    stdev = sqrt(acc / (list.size() - 1));
  }

  std::ostringstream out;
  out.setf(std::ios::fixed, std::ios::floatfield);
  out.precision(1);
  out << "n: " << list.size();
  out << " ave: " << ave;
  out << " min: " << min;
  out << " max: " << max;
  out << " stdev: " << stdev;

  reportStat(name, out.str().c_str());
}

template<class WorkListTy, class Function>
class ForEachWork : public Galois::Executable {
  typedef typename WorkListTy::value_type value_type;
  typedef GaloisRuntime::WorkList::MP_SC_FIFO<value_type> AbortedListTy;
  typedef ParallelThreadContext<WorkListTy> PCTy;

  WorkListTy& global_wl;
  Function& f;

  PerCPU<PCTy> tdata;
  TerminationDetection term;
  AbortedListTy aborted;

  template<bool is_leader>
  void runLoop(PCTy& tld) {
    setThreadContext(&tld);
    Timer T;
    T.start();
    do {
      std::pair<bool, value_type> p = global_wl.pop();
      if (p.first) {
	term.workHappened();
	tld.template doProcess<is_leader>(p.second, f);
	do {
	  p = global_wl.pop();
	  if (p.first) {
	    tld.template doProcess<is_leader>(p.second, f);
	  } else {
	    break;
	  }
	} while(true);
      }
      if (is_leader && tld.drainAborted())
        continue;

      term.localTermination();
    } while (!term.globalTermination());
    T.stop();
    setThreadContext(0);
  }

public:
  ForEachWork(WorkListTy& _wl, Function& _f)
    :global_wl(_wl), f(_f) {}
  
  ~ForEachWork() {
    {
      int numThreads = GaloisRuntime::getSystemThreadPool().getActiveThreads();
      std::vector<long> list;
      for (int i = 0; i < numThreads; ++i)
        list.push_back(tdata.get(i).getIterations());
      summarizeList("IterationDistribution", list);
    }

    for (int i = 1; i < tdata.size(); ++i)
      PCTy::merge(tdata.get(0), tdata.get(i));
    tdata.get().report();
    assert(global_wl.empty());
  }

  virtual void preRun(int tmax) { }

  virtual void postRun() {  }

  virtual void operator()() {
    PCTy& tld = tdata.get();
    tld.setWl(&global_wl);
    tld.setAborted(&aborted);
    if (tdata.myEffectiveID() == 0)
      runLoop<true>(tld);
    else
      runLoop<false>(tld);
  }
};

template<class Function>
class ForAllWork : public Galois::Executable {
  PerCPU<long> tdata;
  Function& f;
  long start, end;
  int numThreads;

public:
  ForAllWork(int _start, int _end, Function& _f) : f(_f), start(_start), end(_end) {
    numThreads = GaloisRuntime::getSystemThreadPool().getActiveThreads();
    assert(numThreads > 0);
  }
  
  ~ForAllWork() { 
    std::vector<long> list;
    for (int i = 0; i < numThreads; ++i)
      list.push_back(tdata.get(i));
    summarizeList("TotalTime", list);
  }

  virtual void preRun(int tmax) { }

  virtual void postRun() {  }

  virtual void operator()() {
    Timer T;
    T.start();
    // Simple blocked assignment
    unsigned int id = tdata.myEffectiveID();
    long range = end - start;
    long block = range / numThreads;
    long base = start + id * block;
    long stop = base + block;
    for (long i = base; i < stop; i++) {
      f(i);
    }
    // Remainder (each thread executes at most one iteration)
    for (long i = start + numThreads * block + id; i < end; i += numThreads) {
      f(i);
    }
    T.stop();
    tdata.get() = T.get();
  }
};

template<class Function, class GWLTy>
void for_each_parallel(GWLTy& GWL, Function& f) {
#ifdef GALOIS_VTUNE
  __itt_resume();
#endif
  ForEachWork<GWLTy, Function> GW(GWL, f);
  ThreadPool& PTP = getSystemThreadPool();
  PTP.run(&GW);
#ifdef GALOIS_VTUNE
  __itt_pause();
#endif
}

template<class Function>
void for_all_parallel(long start, long end, Function& f) {
  ForAllWork<Function> GW(start, end, f);
  ThreadPool& PTP = getSystemThreadPool();
  PTP.run(&GW);
}
}

#endif
