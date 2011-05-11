// simple galois scheduler and runtime -*- C++ -*-
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
  unsigned long conflicts;
  unsigned long iterations;
  unsigned long TotalTime;
  WorkListTy* wl;

  typedef typename WorkListTy::value_type value_type;
  
  using SimpleRuntimeContext::start_iteration;
  using SimpleRuntimeContext::cancel_iteration;
  using SimpleRuntimeContext::commit_iteration;
  using PerIterMem::__resetAlloc;

public:
  ParallelThreadContext()
    :conflicts(0), iterations(0), TotalTime(0), wl(0)
  {}
  
  virtual ~ParallelThreadContext() {}

  unsigned long getTotalTime() { return TotalTime; }
  void setTotalTime(unsigned long L) { TotalTime = L; }
  void set_wl(WorkListTy* WL) { wl = WL; }

  void report() const {
    reportStat("Conflicts", conflicts);
    reportStat("Iterations", iterations);
    //reportStat("TotalTime", TotalTime);
  }

  template<typename Function>
  bool doProcess(value_type val, Function& f) {
    ++iterations;
    start_iteration();
    try {
      f(val, *this);
    } catch (int a) {
      cancel_iteration();
      ++conflicts;
      wl->aborted(val);
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
    lhs.TotalTime += rhs.TotalTime;
  }
  
};

static void summarizeTimes(const std::vector<long>& times) {
  long min = *std::min_element(times.begin(), times.end());
  long max = *std::max_element(times.begin(), times.end());
  double ave = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
 
  double acc = 0.0;
  for (std::vector<long>::const_iterator it = times.begin(), end = times.end(); it != end; ++it) {
    acc += (*it - ave) * (*it - ave);
  }

  double stdev = 0.0;
  if (times.size() > 1) {
    stdev = sqrt(acc / (times.size() - 1));
  }

  std::ostringstream out;
  out << "n: " << times.size();
  out << " ave: " << ave;
  out << " min: " << min;
  out << " max: " << max;
  out << " stdev: " << stdev;

  reportStat("TotalTime", out.str().c_str());
}

template<class WorkListTy, class Function>
class ForEachWork : public Galois::Executable {
  typedef typename WorkListTy::value_type value_type;
  typedef ParallelThreadContext<WorkListTy> PCTy;

  WorkListTy& global_wl;
  Function& f;

  PerCPU<PCTy> tdata;
  TerminationDetection term;

public:
  ForEachWork(WorkListTy& _wl, Function& _f)
    :global_wl(_wl), f(_f) {}
  
  ~ForEachWork() {
    {
      int numThreads = GaloisRuntime::getSystemThreadPool().getActiveThreads();
      std::vector<long> times;
      for (int i = 0; i < numThreads; ++i)
        times.push_back(tdata.get(i).getTotalTime());
      summarizeTimes(times);
    }

    for (int i = 1; i < tdata.size(); ++i)
      PCTy::merge(tdata.get(0), tdata.get(i));
    tdata.get().report();
    assert(global_wl.empty());
  }

  virtual void preRun(int tmax) {
  }

  virtual void postRun() {  }

  virtual void operator()() {
    PCTy& tld = tdata.get();
    setThreadContext(&tld);
    tld.set_wl(&global_wl);
    Timer T;
    T.start();
    do {
      std::pair<bool, value_type> p = global_wl.pop();
      if (p.first) {
	term.workHappened();
	tld.doProcess(p.second, f);
	do {
	  p = global_wl.pop();
	  if (p.first) {
	    tld.doProcess(p.second, f);
	  } else {
	    break;
	  }
	} while(true);
      }
      term.localTermination();
    } while (!term.globalTermination());
    T.stop();
    tld.setTotalTime(T.get());
    setThreadContext(0);
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
    std::vector<long> times;
    for (int i = 0; i < numThreads; ++i)
      times.push_back(tdata.get(i));
    summarizeTimes(times);
  }

  virtual void preRun(int tmax) { }

  virtual void postRun() {  }

  virtual void operator()() {
    Timer T;
    T.start();
    // Simple blocked assignment
    unsigned int id = ThreadPool::getMyID() - 1;
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
