/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#ifndef GALOIS_RUNTIME_EXECUTOR_PARAMETER_H
#define GALOIS_RUNTIME_EXECUTOR_PARAMETER_H

#include "galois/gtuple.h"
#include "galois/Reduction.h"
#include "galois/PerThreadContainer.h"
#include "galois/Traits.h"
#include "galois/Mem.h"
#include "galois/worklists/Simple.h"
#include "galois/runtime/Context.h"
#include "galois/runtime/Executor_ForEach.h"
#include "galois/runtime/Executor_DoAll.h"
#include "galois/runtime/Executor_OnEach.h"
#include "galois/gIO.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <vector>
#include <random>

namespace galois {
namespace runtime {

namespace ParaMeter {


struct StepStatsBase {
  static inline void printHeader(FILE* out) {
    fprintf(out, "LOOPNAME, STEP, PARALLELISM, WORKLIST_SIZE, NEIGHBORHOOD_SIZE\n");
  }

  static inline void dump(FILE* out, const char* loopname, size_t step, size_t parallelism, size_t wlSize, size_t nhSize) {
    assert(out && "StepStatsBase::dump() file handle is null");
    fprintf(out, "%s, %zu, %zu, %zu, %zu\n", loopname, step, parallelism, wlSize, nhSize);
  }
};

struct OrderedStepStats: public StepStatsBase {
  using Base = StepStatsBase;

  const size_t step;
  GAccumulator<size_t> parallelism;
  const size_t wlSize;

  explicit OrderedStepStats (size_t _step, size_t _wlsz): Base(), step (_step), parallelism (),  wlSize (_wlsz) {}

  explicit OrderedStepStats (size_t _step, size_t par, size_t _wlsz): Base(), step (_step), parallelism (), wlSize (_wlsz) {
    parallelism += par;
  }

  void dump(FILE* out, const char* loopname) const {
    Base::dump(out, loopname, step, parallelism.reduce(), wlSize, 0ul);
  }
};


struct UnorderedStepStats: public StepStatsBase {
  using Base = StepStatsBase;

  size_t step;
  GAccumulator<size_t> parallelism;
  GAccumulator<size_t> wlSize;
  GAccumulator<size_t> nhSize;

  UnorderedStepStats (void) : Base(), step(0)
  {}

  void nextStep (void) {
    ++step;
    parallelism.reset();
    wlSize.reset();
    nhSize.reset();
  }

  void dump(FILE* out, const char* loopname) const {
    Base::dump(out, loopname, step, parallelism.reduce(), wlSize.reduce(), nhSize.reduce());
  }
};

// Single ParaMeter stats file per run of an app
// which includes all instances of for_each loops
// run with ParaMeter Executor
FILE* getStatsFile (void);
void closeStatsFile (void);


template <typename T>
class FIFO_WL {

protected:

  using PTcont = galois::PerThreadVector<T>;

  PTcont* curr;
  PTcont* next;

public:

  FIFO_WL(void):
    curr (new PTcont()),
    next (new PTcont())
  {}

  ~FIFO_WL(void) {
    delete curr; curr = nullptr;
    delete next; next = nullptr;
  }

  auto iterateCurr(void) {
    return galois::runtime::makeLocalRange(*curr);
  }

  void pushNext(const T& item) {
    next->get().push_back(item);
  }

  void nextStep(void) {
    std::swap(curr, next);
    next->clear_all_parallel();
  }

  bool empty(void) const {
    return next->empty_all();
  }
};

template <typename T>
class RAND_WL: public FIFO_WL<T> {
  using Base = FIFO_WL<T>;

public:

  auto iterateCurr(void) {
    galois::runtime::on_each_gen(
        [&] (int tid, int numT) {
          auto& lwl = Base::curr->get();

          std::random_device r;
          std::mt19937 rng(r());
          std::shuffle(lwl.begin(), lwl.end(), rng);

        }, std::make_tuple());

    return galois::runtime::makeLocalRange(*Base::curr);
  }
};

template <typename T>
class LIFO_WL: public FIFO_WL<T> {
  using Base = FIFO_WL<T>;

public:

  auto iterateCurr(void) {

    // TODO: use reverse iterator instead of std::reverse
    galois::runtime::on_each_gen(
        [&] (int tid, int numT) {
          auto& lwl = Base::curr->get();
          std::reverse(lwl.begin(), lwl.end());

        }, std::make_tuple());

    return galois::runtime::makeLocalRange(*Base::curr);
  }
};

enum class SchedType {
  FIFO, RAND, LIFO
};

template <typename T, SchedType SCHED>
struct ChooseWL {};

template <typename T> struct ChooseWL<T, SchedType::FIFO> {
  using type = FIFO_WL<T>;
};

template <typename T> struct ChooseWL<T, SchedType::LIFO> {
  using type = LIFO_WL<T>;
};

template <typename T> struct ChooseWL<T, SchedType::RAND> {
  using type = RAND_WL<T>;
};


template<class T, class FunctionTy, class ArgsTy>
class ParaMeterExecutor {

  using value_type = T;
  using GenericWL = typename get_type_by_supertype<wl_tag, ArgsTy>::type::type;
  using WorkListTy = typename GenericWL::template retype<T>;
  using dbg = galois::debug<1>;

  static const bool needsStats = !exists_by_supertype<no_stats_tag, ArgsTy>::value;
  static const bool needsPush = !exists_by_supertype<no_pushes_tag, ArgsTy>::value;
  static const bool needsAborts = !exists_by_supertype<no_conflicts_tag, ArgsTy>::value;
  static const bool needsPia = exists_by_supertype<per_iter_alloc_tag, ArgsTy>::value;
  static const bool needsBreak = exists_by_supertype<parallel_break_tag, ArgsTy>::value;


  struct IterationContext {
    T item;
    bool doabort;
    galois::runtime::UserContextAccess<value_type> facing;
    SimpleRuntimeContext ctx;

    explicit IterationContext(const T& v): item(v), doabort(false) {}

    void reset() {
      doabort = false;
      if (needsPia)
        facing.resetAlloc();

      if (needsPush)
        facing.getPushBuffer().clear();
    }
  };

  using PWL = typename ChooseWL<IterationContext*, WorkListTy::SCHEDULE>::type;


private:
  PWL m_wl;
  FunctionTy m_func;
  const char* loopname;
  FILE* m_statsFile;
  FixedSizeAllocator<IterationContext> m_iterAlloc;
  galois::GReduceLogicalOR m_broken;

  IterationContext* newIteration(const T& item) {
    IterationContext* it = m_iterAlloc.allocate(1);
    assert(it && "IterationContext allocation failed");

    m_iterAlloc.construct(it, item);

    it->reset();
    return it;
  }

  unsigned abortIteration(IterationContext* it) {
    assert(it && "nullptr arg");
    assert(it->doabort && "aborting an iteration without setting its doabort flag");

    unsigned numLocks = it->ctx.cancelIteration();
    it->reset();

    m_wl.pushNext(it);
    return numLocks;
  }

  unsigned commitIteration(IterationContext* it) {
    assert(it && "nullptr arg");

    if (needsPush) {
      for (const auto& item: it->facing.getPushBuffer()) {
        IterationContext* child = newIteration(item);
        m_wl.pushNext(child);
      }
    }

    unsigned numLocks = it->ctx.commitIteration();
    it->reset();

    m_iterAlloc.destroy(it);
    m_iterAlloc.deallocate(it, 1);

    return numLocks;
  }

private:

  void runSimpleStep(UnorderedStepStats& stats) {
    galois::runtime::do_all_gen(m_wl.iterateCurr(),
        [&, this] (IterationContext* it) {

          stats.wlSize += 1;

          setThreadContext(&(it->ctx));

          m_func(it->item, it->facing.data ());
          stats.parallelism += 1;
          unsigned nh = commitIteration(it);
          stats.nhSize += nh;

          setThreadContext(nullptr);
          
        },
        std::make_tuple(
          galois::steal(),
          galois::loopname("ParaM-Simple")));

  }

  void runCautiousStep(UnorderedStepStats& stats) {
    galois::runtime::do_all_gen(m_wl.iterateCurr(),
        [&, this] (IterationContext* it) {

          stats.wlSize += 1;

          setThreadContext(&(it->ctx));
          bool broke = false;

          if (needsBreak) {
            it->facing.setBreakFlag(&broke);
          }

#ifdef GALOIS_USE_LONGJMP_ABORT
          int flag = 0;
          if ((flag = setjmp(execFrame)) == 0) {
            m_func(it->item, it->facing.data ());

          } else {
#else 
          try {
            m_func(it->item, it->facing.data ());

          } catch (const ConflictFlag& flag) {
#endif
            clearConflictLock();
            switch (flag) {
              case galois::runtime::CONFLICT:
                it->doabort = true;
                break;
              default:
                std::abort ();
            }
          }

          if (needsBreak && broke) {
            m_broken.update(true);
          }

          setThreadContext(nullptr);
          
        },
        std::make_tuple(
          galois::steal(),
          galois::loopname("ParaM-Expand-NH")));


    galois::runtime::do_all_gen(m_wl.iterateCurr(), 
        [&, this] (IterationContext* it) {

          if (it->doabort) {
            abortIteration(it);

          } else {
            stats.parallelism += 1;
            unsigned nh = commitIteration(it);
            stats.nhSize += nh;

          }
        },
        std::make_tuple(
          galois::steal(),
          galois::loopname("ParaM-Commit")));
  }

  template <typename R>
  void execute(const R& range) {


    galois::runtime::on_each_gen([&, this] (const unsigned tid, const unsigned numT) {
          auto p = range.local_pair();

          for (auto i = p.first; i != p.second; ++i) {
            IterationContext* it = newIteration(*i);
            m_wl.pushNext(it);
          }
          
        }, std::make_tuple());

    UnorderedStepStats stats;

    while (!m_wl.empty()) {

      m_wl.nextStep();

      if (needsAborts) {
        runCautiousStep(stats);

      } else {
        runSimpleStep(stats);
      }

      // dbg::print("Step: ", stats.step, ", Parallelism: ", stats.parallelism.reduce());
      assert(stats.parallelism.reduce() && "ERROR: No Progress");

      stats.dump(m_statsFile, loopname);
      stats.nextStep();

      if (needsBreak && m_broken.reduce()) {
        break;
      }

          
    } // end while

    closeStatsFile();
  }
public:

  ParaMeterExecutor(const FunctionTy& f, const ArgsTy& args): 
    m_func(f),
    loopname(galois::internal::getLoopName(args)),
    m_statsFile(getStatsFile())
  {}

  // called serially once
  template<typename RangeTy>
  void init(const RangeTy& range) {
    execute(range);
  }

  // called once on each thread followed by a barrier
  template<typename RangeTy>
  void initThread(const RangeTy& range) const {}

  void operator() (void) {}

};


} // end namespace ParaMeter
} // end namespace runtime

namespace worklists {

template<class T=int, runtime::ParaMeter::SchedType SCHED=runtime::ParaMeter::SchedType::FIFO>
class ParaMeter {
public:
  template<bool _concurrent>
  using rethread = ParaMeter<T, SCHED>;

  template<typename _T>
  using retype = ParaMeter<_T, SCHED>;

  using value_type = T;

  constexpr static const runtime::ParaMeter::SchedType SCHEDULE = SCHED;

  using fifo = ParaMeter<T, runtime::ParaMeter::SchedType::FIFO>;
  using random = ParaMeter<T, runtime::ParaMeter::SchedType::RAND>;
  using lifo = ParaMeter<T, runtime::ParaMeter::SchedType::LIFO>;

};

} // end worklists

namespace runtime {

  // hookup into galois::for_each. Invoke galois::for_each with wl<galois::worklists::ParaMeter<> >
template<class T, class FunctionTy, class ArgsTy>
struct ForEachExecutor<galois::worklists::ParaMeter<T>, FunctionTy, ArgsTy>:
  public ParaMeter::ParaMeterExecutor<T, FunctionTy, ArgsTy>
{
  using SuperTy = ParaMeter::ParaMeterExecutor<T, FunctionTy, ArgsTy>;
  ForEachExecutor(const FunctionTy& f, const ArgsTy& args): SuperTy(f, args) { }
};


//! invoke ParaMeter tool to execute a for_each style loop
template<typename R, typename F, typename ArgsTuple>
void for_each_ParaMeter (const R& range, const F& func, const ArgsTuple& argsTuple) {

  using T = typename R::values_type;

  auto tpl = galois::get_default_trait_values(argsTuple,
      std::make_tuple (wl_tag {}),
      std::make_tuple (wl<galois::worklists::ParaMeter<> >()));

  using Tpl_ty = decltype (tpl);

  using Exec = runtime::ParaMeter::ParaMeterExecutor<T, F, Tpl_ty>;
  Exec exec (func, tpl);

  exec.execute (range);

}



} // end namespace runtime
} // end namespace galois
#endif

/*
 * requirements: 
 * - support random and fifo schedules, maybe lifo
 * - write stats to a single file. 
 * - support multi-threaded execution
 *
 * interface:
 * - file set by environment variable
 * - ParaMeter invoked by choosing wl type, e.g. ParaMeter<>::with_rand, or ParaMeter<>::fifo
 */
