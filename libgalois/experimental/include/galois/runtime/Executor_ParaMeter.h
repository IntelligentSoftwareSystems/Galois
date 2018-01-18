/** ParaMeter runtime -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * Implementation of ParaMeter runtime.  Ordered with speculation not
 * supported yet
 *
 * @author Amber Hassaan <ahassaan@ices.utexas.edu>
 */
#ifndef GALOIS_RUNTIME_EXECUTOR_PARAMETER_H
#define GALOIS_RUNTIME_EXECUTOR_PARAMETER_H

#include "galois/gtuple.h"
#include "galois/Reduction.h"
#include "galois/Traits.h"
#include "galois/Mem.h"
#include "galois/runtime/Context.h"
#include "galois/runtime/Executor_ForEach.h"
#include "galois/runtime/Support.h"
#include "galois/gIO.h"
#include "galois/worklists/Simple.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <vector>

#include "llvm/Support/CommandLine.h"

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
    Base::dump(out, loopname, step, parallelism.reduceRO (), wlSize, 0ul);
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


template <typename T, typename PTcont=galois::PerThreadBag<T> >
class ParaMeterFIFO_WL {

protected:

  PTcont* curr;
  PTcont* next;

public:

  ParaMeterFIFO_WL(void):
    curr (new PTcont()),
    next (new PTcont())
  {}

  ~ParaMeterFIFO_WL(void) {
    delete curr; curr = nullptr;
    delete next; next = nullptr;
  }

  auto iterateCurr(void) {
    return galois::runtime::makeLocalRange(*curr);
  }

  void pushNext(const T& item) {
    next->push_back(item);
  }

  void nextStep(void) {
    std::swap(curr, next);
    next->clear_all_parallel();
  }
};

template <typename T>
class ParaMeterRAND_WL: public ParaMeterFIFO_WL<T, galois::PerThreadVector<T> > {
  using Base = ParaMeterFIFO_WL<T, galois::PerThreadVector<T> >;

public:

  void pushNext(void) {
  }
};



template<class T, class FunctionTy, class ArgsTy>
class ParaMeterExecutor {
  typedef T value_type;
  typedef typename worklists::GFIFO<int>::template retype<value_type>::type WorkListTy;

  static const bool needsStats = !exists_by_supertype<does_not_need_stats_tag, ArgsTy>::value;
  static const bool needsPush = !exists_by_supertype<does_not_need_push_tag, ArgsTy>::value;
  static const bool needsAborts = !exists_by_supertype<does_not_need_aborts_tag, ArgsTy>::value;
  static const bool needsPia = exists_by_supertype<needs_per_iter_alloc_tag, ArgsTy>::value;
  static const bool needsBreak = exists_by_supertype<needs_parallel_break_tag, ArgsTy>::value;

  struct IterationContext {
    T val;
    galois::runtime::UserContextAccess<value_type> facing;
    SimpleRuntimeContext ctx;

    explicit IterationContext(const T& v): val(v) {}

    void reset() {
      if (needsPia)
        facing.resetAlloc();

      if (needsPush)
        facing.getPushBuffer().clear();
    }
  };



  IterationContext* newIteration(const T& val) {
    IterationContext* it = m_iterAlloc.allocate(1);
    assert(it && "IterationContext allocation failed");

    m_iterAlloc.construct(it, val);

    it->reset();
    return it;
  }

  unsigned abortIteration(IterationContext* it) {
    clearConflictLock();
    return it->ctx.cancelIteration();
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
    m_iterAlloc.deallocate(it);

    return numLocks;
  }

public:
  static_assert(!needsBreak, "not supported by this executor");

  ParaMeterExecutor(FunctionTy& f, const ArgsTy& args): 
    m_func(f),
    loopname(get_by_supertype<loopname_tag>(args).value),
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

private:
  template <typename R>
  void execute(const R& range) {


    galois::on_each([&, this] (const unsigned tid, const unsigned numT) {
          auto p = range.local_pair();

          for (auto i = p.first; i != p.second; ++i) {
            m_wl.pushNext(*i);
          }
          
        });

    UnorderedStepStats stats;

    while (!m_wl.empty()) {

      m_wl.nextStep();

      galois::do_all(m_wl.iterateCurr(),
          [&, this] (IterationContext* it) {

            stats.wlSize += 1;

            setThreadContext(&(it->ctx));

            bool doabort = false;

            try {
              m_func(it->item, it->facing.data ());

            } catch (ConflictFlag flag) {
              clearConflictLock();
              switch (flag) {
                case galois::runtime::CONFLICT:
                  it->doabort = true;
                  break;
                case galois::runtime::BREAK:
                  GALOIS_DIE("can't handle breaks yet");
                  break;
                default:

                  std::abort ();
              }
            }

            if (doabort) {
              it->doabort = true;
            }

            setThreadContext(nullptr);
            
          },
          galois::steal(),
          galois::loopname("ParaM-Expand-NH"));


      galois::do_all(m_wl.iterateCurr(), 
          [&, this] (IterationContext* it) {

            if (it->doabort) {
              abortIteration(it);
              m_wl.pushNext(it);

            } else {
              stats.parallelism += 1;
              unsigned nh = commitIteration(it);
              stats.nhSize += nh;

            }
          },
          galois::steal(),
          galois::loopname("ParaM-Commit"));

      stat.dump(m_statsFile, loopname);
      stats.nextStep();
          
    } // end while

    closeStatsFile();
  }

private:
  WorkList m_wl;
  FunctionTy m_func;
  const char* loopname;
  FILE* m_statsFile;
  FixedSizeAllocator<IterationContext> m_iterAlloc;
};


enum class SchedType {
  FIFO, RAND, LIFO;
};


} // end namespace ParaMeter
} // end namespace runtime

namespace worklists {

template<class T=int, ParaMeter::SchedType SCHED=ParaMeter::SchedType::FIFO>
class ParaMeter {
public:
  template<bool _concurrent>
  struct rethread { typedef ParaMeter<T, SCHED> type; };

  template<typename _T>
  struct retype { typedef ParaMeter<_T, SCHED> type; };

  typedef T value_type;


  using with_fifo = ParaMeter<T, ParaMeter::SchedType::FIFO>;
  using with_random = ParaMeter<T, ParaMeter::SchedType::RAND>;
  using with_lifo = ParaMeter<T, ParaMeter::SchedType::LIFO>;

};

} // end worklists

namespace runtime {

  // hookup into galois::for_each. Invoke galois::for_each with wl<galois::worklists::ParaMeter<> >
template<class T, class FunctionTy, class ArgsTy>
class ForEachExecutor<galois::worklists::ParaMeter<T>, FunctionTy, ArgsTy>:
  public ParaMeter::ParaMeterExecutor<T, FunctionTy, ArgsTy>
{
  typedef ParaMeter::ParaMeterExecutor<T, FunctionTy, ArgsTy> SuperTy;
  ForEachExecutor(const FunctionTy& f, const ArgsTy& args): SuperTy(f, args) { }
};


//! invoke ParaMeter tool to execute a for_each style loop
template<typename R, typename F, typename ArgsTuple>
void for_each_ParaMeter (const R& range, const F& func, const ArgsTuple& argsTuple) {

  using T = typename R::values_type;

  auto tpl = galois::get_default_trait_values(argsTuple,
      std::make_tuple (wl_tag {}, loopname_tag {}),
      std::make_tuple (wl<ParaMeter<> > {}, default_loopname {}));

  using Tpl_ty = decltype (tpl);
  using WL = typename get_by_supertype<wl_tag, Tpl_ty>::type::template retype<T>::type;

  using Exec = ParaMeterExecutor<T, F, Tpl_ty>;
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
 * - ParaMeter invoked by choosing wl type, e.g. ParaMeter<>::with_rand, or ParaMeter<>::with_fifo
 */
