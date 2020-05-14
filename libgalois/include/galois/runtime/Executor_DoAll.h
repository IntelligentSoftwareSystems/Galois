/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

#ifndef GALOIS_RUNTIME_EXECUTOR_DOALL_H
#define GALOIS_RUNTIME_EXECUTOR_DOALL_H

#include "galois/config.h"
#include "galois/gIO.h"
#include "galois/runtime/Executor_OnEach.h"
#include "galois/runtime/OperatorReferenceTypes.h"
#include "galois/runtime/Statistics.h"
#include "galois/substrate/Barrier.h"
#include "galois/substrate/CompilerSpecific.h"
#include "galois/substrate/PaddedLock.h"
#include "galois/substrate/PerThreadStorage.h"
#include "galois/substrate/Termination.h"
#include "galois/substrate/ThreadPool.h"
#include "galois/Timer.h"

namespace galois::runtime {

namespace internal {

template <typename R, typename F, typename ArgsTuple>
class DoAllStealingExec {

  typedef typename R::local_iterator Iter;
  typedef typename std::iterator_traits<Iter>::difference_type Diff_ty;

  enum StealAmt { HALF, FULL };

  constexpr static const bool NEED_STATS =
      galois::internal::NeedStats<ArgsTuple>::value;
  constexpr static const bool MORE_STATS =
      NEED_STATS && has_trait<more_stats_tag, ArgsTuple>();
  constexpr static const bool USE_TERM = false;

  struct ThreadContext {

    alignas(substrate::GALOIS_CACHE_LINE_SIZE) substrate::SimpleLock work_mutex;
    unsigned id;

    Iter shared_beg;
    Iter shared_end;
    Diff_ty m_size;
    size_t num_iter;

    // Stats

    ThreadContext()
        : work_mutex(), id(substrate::getThreadPool().getMaxThreads()),
          shared_beg(), shared_end(), m_size(0), num_iter(0) {
      // TODO: fix this initialization problem,
      // see initThread
    }

    ThreadContext(unsigned id, Iter beg, Iter end)
        : work_mutex(), id(id), shared_beg(beg), shared_end(end),
          m_size(std::distance(beg, end)), num_iter(0) {}

    bool doWork(F func, const unsigned chunk_size) {
      Iter beg(shared_beg);
      Iter end(shared_end);

      bool didwork = false;

      while (getWork(beg, end, chunk_size)) {

        didwork = true;

        for (; beg != end; ++beg) {
          if (NEED_STATS) {
            ++num_iter;
          }
          func(*beg);
        }
      }

      return didwork;
    }

    bool hasWorkWeak() const { return (m_size > 0); }

    bool hasWork() const {
      bool ret = false;

      work_mutex.lock();
      {
        ret = hasWorkWeak();

        if (m_size > 0) {
          assert(shared_beg != shared_end);
        }
      }
      work_mutex.unlock();

      return ret;
    }

  private:
    bool getWork(Iter& priv_beg, Iter& priv_end, const unsigned chunk_size) {
      bool succ = false;

      work_mutex.lock();
      {
        if (hasWorkWeak()) {
          succ = true;

          Iter nbeg = shared_beg;
          if (m_size <= chunk_size) {
            nbeg   = shared_end;
            m_size = 0;

          } else {
            std::advance(nbeg, chunk_size);
            m_size -= chunk_size;
            assert(m_size > 0);
          }

          priv_beg   = shared_beg;
          priv_end   = nbeg;
          shared_beg = nbeg;
        }
      }
      work_mutex.unlock();

      return succ;
    }

    void steal_from_end_impl(Iter& steal_beg, Iter& steal_end, const Diff_ty sz,
                             std::forward_iterator_tag) {

      // steal from front for forward_iterator_tag
      steal_beg = shared_beg;
      std::advance(shared_beg, sz);
      steal_end = shared_beg;
    }

    void steal_from_end_impl(Iter& steal_beg, Iter& steal_end, const Diff_ty sz,
                             std::bidirectional_iterator_tag) {

      steal_end = shared_end;
      std::advance(shared_end, -sz);
      steal_beg = shared_end;
    }

    void steal_from_end(Iter& steal_beg, Iter& steal_end, const Diff_ty sz) {
      assert(sz > 0);
      steal_from_end_impl(
          steal_beg, steal_end, sz,
          typename std::iterator_traits<Iter>::iterator_category());
    }

    void steal_from_beg(Iter& steal_beg, Iter& steal_end, const Diff_ty sz) {
      assert(sz > 0);
      steal_beg = shared_beg;
      std::advance(shared_beg, sz);
      steal_end = shared_beg;
    }

  public:
    bool stealWork(Iter& steal_beg, Iter& steal_end, Diff_ty& steal_size,
                   StealAmt amount, size_t chunk_size) {
      bool succ = false;

      if (work_mutex.try_lock()) {

        if (hasWorkWeak()) {
          succ = true;

          if (amount == HALF && m_size > (Diff_ty)chunk_size) {
            steal_size = m_size / 2;
          } else {
            steal_size = m_size;
          }

          if (m_size <= steal_size) {
            steal_beg = shared_beg;
            steal_end = shared_end;

            shared_beg = shared_end;

            steal_size = m_size;
            m_size     = 0;

          } else {

            // steal_from_end (steal_beg, steal_end, steal_size);
            steal_from_beg(steal_beg, steal_end, steal_size);
            m_size -= steal_size;
          }
        }

        work_mutex.unlock();
      }

      return succ;
    }

    void assignWork(const Iter& beg, const Iter& end, const Diff_ty sz) {
      work_mutex.lock();
      {
        assert(!hasWorkWeak());
        assert(beg != end);
        assert(std::distance(beg, end) == sz);

        shared_beg = beg;
        shared_end = end;
        m_size     = sz;
      }
      work_mutex.unlock();
    }
  };

private:
  GALOIS_ATTRIBUTE_NOINLINE bool
  transferWork(ThreadContext& rich, ThreadContext& poor, StealAmt amount) {

    assert(rich.id != poor.id);
    assert(rich.id < galois::getActiveThreads());
    assert(poor.id < galois::getActiveThreads());

    Iter steal_beg;
    Iter steal_end;

    // stealWork should initialize to a more appropriate value
    Diff_ty steal_size = 0;

    bool succ =
        rich.stealWork(steal_beg, steal_end, steal_size, amount, chunk_size);

    if (succ) {
      assert(steal_beg != steal_end);
      assert(std::distance(steal_beg, steal_end) == steal_size);

      poor.assignWork(steal_beg, steal_end, steal_size);
    }

    return succ;
  }

  GALOIS_ATTRIBUTE_NOINLINE bool stealWithinSocket(ThreadContext& poor) {

    bool sawWork   = false;
    bool stoleWork = false;

    auto& tp = substrate::getThreadPool();

    const unsigned maxT     = galois::getActiveThreads();
    const unsigned my_pack  = substrate::ThreadPool::getSocket();
    const unsigned per_pack = tp.getMaxThreads() / tp.getMaxSockets();

    const unsigned pack_beg = my_pack * per_pack;
    const unsigned pack_end = (my_pack + 1) * per_pack;

    for (unsigned i = 1; i < pack_end; ++i) {

      // go around the socket in circle starting from the next thread
      unsigned t = (poor.id + i) % per_pack + pack_beg;
      assert((t >= pack_beg) && (t < pack_end));

      if (t < maxT) {
        if (workers.getRemote(t)->hasWorkWeak()) {
          sawWork = true;

          stoleWork = transferWork(*workers.getRemote(t), poor, HALF);

          if (stoleWork) {
            break;
          }
        }
      }
    }

    return sawWork || stoleWork;
  }

  GALOIS_ATTRIBUTE_NOINLINE bool stealOutsideSocket(ThreadContext& poor,
                                                    const StealAmt& amt) {
    bool sawWork   = false;
    bool stoleWork = false;

    auto& tp       = substrate::getThreadPool();
    unsigned myPkg = substrate::ThreadPool::getSocket();
    // unsigned maxT = LL::getMaxThreads ();
    unsigned maxT = galois::getActiveThreads();

    for (unsigned i = 0; i < maxT; ++i) {
      ThreadContext& rich = *(workers.getRemote((poor.id + i) % maxT));

      if (tp.getSocket(rich.id) != myPkg) {
        if (rich.hasWorkWeak()) {
          sawWork = true;

          stoleWork = transferWork(rich, poor, amt);
          // stoleWork = transferWork (rich, poor, HALF);

          if (stoleWork) {
            break;
          }
        }
      }
    }

    return sawWork || stoleWork;
  }

  GALOIS_ATTRIBUTE_NOINLINE bool trySteal(ThreadContext& poor) {
    bool ret = false;

    ret = stealWithinSocket(poor);

    if (ret) {
      return true;
    }

    substrate::asmPause();

    if (substrate::getThreadPool().isLeader(poor.id)) {
      ret = stealOutsideSocket(poor, HALF);

      if (ret) {
        return true;
      }
      substrate::asmPause();
    }

    ret = stealOutsideSocket(poor, HALF);
    if (ret) {
      return true;
    }
    substrate::asmPause();

    return ret;
  }

private:
  R range;
  F func;
  const char* loopname;
  Diff_ty chunk_size;
  substrate::PerThreadStorage<ThreadContext> workers;

  substrate::TerminationDetection& term;

  // for stats
  PerThreadTimer<MORE_STATS> totalTime;
  PerThreadTimer<MORE_STATS> initTime;
  PerThreadTimer<MORE_STATS> execTime;
  PerThreadTimer<MORE_STATS> stealTime;
  PerThreadTimer<MORE_STATS> termTime;

public:
  DoAllStealingExec(const R& _range, F _func, const ArgsTuple& argsTuple)
      : range(_range), func(_func),
        loopname(galois::internal::getLoopName(argsTuple)),
        chunk_size(get_trait_value<chunk_size_tag>(argsTuple).value),
        term(substrate::getSystemTermination(activeThreads)),
        totalTime(loopname, "Total"), initTime(loopname, "Init"),
        execTime(loopname, "Execute"), stealTime(loopname, "Steal"),
        termTime(loopname, "Term") {
    assert(chunk_size > 0);
  }

  // parallel call
  void initThread(void) {
    initTime.start();

    term.initializeThread();

    unsigned id = substrate::ThreadPool::getTID();

    *workers.getLocal(id) =
        ThreadContext(id, range.local_begin(), range.local_end());

    initTime.stop();
  }

  ~DoAllStealingExec() {
// executed serially
#ifndef NDEBUG
    for (unsigned i = 0; i < workers.size(); ++i) {
      auto& ctx = *(workers.getRemote(i));
      assert(!ctx.hasWork() && "Unprocessed work left");
    }
#endif

    // printStats ();
  }

  void operator()(void) {

    ThreadContext& ctx = *workers.getLocal();
    totalTime.start();

    while (true) {
      bool workHappened = false;

      execTime.start();

      if (ctx.doWork(func, chunk_size)) {
        workHappened = true;
      }

      execTime.stop();

      assert(!ctx.hasWork());

      stealTime.start();
      bool stole = trySteal(ctx);
      stealTime.stop();

      if (stole) {
        continue;

      } else {

        assert(!ctx.hasWork());
        if (USE_TERM) {
          termTime.start();
          term.localTermination(workHappened);

          bool quit = term.globalTermination();
          termTime.stop();

          if (quit) {
            break;
          }
        } else {
          break;
        }
      }
    }

    totalTime.stop();
    assert(!ctx.hasWork());

    if (NEED_STATS) {
      galois::runtime::reportStat_Tsum(loopname, "Iterations", ctx.num_iter);
    }
  }
};

template <bool _STEAL>
struct ChooseDoAllImpl {

  template <typename R, typename F, typename ArgsT>
  static void call(const R& range, F&& func, const ArgsT& argsTuple) {

    internal::DoAllStealingExec<
        R, OperatorReferenceType<decltype(std::forward<F>(func))>, ArgsT>
        exec(range, std::forward<F>(func), argsTuple);

    substrate::Barrier& barrier = getBarrier(activeThreads);

    substrate::getThreadPool().run(
        activeThreads, [&exec](void) { exec.initThread(); }, std::ref(barrier),
        std::ref(exec));
  }
};

template <>
struct ChooseDoAllImpl<false> {

  template <typename R, typename F, typename ArgsT>
  static void call(const R& range, F func, const ArgsT& argsTuple) {

    runtime::on_each_gen(
        [&](const unsigned int, const unsigned int) {
          static constexpr bool NEED_STATS =
              galois::internal::NeedStats<ArgsT>::value;
          static constexpr bool MORE_STATS =
              NEED_STATS && has_trait<more_stats_tag, ArgsT>();

          const char* const loopname = galois::internal::getLoopName(argsTuple);

          PerThreadTimer<MORE_STATS> totalTime(loopname, "Total");
          PerThreadTimer<MORE_STATS> initTime(loopname, "Init");
          PerThreadTimer<MORE_STATS> execTime(loopname, "Work");

          totalTime.start();
          initTime.start();

          auto begin     = range.local_begin();
          const auto end = range.local_end();

          initTime.stop();

          execTime.start();

          size_t iter = 0;

          while (begin != end) {
            func(*begin++);
            if (NEED_STATS) {
              ++iter;
            }
          }
          execTime.stop();

          totalTime.stop();

          if (NEED_STATS) {
            galois::runtime::reportStat_Tsum(loopname, "Iterations", iter);
          }
        },
        std::make_tuple());
  }
};

} // end namespace internal

template <typename R, typename F, typename ArgsTuple>
void do_all_gen(const R& range, F&& func, const ArgsTuple& argsTuple) {

  static_assert(!has_trait<char*, ArgsTuple>(), "old loopname");
  static_assert(!has_trait<char const*, ArgsTuple>(), "old loopname");
  static_assert(!has_trait<bool, ArgsTuple>(), "old steal");

  auto argsT = std::tuple_cat(
      argsTuple,
      get_default_trait_values(argsTuple, std::make_tuple(chunk_size_tag{}),
                               std::make_tuple(chunk_size<>{})));

  using ArgsT = decltype(argsT);

  constexpr bool TIME_IT = has_trait<loopname_tag, ArgsT>();
  CondStatTimer<TIME_IT> timer(galois::internal::getLoopName(argsT));

  timer.start();

  constexpr bool STEAL = has_trait<steal_tag, ArgsT>();

  OperatorReferenceType<decltype(std::forward<F>(func))> func_ref = func;
  internal::ChooseDoAllImpl<STEAL>::call(range, func_ref, argsT);

  timer.stop();
}

} // namespace galois::runtime

#endif
