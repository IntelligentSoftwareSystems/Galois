/** Do All-*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
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
 * works with Per Thread Worklists
 *
 * @author <ahassaan@ices.utexas.edu>
 */

#ifndef GALOIS_RUNTIME_DOALLCOUPLED_H
#define GALOIS_RUNTIME_DOALLCOUPLED_H

#include <algorithm>
#include <vector>
#include <limits>

#include <cstdio>
#include <ctime>


#include "Galois/Substrate/Barrier.h"
#include "Galois/Substrate/PerThreadStorage.h"
#include "Galois/Runtime/Support.h"
#include "Galois/Substrate/Termination.h"
#include "Galois/Substrate/ThreadPool.h"
#include "Galois/Substrate/PaddedLock.h"
#include "Galois/Substrate/CompilerSpecific.h"
#include "Galois/Substrate/gio.h"

#include "Galois/Timer.h"
#include "Galois/OrderedTraits.h"


namespace Galois {

  template <bool enabled> 
  class ThreadTimer {
    timespec m_start;
    timespec m_stop;
    int64_t  m_nsec;

  public:
    ThreadTimer (): m_nsec (0) {};

    void start (void) {
      clock_gettime (CLOCK_THREAD_CPUTIME_ID, &m_start);
    }

    void stop (void) {
      clock_gettime (CLOCK_THREAD_CPUTIME_ID, &m_stop);
      m_nsec += (m_stop.tv_nsec - m_start.tv_nsec);
      m_nsec += ((m_stop.tv_sec - m_start.tv_sec) << 30); // multiply by 1G
    }

    int64_t get_nsec(void) const { return m_nsec; }

    int64_t get_sec(void) const { return (m_nsec >> 30); }
      
  };

  template <>
  class ThreadTimer<false> {
  public:
    void start (void) const  {}
    void stop (void) const  {}
    int64_t get_nsec (void) const { return 0; }
    int64_t get_sec (void) const  { return 0; }
  };

  template <typename T>
  class AggStatistic {

    const char* m_name;
    std::vector<T> m_values;

    T m_min;
    T m_max;
    T m_sum;

  public:

    AggStatistic (const char* name=NULL) : 
      m_name (name),
      m_values (),
      m_min (std::numeric_limits<T>::max ()),
      m_max (std::numeric_limits<T>::min ()),
      m_sum ()
    
    {
      if (name == NULL) {
        m_name = "STAT";
      }
    }

    void add (const T& val) {
      m_values.push_back (val);

      m_min = std::min (m_min, val);
      m_max = std::max (m_max, val);

      m_sum += val;
    }

    T range () const { return m_max - m_min; }

    T average () const { return m_sum / T (m_values.size ()); }

    void print (void) const { 
      
      Substrate::gPrint (m_name , " [" , m_values.size () , "]"
        , ", max = " , m_max
        , ", min = " , m_min
        , ", sum = " , m_sum
        , ", avg = " , average ()
        , ", range = " , range () 
        , "\n");

      Substrate::gPrint (m_name , " Values[" , m_values.size () , "] = [\n");

      for (typename std::vector<T>::const_iterator i = m_values.begin (), endi = m_values.end ();
          i != endi; ++i) {
        Substrate::gPrint ( *i , ", ");
      }
      Substrate::gPrint ("]\n");
    }

  };
} // end namespace Galois


namespace Galois {
namespace Runtime {

namespace details {

template <typename R, typename F, typename ArgsTuple>
class DoAllCoupledExec {

  typedef typename R::local_iterator Iter;
  typedef typename std::iterator_traits<Iter>::difference_type Diff_ty;

  enum StealAmt {
    HALF, FULL
  };

  struct ThreadContext {

    GALOIS_ATTRIBUTE_ALIGN_CACHE_LINE Substrate::SimpleLock work_mutex;
    unsigned id;

    Iter shared_beg;
    Iter shared_end;
    Diff_ty m_size;
    size_t num_iter;

    // Stats
    static const bool ENABLE_TIMER = false;
    Galois::ThreadTimer<ENABLE_TIMER> timer;
    Galois::ThreadTimer<ENABLE_TIMER> work_timer;
    Galois::ThreadTimer<ENABLE_TIMER> steal_timer;
    Galois::ThreadTimer<ENABLE_TIMER> term_timer;

    ThreadContext () 
      :
        work_mutex (),
        id (Substrate::ThreadPool::getThreadPool().getMaxThreads ()), // TODO: fix this initialization problem, see initThread
        shared_beg (),
        shared_end (),
        m_size (0),
        num_iter (0)
    {}


    ThreadContext (
        unsigned id, 
        Iter beg,
        Iter end) 
      : 
        work_mutex (),
        id (id), 
        shared_beg (beg),
        shared_end (end),
        m_size (std::distance (beg, end)),
        num_iter (0)
    {}


    bool doWork (F& func, const unsigned chunk_size) {
      Iter beg (shared_beg);
      Iter end (shared_end);

      bool didwork = false;

      while (getWork (beg, end, chunk_size)) {

        didwork = true;

        for (; beg != end; ++beg) {
          ++num_iter;
          func (*beg);
        }
      }

      return didwork;
    }

    bool hasWorkWeak () const {
      return (m_size > 0);
    }

    bool hasWork () const {
      bool ret = false;

      work_mutex.lock ();
      {
        ret = hasWorkWeak ();

        if (m_size > 0) {
          assert (shared_beg != shared_end);
        }
      }
      work_mutex.unlock ();

      return ret;
    }

private:

    bool getWork (Iter& priv_beg, Iter& priv_end, const unsigned chunk_size) {
      bool succ = false;

      work_mutex.lock ();
      {
        if (hasWorkWeak ()) {
          succ = true;

          Iter nbeg = shared_beg;
          if (m_size <= chunk_size) {
            nbeg = shared_end;
            m_size = 0;

          } else {
            std::advance (nbeg, chunk_size);
            m_size -= chunk_size;
            assert (m_size > 0);
          }

          priv_beg = shared_beg;
          priv_end = nbeg;
          shared_beg = nbeg;

        }
      }
      work_mutex.unlock ();

      return succ;
    }

    void steal_from_end_impl (Iter& steal_beg, Iter& steal_end, const Diff_ty sz
        , std::forward_iterator_tag) {

      // steal from front for forward_iterator_tag
      steal_beg = shared_beg;
      std::advance (shared_beg, sz);
      steal_end = shared_beg;

    }

    void steal_from_end_impl (Iter& steal_beg, Iter& steal_end, const Diff_ty sz
        , std::bidirectional_iterator_tag) {

      steal_end = shared_end;
      std::advance(shared_end, -sz);
      steal_beg = shared_end;
    }


    void steal_from_end (Iter& steal_beg, Iter& steal_end, const Diff_ty sz) {
      assert (sz > 0);
      steal_from_end_impl (steal_beg, steal_end, sz, typename std::iterator_traits<Iter>::iterator_category ());
    }

    void steal_from_beg (Iter& steal_beg, Iter& steal_end, const Diff_ty sz) {
      assert (sz > 0);
      steal_beg = shared_beg;
      std::advance (shared_beg, sz);
      steal_end = shared_beg;

    }

public:

    bool stealWork (Iter& steal_beg, Iter& steal_end, Diff_ty& steal_size, StealAmt amount, size_t chunk_size) {
      bool succ = false;

      if (work_mutex.try_lock ()) {

        if (hasWorkWeak ()) {
          succ = true;


          if (amount == HALF && m_size > (decltype(m_size))chunk_size) {
            steal_size = m_size / 2;
          } else {
            steal_size = m_size;
          }

          if (m_size <= steal_size) {
            steal_beg = shared_beg;
            steal_end = shared_end;

            shared_beg = shared_end;

            steal_size = m_size;
            m_size = 0;

          } else {

            // steal_from_end (steal_beg, steal_end, steal_size);
            steal_from_beg (steal_beg, steal_end, steal_size);
            m_size -= steal_size;

          }
        }

        work_mutex.unlock ();
      }

      return succ;
    }


    void assignWork (const Iter& beg, const Iter& end, const Diff_ty sz) {
      work_mutex.lock ();
      {
        assert (!hasWorkWeak ());
        assert (beg != end);
        assert (std::distance (beg, end) == sz);

        shared_beg = beg;
        shared_end = end;
        m_size = sz;
      }
      work_mutex.unlock ();
    }



  };


private:
 
  GALOIS_ATTRIBUTE_NOINLINE bool transferWork (ThreadContext& rich, ThreadContext& poor, StealAmt amount) {

    assert (rich.id != poor.id);
    assert (rich.id < Galois::getActiveThreads ());
    assert (poor.id < Galois::getActiveThreads ());

    Iter steal_beg;
    Iter steal_end;

    // stealWork should initialize to a more appropriate value
    Diff_ty steal_size = 0;

    bool succ = rich.stealWork(steal_beg, steal_end, steal_size, amount, chunk_size);

    if (succ) {
      assert (steal_beg != steal_end);
      assert (std::distance (steal_beg, steal_end) == steal_size);

      poor.assignWork (steal_beg, steal_end, steal_size);
    }

    return succ;
  }

  GALOIS_ATTRIBUTE_NOINLINE bool stealWithinPackage (ThreadContext& poor) {

    bool sawWork = false;
    bool stoleWork = false;

    auto& tp = Substrate::ThreadPool::getThreadPool();

    const unsigned maxT = Galois::getActiveThreads ();
    const unsigned my_pack = Substrate::ThreadPool::getPackage ();
    const unsigned per_pack = tp.getMaxThreads() / tp.getMaxPackages ();

    const unsigned pack_beg = my_pack * per_pack;
    const unsigned pack_end = (my_pack + 1) * per_pack;

    for (unsigned i = 1; i < pack_end; ++i) {

      // go around the package in circle starting from the next thread
      unsigned t = (poor.id + i) % per_pack + pack_beg; 
      assert ( (t >= pack_beg) && (t < pack_end));

      if (t < maxT) { 
        if (workers.getRemote (t)->hasWorkWeak ()) {
          sawWork = true;

          stoleWork = transferWork (*workers.getRemote (t), poor, HALF);

          if (stoleWork) { 
            break;
          }
        }
      }
    }
    
    return sawWork || stoleWork;
  }

  GALOIS_ATTRIBUTE_NOINLINE bool stealOutsidePackage (ThreadContext& poor, const StealAmt& amt) {
    bool sawWork = false;
    bool stoleWork = false;

    auto& tp = Substrate::ThreadPool::getThreadPool();
    unsigned myPkg = Substrate::ThreadPool::getPackage();
    // unsigned maxT = LL::getMaxThreads ();
    unsigned maxT = Galois::getActiveThreads ();

    for (unsigned i = 0; i < maxT; ++i) {
      ThreadContext& rich = *(workers.getRemote ((poor.id + i) % maxT));

      if (tp.getPackage(rich.id) != myPkg) {
        if (rich.hasWorkWeak ()) {
          sawWork = true;

          stoleWork = transferWork (rich, poor, amt);
          // stoleWork = transferWork (rich, poor, HALF);

          if (stoleWork) {
            break;
          }
        }
      }
    }

    return sawWork || stoleWork;

  }

  /*
  GALOIS_ATTRIBUTE_NOINLINE bool stealFlat (ThreadContext& poor, const unsigned maxT) {

    // TODO: test performance of sawWork + stoleWork vs stoleWork only
    bool sawWork = false;
    bool stoleWork = false;

    assert ((LL::getMaxCores () / LL::getMaxPackages ()) >= 1);

    // TODO: check this steal amount. e.g. all hungry threads in one package may
    // steal too much work from full threads in another package
    // size_t stealAmt = chunk_size * (LL::getMaxCores () / LL::getMaxPackages ());
    size_t stealAmt = chunk_size;

    for (unsigned i = 1; i < maxT; ++i) { // skip poor.id by starting at 1

      unsigned t = (poor.id + i) % maxT;

      if (workers.getRemote (t)->hasWorkWeak ()) {
        sawWork = true;

        stoleWork = transferWork (*workers.getRemote (t), poor, stealAmt);

        if (stoleWork) {
          break;
        }
      }
    }

    return sawWork || stoleWork;
  }


  GALOIS_ATTRIBUTE_NOINLINE bool stealWithinActive (ThreadContext& poor) {

    return stealFlat (poor, Galois::getActiveThreads ());
  }

  GALOIS_ATTRIBUTE_NOINLINE bool stealGlobal (ThreadContext& poor) {
    return stealFlat (poor, LL::getMaxThreads ());
  }

  */


  GALOIS_ATTRIBUTE_NOINLINE bool trySteal (ThreadContext& poor) {
    bool ret = false;

    ret = stealWithinPackage (poor);

    if (ret) { return true; }

    Substrate::asmPause ();

    if (Substrate::ThreadPool::getThreadPool().isLeader(poor.id)) {
      ret = stealOutsidePackage (poor, HALF);

      if (ret) { return true; }
      Substrate::asmPause ();
    }

    ret = stealOutsidePackage (poor, HALF);
    if (ret) { return true; } 
    Substrate::asmPause ();

    return ret;

    // if (stealWithinPackage (poor)) {
      // return true;
    // } else if (LL::isPackageLeader(poor.id) 
        // && stealOutsidePackage (poor)) {
      // return true;
    // } else if (stealOutsidePackage (poor)) {
      // return true;
// 
    // } else {
      // return false;
    // }
 

    // if (stealWithinPackage (poor)) {
      // return true;
    // } else if (stealWithinActive (poor)) {
      // return true;
    // } else if (stealGlobal (poor)) {
      // return true;
    // } else {
      // return false;
    // }
  }



  void printStats () {

    Galois::AggStatistic<size_t> iter ("Iterations: ");
    Galois::AggStatistic<int64_t> time ("Total time (nsec): ");
    Galois::AggStatistic<int64_t> work_timer ("Work time (nsec): ");
    Galois::AggStatistic<int64_t> steal_timer ("Steal time (nsec): ");
    Galois::AggStatistic<int64_t> term_timer ("Termination time (nsec): ");


    for (unsigned i = 0; i < Galois::getActiveThreads (); ++i) {
      ThreadContext& ctx = *workers.getRemote (i);

      iter.add (ctx.num_iter);
      time.add (ctx.timer.get_nsec ());

      work_timer.add (ctx.work_timer.get_nsec ());
      steal_timer.add (ctx.steal_timer.get_nsec ());
      term_timer.add (ctx.term_timer.get_nsec ());
    }

    // size_t  ave_iter =  total_iter / Galois::getActiveThreads ();

    iter.print ();
    time.print ();
    // work_timer.print ();
    // steal_timer.print ();
    // term_timer.print ();
    Substrate::gPrint ("--------\n");
  }


private:


  R range;
  F func;
  const char* loopname;
  Diff_ty chunk_size;
  Substrate::PerThreadStorage<ThreadContext> workers;

  Substrate::TerminationDetection& term;

  // for stats



public:

  DoAllCoupledExec (
      const R& _range,
      const F& _func, 
      const ArgsTuple& argsTuple)
    : 
      range (_range),
      func (_func), 
      loopname (get_by_supertype<loopname_tag> (argsTuple).value),
      chunk_size (get_by_supertype<chunk_size_tag> (argsTuple).value),
      term(Substrate::getSystemTermination(activeThreads))
  {
    assert (chunk_size > 0);
    // std::printf ("DoAllCoupledExec loopname: %s, work size: %ld, chunk_size: %u\n", loopname, std::distance(range.begin (), range.end ()), chunk_size);



    static_assert(!exists_by_supertype<char*, ArgsTuple>::value, "old loopname");
    static_assert(!exists_by_supertype<char const *, ArgsTuple>::value, "old loopname");
    static_assert(!exists_by_supertype<bool, ArgsTuple>::value, "old steal");

  }

  // parallel call
  void initThread (void) {
    term.initializeThread();

    unsigned id = Substrate::ThreadPool::getTID();

    *workers.getLocal(id) = ThreadContext(id, range.local_begin(), range.local_end());
  }




  ~DoAllCoupledExec () {
    // executed serially
    for (unsigned i = 0; i < workers.size (); ++i) {
      auto& ctx = *(workers.getRemote (i));
      assert (!ctx.hasWork () &&  "Unprocessed work left");
      Galois::Runtime::reportStat (loopname, "Iterations", ctx.num_iter, ctx.id);
    }

    // printStats ();
  }

  void operator () (void) {
    const bool USE_TERM = false;

    
    ThreadContext& ctx = *workers.getLocal ();
    ctx.timer.start ();


    while (true) {
      bool workHappened = false;

      ctx.work_timer.start ();

      if (ctx.doWork (func, chunk_size)) {
        workHappened = true;
      }

      ctx.work_timer.stop ();

      assert (!ctx.hasWork ());

      ctx.steal_timer.start ();
      bool stole = trySteal (ctx);
      ctx.steal_timer.stop ();

      if (stole) {
        continue;

      } else {

        assert (!ctx.hasWork ());
        if (USE_TERM) {
          ctx.term_timer.start ();
          term.localTermination (workHappened);

          bool quit = term.globalTermination ();
          ctx.term_timer.stop ();


          if (quit) {
            break;
          }
        } else {
          break;
        }
      }

    }

    ctx.timer.stop ();
    assert (!ctx.hasWork ());

    // Galois::Runtime::reportStat (loopname, "Iterations", ctx.num_iter, ctx.id);
  }



};

} // end namespace details


template <typename R, typename F, typename _ArgsTuple>
void do_all_coupled (const R& range, const F& func, const _ArgsTuple& argsTuple) {
  auto argsT = std::tuple_cat (argsTuple, 
      get_default_trait_values (argsTuple,
        std::make_tuple (loopname_tag {}, chunk_size_tag {}), 
        std::make_tuple (default_loopname {}, default_chunk_size {})));

  // Creates a timer for this do_all loop
  std::string loopName(get_by_supertype<loopname_tag>(argsT).value);
  std::string num_run_identifier = get_by_supertype<numrun_tag>(argsT).value;
  std::string timer_do_all_str("DO_ALL_IMPL_" + loopName + "_" + num_run_identifier);
  Galois::StatTimer Timer_do_all_impl(timer_do_all_str.c_str());

  Timer_do_all_impl.start();

  using ArgsT = decltype (argsT);
  details::DoAllCoupledExec<R, F, ArgsT> exec (range, func, argsT);

  Substrate::Barrier& barrier = getBarrier(activeThreads);

  Substrate::ThreadPool::getThreadPool().run(activeThreads, 
      [&exec] (void) { exec.initThread (); },
      std::ref(barrier),
      std::ref(exec));

  Timer_do_all_impl.stop();
}

template <typename R, typename F, typename _ArgsTuple>
void do_all_coupled_detailed (const R& range, const F& func, const _ArgsTuple& argsTuple) {

  auto argsT = std::tuple_cat (argsTuple, 
      get_default_trait_values (argsTuple,
        std::make_tuple (loopname_tag {}, chunk_size_tag {}), 
        std::make_tuple (default_loopname {}, default_chunk_size {})));
  using ArgsT = decltype (argsT);
  details::DoAllCoupledExec<R, F, ArgsT> exec (range, func, argsT);

  Runtime::on_each_impl (
      [&exec] (const unsigned tid, const unsigned numT) {
        exec.initThread ();
      });


  std::vector<ThreadTimer<true> > perThrdTimer (Galois::getActiveThreads ());
  

  Runtime::on_each_impl(
      [&exec, &perThrdTimer] (const unsigned tid, const unsigned numT) {

        perThrdTimer[tid].start ();
        exec ();
        perThrdTimer[tid].stop ();
        
      });

  int64_t maxTime = 0;
  for (const auto& t: perThrdTimer) {
    if (maxTime < t.get_nsec ()) {
      maxTime = t.get_nsec ();
    }
  }

  const char* const ln = get_by_supertype<loopname_tag> (argsT).value;

  Runtime::on_each_impl( 
      [&maxTime, &perThrdTimer, ln] (const unsigned tid, const unsigned numT) {
        GALOIS_ASSERT ((maxTime - perThrdTimer[tid].get_nsec ()) >= 0);
        Runtime::reportStat (ln, "LoadImbalance", (unsigned long)(maxTime - perThrdTimer[tid].get_nsec ()), 0);
      });


}

} // end namespace Runtime
} // end namespace Galois

#endif //  GALOIS_RUNTIME_DO_ALL_COUPLED_H_
