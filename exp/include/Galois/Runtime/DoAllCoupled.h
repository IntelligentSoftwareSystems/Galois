/** Do All-*- C++ -*-
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


#include "Galois/Runtime/Barrier.h"
#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Runtime/Support.h"
#include "Galois/Runtime/Termination.h"
#include "Galois/Runtime/ThreadPool.h"
#include "Galois/Runtime/ll/PaddedLock.h"
#include "Galois/Runtime/ll/CompilerSpecific.h"
#include "Galois/Runtime/ll/gio.h"

#include "Galois/Timer.h"


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

    T average () const { return m_sum / m_values.size (); }

    void print (void) const { 
      
      Runtime::LL::gPrint (m_name , " [" , m_values.size () , "]"
        , ", max = " , m_max
        , ", min = " , m_min
        , ", sum = " , m_sum
        , ", avg = " , average ()
        , ", range = " , range () 
        , "\n");

      Runtime::LL::gPrint (m_name , " Values[" , m_values.size () , "] = [\n");

      for (typename std::vector<T>::const_iterator i = m_values.begin (), endi = m_values.end ();
          i != endi; ++i) {
        Runtime::LL::gPrint ( *i , ", ");
      }
      Runtime::LL::gPrint ("]\n");
    }

  };
} // end namespace Galois


namespace Galois {
namespace Runtime {

namespace details {

static const unsigned DEFAULT_CHUNK_SIZE = 16;

template <typename R, typename F>
class DoAllCoupledExec {

  typedef typename R::local_iterator Iter;
  typedef typename std::iterator_traits<Iter>::difference_type Diff_ty;

  enum StealAmt {
    HALF, FULL
  };

  struct ThreadContext {

    GALOIS_ATTRIBUTE_ALIGN_CACHE_LINE LL::SimpleLock work_mutex;
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
        id (std::numeric_limits<unsigned>::max ()),
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


          if (amount == HALF && m_size > chunk_size) {
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

    Iter steal_beg;
    Iter steal_end;
    Diff_ty steal_size;

    bool succ = rich.stealWork (steal_beg, steal_end, steal_size, amount, chunk_size);

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

    unsigned my_pack = LL::getPackageForSelf (poor.id);
    unsigned per_pack = LL::getMaxThreads () / LL::getMaxPackages ();

    unsigned pack_beg = my_pack * per_pack;
    unsigned pack_end = (my_pack + 1) * per_pack;

    for (unsigned i = pack_beg + 1; i < pack_end; ++i) {
      // go around the package in circle starting from the next thread
      unsigned t = pack_beg + ((poor.id + i) % per_pack);
      assert ( (t >= pack_beg) && (t < pack_end));

      if (workers.getRemote (t)->hasWorkWeak ()) {
        sawWork = true;

        stoleWork = transferWork (*workers.getRemote (t), poor, HALF);

        if (stoleWork) { 
          break;
        }
      }
    }
    
    return sawWork || stoleWork;
  }

  GALOIS_ATTRIBUTE_NOINLINE bool stealOutsidePackage (ThreadContext& poor) {
    bool sawWork = false;
    bool stoleWork = false;

    unsigned myPkg = LL::getPackageForThread (poor.id);
    unsigned maxT = LL::getMaxThreads ();

    for (unsigned i = 0; i < maxT; ++i) {
      ThreadContext& rich = *(workers.getRemote ((poor.id + i) % maxT));

      if (LL::getPackageForThread (rich.id) != myPkg) {
        if (rich.hasWorkWeak ()) {
          sawWork = true;

          // stoleWork = transferWork (rich, poor, FULL);
          stoleWork = transferWork (rich, poor, HALF);

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

    LL::asmPause ();

    if (LL::isPackageLeader(poor.id)) {
      ret = stealOutsidePackage (poor);

      if (ret) { return true; }
      LL::asmPause ();
    }

    ret = stealOutsidePackage (poor);
    if (ret) { return true; } 
    LL::asmPause ();

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
    LL::gPrint ("--------\n");
  }


private:
  R range;
  F func;
  const char* loopname;
  Diff_ty chunk_size;
  Galois::Runtime::PerThreadStorage<ThreadContext> workers;

  TerminationDetection& term;

  // for stats



public:

  DoAllCoupledExec (
      const R& _range,
      const F& _func, 
      const char* _loopname,
      const size_t _chunk_size)
    : 
      range (_range),
      func (_func), 
      loopname (_loopname),
      chunk_size (_chunk_size),
      term(getSystemTermination())
  {

    chunk_size = std::max (Diff_ty (1), Diff_ty (chunk_size));
    assert (chunk_size > 0);

  }

  // parallel call
  void initThread (void) {
    term.initializeThread ();

    unsigned id = LL::getTID ();

    *workers.getLocal (id) = ThreadContext (id, range.local_begin (), range.local_end ());

  }




  ~DoAllCoupledExec () {
    // executed serially
    for (unsigned i = 0; i < workers.size (); ++i) {
      assert (!workers.getRemote (i)->hasWork () &&  "Unprocessed work left");
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
  }



};

} // end namespace details


template <typename R, typename F>
void do_all_coupled (const R& range, const F& func, const char* loopname=0, const size_t chunk_size=details::DEFAULT_CHUNK_SIZE) {

  assert (!inGaloisForEach);
  inGaloisForEach = true;

  details::DoAllCoupledExec<R, F> exec (range, func, loopname, chunk_size);
  Barrier& barrier = getSystemBarrier();

  getSystemThreadPool().run(activeThreads, 
      [&exec] (void) { exec.initThread (); },
      std::ref(barrier),
      std::ref(exec));
  
  inGaloisForEach = false;


}

} // end namespace Runtime
} // end namespace Galois

#endif //  GALOIS_RUNTIME_DO_ALL_COUPLED_H_
