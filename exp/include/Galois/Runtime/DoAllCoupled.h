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
#include <iostream>

#include <cstdio>
#include <ctime>


#include "Galois/Runtime/Barrier.h"
#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Runtime/Support.h"
#include "Galois/Runtime/Termination.h"
#include "Galois/Runtime/ThreadPool.h"
#include "Galois/Runtime/ll/PaddedLock.h"
#include "Galois/Runtime/ll/CompilerSpecific.h"

#include "Galois/Timer.h"

#define CHUNK_FACTOR 16

#undef ENABLE_DO_ALL_TIMERS

#define USE_NEW_DO_ALL_COUPLED


namespace Galois {
  class ThreadTimer {
    timespec m_start;
    timespec m_stop;

    int64_t  m_nsec;

  public:
    ThreadTimer (): m_nsec (0) {};

    void start () {
      clock_gettime (CLOCK_THREAD_CPUTIME_ID, &m_start);
    }

    void stop () {
      clock_gettime (CLOCK_THREAD_CPUTIME_ID, &m_stop);
      m_nsec += (m_stop.tv_nsec - m_start.tv_nsec);
      m_nsec += ((m_stop.tv_sec - m_start.tv_sec) << 30); // multiply by 1G
    }

    int64_t get_nsec () const { return m_nsec; }

    int64_t get_sec () const { return (m_nsec >> 30); }
      
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

    void print (std::ostream& out=std::cout) const { 
      out << m_name << " [" << m_values.size () << "]"
        << ", max = " << m_max
        << ", min = " << m_min
        << ", sum = " << m_sum
        << ", avg = " << average ()
        << ", range = " << range () 
        << std::endl;

      out << m_name << " Values[" << m_values.size () << "] = [";
      for (typename std::vector<T>::const_iterator i = m_values.begin (), endi = m_values.end ();
          i != endi; ++i) {
        out << *i << ", ";
      }
      out << "]" << std::endl;
    }

  };
}


namespace Galois {
namespace Runtime {

template <typename Iter>
struct Range {
  typedef typename std::iterator_traits<Iter>::difference_type Diff_ty;

  Iter m_beg;
  Iter m_end;
  Diff_ty m_size;

  Range (): m_size (0) {}

  Range (Iter _beg, Iter _end, size_t _size)
    : m_beg (_beg), m_end (_end), m_size (_size)
  {
    assert (m_size >= 0);
  }

  Range (Iter _beg, Iter _end)
    : m_beg (_beg), m_end (_end), m_size (std::distance (_beg, _end))
  {}

};

template <typename Iter, typename FuncTp>
class DoAllCoupledExec {

  typedef typename std::iterator_traits<Iter>::difference_type Diff_ty;


  struct ThreadContext {

    typedef Range<Iter> Range_ty;

    GALOIS_ATTRIBUTE_ALIGN_CACHE_LINE LL::SimpleLock<true> range_mutex;
    unsigned id;

    Range_ty range;
    Iter local_beg;
    Iter local_end;
    size_t num_iter;

    // Stats
#ifdef ENABLE_DO_ALL_TIMERS
    Galois::Timer timer;
    Galois::ThreadTimer work_timer;
    Galois::ThreadTimer steal_timer;
    Galois::ThreadTimer term_timer;
#endif

    ThreadContext () 
      :
        range_mutex (),
        id (std::numeric_limits<unsigned>::max ())
    {}


    ThreadContext (
        unsigned id, 
        const Range_ty& range) 
      : 
        range_mutex (),
        id (id), 
        range (range),
        local_beg (range.m_beg), 
        local_end (range.m_beg), 
        num_iter (0)
    {}


    void doWork (FuncTp& func) {
      for (; local_beg != local_end; ++local_beg) {
        ++num_iter;
        func (*local_beg);
      }
    }

    bool hasWorkWeak () const {
      return (range.m_size > 0);
    }

    bool hasWork () const {
      bool ret = false;

      range_mutex.lock ();
      {
        ret = hasWorkWeak ();

        if (range.m_size > 0) {
          assert (range.m_beg != range.m_end);
        }
      }
      range_mutex.unlock ();

      return ret;
    }

    bool getWork (const unsigned chunk_size) {
      bool succ = false;

      range_mutex.lock ();
      {
        if (hasWorkWeak ()) {
          succ = true;

          Iter nbeg = range.m_beg;
          if (range.m_size <= chunk_size) {
            nbeg = range.m_end;
            range.m_size = 0;

          } else {
            std::advance (nbeg, chunk_size);
            range.m_size -= chunk_size;
            assert (range.m_size > 0);
          }

          local_beg = range.m_beg;
          local_end = nbeg;
          range.m_beg = nbeg;

        }
      }
      range_mutex.unlock ();

      return succ;
    }

private:
    void steal_from_end_impl (Iter& steal_beg, Iter& steal_end, const Diff_ty sz
        , std::forward_iterator_tag) {

      // steal from front for forward_iterator_tag
      steal_beg = range.m_beg;
      std::advance (range.m_beg, sz);
      steal_end = range.m_beg;

    }

    void steal_from_end_impl (Iter& steal_beg, Iter& steal_end, const Diff_ty sz
        , std::bidirectional_iterator_tag) {

      steal_end = range.m_end;
      std::advance(range.m_end, -sz);
      steal_beg = range.m_end;
    }


    void steal_from_end (Iter& steal_beg, Iter& steal_end, const Diff_ty sz) {
      assert (sz > 0);
      steal_from_end_impl (steal_beg, steal_end, sz, typename std::iterator_traits<Iter>::iterator_category ());
    }

public:

    bool stealWork (Iter& steal_beg, Iter& steal_end, Diff_ty& steal_size) {
      bool succ = false;

      if (range_mutex.try_lock ()) {

        if (hasWorkWeak ()) {
          succ = true;


          if (range.m_size < steal_size) {
            steal_beg = range.m_beg;
            steal_end = range.m_end;

            range.m_beg = range.m_end;

            steal_size = range.m_size;
            range.m_size = 0;

          } else {

            steal_from_end (steal_beg, steal_end, steal_size);
            range.m_size -= steal_size;

          }
        }

        range_mutex.unlock ();
      }

      return succ;
    }


    void assignWork (const Iter& beg, const Iter& end, const Diff_ty sz) {
      range_mutex.lock ();
      {
        assert (!hasWorkWeak ());
        assert (beg != end);

        range = Range_ty (beg, end, sz);
        local_beg = range.m_beg;
        local_end = range.m_beg;
      }
      range_mutex.unlock ();
    }



  };


private:
 
#ifdef USE_NEW_DO_ALL_COUPLED

  GALOIS_ATTRIBUTE_NOINLINE bool transferWork (ThreadContext& rich, ThreadContext& poor, Diff_ty steal_size) {

    assert (rich.id != poor.id);

    Iter steal_beg;
    Iter steal_end;

    bool succ = rich.stealWork (steal_beg, steal_end, steal_size);

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
      unsigned t = pack_beg + ((poor.id + 1) % per_pack);
      assert ( (t >= pack_beg) && (t < pack_end));

      if (workers.getRemote (t)->hasWorkWeak ()) {
        sawWork = true;

        stoleWork = transferWork (*workers.getRemote (t), poor, chunk_size);

        if (stoleWork) { 
          break;
        }
      }
    }
    
    return sawWork || stoleWork;
  }

  GALOIS_ATTRIBUTE_NOINLINE bool stealFlat (ThreadContext& poor, const unsigned maxT) {

    // TODO: test performance of sawWork + stoleWork vs stoleWork only
    bool sawWork = false;
    bool stoleWork = false;

    assert ((LL::getMaxCores () / LL::getMaxPackages ()) > 1);

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


  GALOIS_ATTRIBUTE_NOINLINE bool trySteal (ThreadContext& poor) {

    if (stealWithinPackage (poor)) {
      return true;
    } else if (stealWithinActive (poor)) {
      return true;
    } else if (stealGlobal (poor)) {
      return true;
    } else {
      return false;
    }
  }



#else

  bool findRichSeq (const unsigned poor_id, unsigned& rich_id) {
    bool succ = false;

    unsigned numT = LL::getMaxThreads ();
    for (unsigned i = 1; i < numT; ++i) { // skip poor_id by starting at 1

      unsigned t = (poor_id + i) % numT;
      if (workers.getRemote (t)->hasWorkWeak ()) {

        rich_id = t;
        succ = true;
        break;
      }
    }

    return succ;
  }

  bool findRichInPackage (const unsigned poor_id, unsigned& rich_id) {
    bool succ = false;

    unsigned my_pack = LL::getPackageForSelf (poor_id);
    unsigned per_pack = LL::getMaxThreads () / LL::getMaxPackages ();

    unsigned pack_beg = my_pack * per_pack;
    unsigned pack_end = (my_pack + 1) * per_pack;

    for (unsigned i = pack_beg + 1; i < pack_end; ++i) {
      // go around the package in circle starting from the next thread
      unsigned t = pack_beg + ((poor_id + 1) % per_pack);
      assert ( (t >= pack_beg) && (t < pack_end));

      if (workers.getRemote (t)->hasWorkWeak ()) {
        rich_id = t;
        succ = true;
        break;
      }
    }

    return succ;
  }


  bool tryStealing (ThreadContext& poor) {
    assert (!poor.hasWork ());

    bool ret = true;

    unsigned rich_id = 0;

    unsigned chunks_to_steal = 1;

    if (!findRichInPackage (poor.id, rich_id)) {
      if (!findRichSeq (poor.id, rich_id)) {
        // failure to find a thread to steal from
        ret = false;

      } else {
        // failed to find in package, but found work outside package, so instead of stealing one
        // chunk, we steal more i.e. num chunks == num cores per package
        chunks_to_steal = std::max (unsigned (1), LL::getMaxCores() / LL::getMaxPackages ());
      }
    }

    if (ret) { 
      Iter beg;
      Iter end;

      Diff_ty stealAmt = chunks_to_steal * chunk_size;

      if (workers.getRemote (rich_id)->stealWork (beg, end, stealAmt)) {
        poor.assignWork (beg, end, stealAmt);
        ret = true;
      } else {
        ret = false;
      }
    }

    return ret;
  }

  bool findRichFlatSeq (const unsigned poor_id, unsigned& rich_id) {
    bool succ = false;
    
    unsigned numT = LL::getMaxThreads ();
    for (unsigned i = 0; i < numT; ++i) {
      if (workers.getRemote (i)->hasWorkWeak ()) {
        succ = true;
        rich_id = i;
        break;
      }
    }

    return succ;
  }
#endif

  void printStats () {

#ifdef ENABLE_DO_ALL_TIMERS
    Galois::AggStatistic<size_t> iter ("Iterations: ");
    Galois::AggStatistic<unsigned long> time ("Total time (usec): ");
    Galois::AggStatistic<int64_t> work_timer ("Work time (nsec): ");
    Galois::AggStatistic<int64_t> steal_timer ("Steal time (nsec): ");
    Galois::AggStatistic<int64_t> term_timer ("Termination time (nsec): ");


    for (unsigned i = 0; i < Galois::getActiveThreads (); ++i) {
      ThreadContext& ctx = *workers.getRemote (i);

      iter.add (ctx.num_iter);
      time.add (ctx.timer.get_usec ());

      work_timer.add (ctx.work_timer.get_nsec ());
      steal_timer.add (ctx.steal_timer.get_nsec ());
      term_timer.add (ctx.term_timer.get_nsec ());
    }

    // size_t  ave_iter =  total_iter / Galois::getActiveThreads ();

    iter.print ();
    time.print ();
    work_timer.print ();
    steal_timer.print ();
    term_timer.print ();
#endif
  }


private:
  FuncTp func;
  const char* loopname;
  Diff_ty chunk_size;
  Galois::Runtime::PerThreadStorage<ThreadContext> workers;

  TerminationDetection& term;

  // for stats



public:

  DoAllCoupledExec (
      const PerThreadStorage<Range<Iter> >& ranges, 
      FuncTp& _func, 
      const char* _loopname,
      const size_t _chunk_size)
    : 
      func (_func), 
      loopname (_loopname),
      chunk_size (_chunk_size),
      term(getSystemTermination())
  {

    assert (ranges.size () == workers.size ());


    for (unsigned i = 0; i < ranges.size (); ++i) {
      *workers.getRemote (i) = ThreadContext (i, *ranges.getRemote (i));
    }

    chunk_size = std::max (Diff_ty (1), Diff_ty (chunk_size));
    assert (chunk_size > 0);
  }


  ~DoAllCoupledExec () {
    // executed serially
    for (unsigned i = 0; i < workers.size (); ++i) {
      assert (!workers.getRemote (i)->hasWork () &&  "Unprocessed work left");
    }

    // printStats ();
  }

#ifdef USE_NEW_DO_ALL_COUPLED

  void operator () () {

    static const bool USE_TERM = false;

    ThreadContext& ctx = *workers.getLocal ();

    bool workHappened = false;

    while (true) {

      while (ctx.getWork (chunk_size)) {
        ctx.doWork (func);

        if (USE_TERM) { workHappened = true; }
      }

      assert (!ctx.hasWork ());

      if (trySteal (ctx)) {
        continue;

      } else  { 
        assert (!ctx.hasWork ());

        if (USE_TERM) { 
          term.localTermination (workHappened);
	  workHappened = false;
          if (term.globalTermination ()) {
            break;
          }
        } else { 
          break; // no work, no steal, so exiting
        }
      }


    } // end outer while 

    assert (!ctx.hasWork ());

  }

#else 

  void operator () () {

    ThreadContext& ctx = *workers.getLocal ();
    bool workHappened = false;

#ifdef ENABLE_DO_ALL_TIMERS
    ctx.timer.start ();
#endif

    do {

#ifdef ENABLE_DO_ALL_TIMERS
      ctx.steal_timer.start ();
#endif
      if (!ctx.hasWork ()) {
        tryStealing (ctx);
      }
#ifdef ENABLE_DO_ALL_TIMERS
      ctx.steal_timer.stop ();
#endif


#ifdef ENABLE_DO_ALL_TIMERS
      ctx.work_timer.start ();
#endif
      while (ctx.getWork (chunk_size)) {

        workHappened = true;
        ctx.doWork (func);
      }
#ifdef ENABLE_DO_ALL_TIMERS
      ctx.work_timer.stop ();
#endif


#ifdef ENABLE_DO_ALL_TIMERS
      ctx.term_timer.start ();
#endif

      term.localTermination (workHappened);
      workHappened = true;

#ifdef ENABLE_DO_ALL_TIMERS
      ctx.term_timer.stop ();
#endif



    } while (!term.globalTermination ());

#ifdef ENABLE_DO_ALL_TIMERS
    ctx.timer.stop ();
#endif

  }

#endif


};

namespace HIDDEN {
  size_t calc_chunk_size (size_t totalDist) {
    size_t numT = Galois::getActiveThreads ();

    size_t num_chunks = std::max (CHUNK_FACTOR * numT, numT * numT);
    return std::max(size_t (1), totalDist / num_chunks);
  }
} // end namespace HIDDEN


template <typename Iter, typename FuncTp>
void do_all_coupled_impl (PerThreadStorage<Range<Iter> >& ranges, FuncTp& func, const char* loopname, const size_t chunk_size) {

  // assert (!inGaloisForEach);
  // inGaloisForEach = true;


  DoAllCoupledExec<Iter, FuncTp> exec (ranges, func, loopname, chunk_size);

  RunCommand w[2] = { std::ref (exec), std::ref (getSystemBarrier ()) };

  getSystemThreadPool ().run (&w[0], &w[2], activeThreads);
  
  // inGaloisForEach = false;
}


template <typename WL, typename FuncTp>
void do_all_coupled (WL& workList, FuncTp func, const char* loopname=0, size_t chunk_size=0) {
  typedef typename WL::local_iterator Iter;

  PerThreadStorage<Range<Iter> > ranges;

  for (unsigned i = 0; i < workList.numRows (); ++i) {
    *ranges.getRemote (i) = Range<Iter> (workList[i].begin (), workList[i].end (), workList[i].size ());
  }

  
  if (chunk_size == 0) {
    chunk_size = HIDDEN::calc_chunk_size (workList.size_all ());
  }

  do_all_coupled_impl (ranges, func, loopname, chunk_size);
}

template <typename WL, typename FuncTp>
void do_all_coupled_reverse (WL& workList, FuncTp func, const char* loopname=0, size_t chunk_size=0) {

  typedef typename WL::local_reverse_iterator Iter;

  // default construction
  //PerThreadStorage<Range<Iter> > ranges (Range<Iter> (workList[0].rbegin (), workList[0].rbegin ()));
  PerThreadStorage<Range<Iter> > ranges;


  for (unsigned i = 0; i < workList.numRows (); ++i) {
    *ranges.getRemote (i) = Range<Iter> (workList[i].rbegin (), workList[i].rend (), workList[i].size ());
  }

  if (chunk_size == 0) {
    chunk_size = HIDDEN::calc_chunk_size (workList.size_all ());
  }


  do_all_coupled_impl (ranges, func, loopname, chunk_size);

}


template <typename Iter, typename FuncTp>
void do_all_coupled (const Iter beg, const Iter end, FuncTp func, const char* loopname=0, unsigned chunk_size=0) {
  typedef typename std::iterator_traits<Iter>::difference_type Diff_ty;

  // corner case
  if (beg == end) { 
    return;
  }

  //PerThreadStorage<Range<Iter> > ranges (Range<Iter> (beg, beg)); // default construction
  PerThreadStorage<Range<Iter> > ranges;

  Diff_ty total = std::distance (beg, end);

  unsigned numT = Galois::getActiveThreads ();

  assert (numT >= 1);
  Diff_ty perThread = (total + (numT -1)) / numT; // rounding the integer division up
  assert (perThread >= 1);

  

  // We want to support forward iterators as efficiently as possible
  // therefore, we traverse from beg to end once in blocks of numThread
  // except, when we get to last block, we need to make sure iterators
  // don't go past end
  Iter b = beg; // start at beginning
  Diff_ty inc_amount = perThread;

  // iteration when we are in the last interval [b,e)
  // inc_amount must be adjusted at that point
  assert (total >= 0);
  assert (perThread >= 0);
    
  unsigned last = std::min ((numT - 1), unsigned(total/perThread));

  for (unsigned i = 0; i <= last; ++i) {

    if (i == last) {
      inc_amount = std::distance (b, end);
      assert (inc_amount >= 0);
    }

    Iter e = b;
    std::advance (e, inc_amount); // e = b + perThread;

    *ranges.getRemote (i) = Range<Iter> (b, e, inc_amount);

    b = e;
  }

  for (unsigned i = last + 1; i < numT; ++i) {
    *ranges.getRemote (i) = Range<Iter> (end, end);
  }

  if (chunk_size == 0) {
    chunk_size = HIDDEN::calc_chunk_size (total);
  }

  do_all_coupled_impl (ranges, func, loopname, chunk_size);
}


}
}

#endif //  GALOIS_RUNTIME_DO_ALL_COUPLED_H_
