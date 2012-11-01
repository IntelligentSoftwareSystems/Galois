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
#include "Galois/Runtime/LoopHooks.h"
#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Runtime/Support.h"
#include "Galois/Runtime/Termination.h"
#include "Galois/Runtime/ThreadPool.h"
#include "Galois/Runtime/ll/PaddedLock.h"
#include "Galois/Runtime/ll/CompilerSpecific.h"

#include "Galois/Timer.h"

#define  MAX_CHUNK_SIZE 8UL

#undef ENABLE_DO_ALL_TIMERS
// TODO: assume bidirectional iterators as the lcd
// and improve the code below

namespace std {

template <typename Iter>
void decrease (Iter& i, typename iterator_traits<Iter>::difference_type dist,
    random_access_iterator_tag) {
  i -= dist;
}

template <typename Iter>
void decrease (Iter& i, typename iterator_traits<Iter>::difference_type dist,
    bidirectional_iterator_tag) {
  while (dist > 0) {
    --i;
    --dist;
  }
}

template <typename Iter>
void decrease (Iter& i, typename iterator_traits<Iter>::difference_type dist) {
  decrease (i, dist, typename iterator_traits<Iter>::iterator_category ());
}


} // end namespace std


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
        << ", ave = " << average ()
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


namespace GaloisRuntime {

template <typename Iter>
struct Range {
  typedef typename std::iterator_traits<Iter>::difference_type Diff_ty;

  Iter begin;
  Iter end;
  Diff_ty size;

  Range (): size (0) {}

  Range (Iter _begin, Iter _end, size_t _size)
    : begin (_begin), end (_end), size (_size)
  {
    assert (size >= 0);
  }

  Range (Iter _begin, Iter _end)
    : begin (_begin), end (_end), size (std::distance (_begin, _end))
  {}

};

template <typename Iter, typename FuncTp>
class DoAllCoupledExec {

  typedef typename std::iterator_traits<Iter>::difference_type Diff_ty;


  struct ThreadContext {

    GALOIS_ATTRIBUTE_ALIGN_CACHE_LINE LL::SimpleLock<true> work_lock;

    unsigned id;

    Iter curr_begin;
    Iter curr_end;
    Iter work_end;

    // Stats
    size_t num_iter;
#ifdef ENABLE_DO_ALL_TIMERS
    Galois::Timer timer;
    Galois::ThreadTimer work_timer;
    Galois::ThreadTimer steal_timer;
    Galois::ThreadTimer term_timer;
#endif

    ThreadContext () 
      :
        work_lock (),
        id (std::numeric_limits<unsigned>::max ()),
        num_iter (0)
    {}


    ThreadContext (
        unsigned id, 
        const Range<Iter>& range) 
      : 
        work_lock (),
        id (id), 
        curr_begin (range.begin), 
        curr_end (range.begin), 
        work_end (range.end),
        num_iter (0)
    {}


    void doWork (FuncTp& func) {
      for (; curr_begin != curr_end; ++curr_begin) {
        func (*curr_begin);
        ++num_iter;
      }
    }

    bool hasWorkWeak () const {
      return (curr_end != work_end);
    }

    bool hasWork () const {
      bool ret = false;

      work_lock.lock ();
      {
        ret = hasWorkWeak ();
      }
      work_lock.unlock ();

      return ret;
    }

    bool getWork (const unsigned chunk_size) {
      bool succ = false;

      work_lock.lock ();
      {
        if (hasWorkWeak ()) {
          succ = true;

          Diff_ty d = std::distance (curr_end, work_end); // TODO: check formal param
          if (d < chunk_size) {
            curr_end = work_end;

          } else {
            std::advance (curr_end, chunk_size);
          }

        }
      }
      work_lock.unlock ();

      return succ;
    }

    bool stealWork (Iter& steal_begin, Iter& steal_end, const unsigned steal_size) {
      bool succ = false;

      work_lock.lock ();
      { 
        if (hasWorkWeak ()) {
          succ = true;
          Diff_ty d = std::distance (curr_end, work_end);

          if (d < steal_size) {
            steal_begin = curr_end;
            steal_end = work_end;

          } else {
            
            // steal_begin = curr_end;
            // std::advance (steal_begin, (d - steal_size));
// 
            // steal_end = work_end;

            steal_begin = work_end;
            steal_end = work_end;
            std::decrease (steal_begin, steal_size);
          }

          // don't forget to shrink my range to beginning of steal range
          work_end = steal_begin;
        }
      }
      work_lock.unlock ();

      return succ;
    }

    void assignWork (Iter begin, Iter end) {
      work_lock.lock ();
      {
        assert (!hasWorkWeak ());
        assert (begin != end);

        curr_begin = curr_end = begin;
        work_end = end;
      }
      work_lock.unlock ();
    }


  };


private:

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

    unsigned my_pack = LL::getPackageForThread (poor_id);
    unsigned per_pack = LL::getMaxThreads () / LL::getMaxPackages ();

    unsigned pack_begin = my_pack * per_pack;
    unsigned pack_end = (my_pack + 1) * per_pack;

    for (unsigned i = pack_begin + 1; i < pack_end; ++i) {
      // go around the package in circle starting from the next thread
      unsigned t = pack_begin + ((poor_id + 1) % per_pack);
      assert ( (t >= pack_begin) && (t < pack_end));

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
      Iter begin;
      Iter end;

      if (workers.getRemote (rich_id)->stealWork (begin, end, (chunks_to_steal*chunk_size))) {
        poor.assignWork (begin, end);
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
  GaloisRuntime::PerThreadStorage<ThreadContext> workers;
  TerminationDetection term;

  // for stats



public:

  DoAllCoupledExec (
      const PerThreadStorage<Range<Iter> >& ranges, 
      FuncTp& _func, 
      const char* _loopname,
      const unsigned maxChunkSize)
    : 
      func (_func), 
      loopname (_loopname),
      chunk_size (1)
  {

    assert (ranges.size () == workers.size ());

    assert (maxChunkSize > 0);

    for (unsigned i = 0; i < ranges.size (); ++i) {
      *workers.getRemote (i) = ThreadContext (i, *ranges.getRemote (i));
    }

    chunk_size = std::max (Diff_ty (1), Diff_ty (maxChunkSize));
  }


  ~DoAllCoupledExec () {
    // executed serially
    for (unsigned i = 0; i < workers.size (); ++i) {
      assert (!workers.getRemote (i)->hasWork () &&  "Unprocessed work left");
    }

    // printStats ();
  }

  void operator () () {

    ThreadContext& ctx = *workers.getLocal ();
    TerminationDetection::TokenHolder* localterm = term.getLocalTokenHolder ();

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

        localterm->workHappened ();
        ctx.doWork (func);
      }
#ifdef ENABLE_DO_ALL_TIMERS
      ctx.work_timer.stop ();
#endif


#ifdef ENABLE_DO_ALL_TIMERS
      ctx.term_timer.start ();
#endif

      term.localTermination ();

#ifdef ENABLE_DO_ALL_TIMERS
      ctx.term_timer.stop ();
#endif



    } while (!term.globalTermination ());

#ifdef ENABLE_DO_ALL_TIMERS
    ctx.timer.stop ();
#endif

  }


  // void operator () () {
// 
    // ThreadContext& ctx = workers.get ();
    // TerminationDetection::TokenHolder* localterm = term.getLocalTokenHolder ();
// 
// #ifdef ENABLE_DO_ALL_TIMERS
    // ctx.timer.start ();
// #endif
// 
    // do {
// 
// 
      // do {
// 
// #ifdef ENABLE_DO_ALL_TIMERS
        // ctx.work_timer.start ();
// #endif
        // while (ctx.getWork (chunk_size)) {
          // localterm->workHappened ();
          // ctx.doWork (func);
        // }
// #ifdef ENABLE_DO_ALL_TIMERS
        // ctx.work_timer.stop ();
// #endif
// 
        // if (ctx.hasWorkWeak ()) { std::abort (); } // should be empty at this point
// 
// #ifdef ENABLE_DO_ALL_TIMERS
        // ctx.steal_timer.start ();
// #endif
        // if (tryStealing (ctx)) {
          // continue;
        // }
// #ifdef ENABLE_DO_ALL_TIMERS
        // ctx.steal_timer.stop ();
// #endif
// 
      // } while (ctx.hasWorkWeak ()); // can tolerate
// 
// 
// #ifdef ENABLE_DO_ALL_TIMERS
      // ctx.term_timer.start ();
// #endif
      // term.localTermination ();
// #ifdef ENABLE_DO_ALL_TIMERS
      // ctx.term_timer.stop ();
// #endif
// 
// 
    // } while (!term.globalTermination ());
// 
// #ifdef ENABLE_DO_ALL_TIMERS
    // ctx.timer.stop ();
// #endif
  // }
};


template <typename Iter, typename FuncTp>
void do_all_coupled_impl (PerThreadStorage<Range<Iter> >& ranges, FuncTp& func, const char* loopname, const unsigned maxChunkSize) {

  assert (!inGaloisForEach);
  inGaloisForEach = true;


  DoAllCoupledExec<Iter, FuncTp> exec (ranges, func, loopname, maxChunkSize);

  RunCommand w[2] = { Config::ref (exec), Config::ref (getSystemBarrier ()) };

  getSystemThreadPool ().run (&w[0], &w[2]);
  
  inGaloisForEach = false;
}

template <typename WL, typename FuncTp>
void do_all_serial (WL& workList, FuncTp func, const char* loopname=0) {

  for (unsigned i = 0; i < workList.numRows (); ++i) {
    std::for_each (workList[i].begin (), workList[i].end (), func);
  }
}


template <typename WL, typename FuncTp>
void do_all_coupled (WL& workList, FuncTp func, const char* loopname=0, const unsigned maxChunkSize=MAX_CHUNK_SIZE) {
  typedef typename WL::local_iterator Iter;

  // default construction
  //PerThreadStorage<Range<Iter> > ranges (Range<Iter> (workList[0].begin (), workList[0].begin ()));
  PerThreadStorage<Range<Iter> > ranges;


  for (unsigned i = 0; i < workList.numRows (); ++i) {
    *ranges.getRemote (i) = Range<Iter> (workList[i].begin (), workList[i].end (), workList[i].size ());
  }


  do_all_coupled_impl (ranges, func, loopname, maxChunkSize);

}

template <typename WL, typename FuncTp>
void do_all_coupled_reverse (WL& workList, FuncTp func, const char* loopname=0, const unsigned maxChunkSize=MAX_CHUNK_SIZE) {
  typedef typename WL::local_reverse_iterator Iter;

  // default construction
  //PerThreadStorage<Range<Iter> > ranges (Range<Iter> (workList[0].rbegin (), workList[0].rbegin ()));
  PerThreadStorage<Range<Iter> > ranges;


  for (unsigned i = 0; i < workList.numRows (); ++i) {
    *ranges.getRemote (i) = Range<Iter> (workList[i].rbegin (), workList[i].rend (), workList[i].size ());
  }


  do_all_coupled_impl (ranges, func, loopname, maxChunkSize);

}


template <typename Iter, typename FuncTp>
void do_all_coupled (const Iter begin, const Iter end, FuncTp func, const char* loopname=0, const unsigned maxChunkSize=MAX_CHUNK_SIZE) {
  typedef typename std::iterator_traits<Iter>::difference_type Diff_ty;

  // corner case
  if (begin == end) { 
    return;
  }

  //PerThreadStorage<Range<Iter> > ranges (Range<Iter> (begin, begin)); // default construction
  PerThreadStorage<Range<Iter> > ranges;

  Diff_ty total = std::distance (begin, end);

  unsigned numT = Galois::getActiveThreads ();

  assert (numT >= 1);
  Diff_ty perThread = (total + (numT -1)) / numT; // rounding the integer division up
  assert (perThread >= 1);

  

  // We want to support forward iterators as efficiently as possible
  // therefore, we traverse from begin to end once in blocks of numThread
  // except, when we get to last block, we need to make sure iterators
  // don't go past end
  Iter b = begin; // start at beginning
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


  do_all_coupled_impl (ranges, func, loopname, maxChunkSize);
}




}
#endif //  GALOIS_RUNTIME_DO_ALL_COUPLED_H_
