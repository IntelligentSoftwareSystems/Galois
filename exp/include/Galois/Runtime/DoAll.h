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
#ifndef GALOIS_RUNTIME_DO_ALL_COUPLED_H_
#define GALOIS_RUNTIME_DO_ALL_COUPLED_H_

#include <algorithm>
#include <vector>
#include <limits>
#include <iostream>

#include <cstdio>


#include "Galois/Runtime/Barrier.h"
#include "Galois/Runtime/LoopHooks.h"
#include "Galois/Runtime/PerCPU.h"
#include "Galois/Runtime/Support.h"
#include "Galois/Runtime/Termination.h"
#include "Galois/Runtime/Threads.h"
#include "Galois/Runtime/ll/PaddedLock.h"


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



namespace GaloisRuntime {

template <typename Iter>
struct Range {
  typedef typename std::iterator_traits<Iter>::difference_type difference_type;

  Iter begin;
  Iter end;
  difference_type size;

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

  typedef typename std::iterator_traits<Iter>::difference_type difference_type;


  struct ThreadContext {

    LL::PaddedLock<true> work_lock;

    unsigned id;

    Iter curr_begin;
    Iter curr_end;
    Iter work_end;

    size_t num_iter;

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

          difference_type d = std::distance (curr_end, work_end); // TODO: check formal param
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
          difference_type d = std::distance (curr_end, work_end);

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

  static const unsigned MAX_CHUNK_SIZE = 1024;
  static const unsigned CHUNK_SCALING_FACTOR = 64;

  void computeChunkSize (const size_t totalSize) {
    chunk_size = std::max (size_t (1), totalSize / (CHUNK_SCALING_FACTOR * Galois::getActiveThreads ()));

    chunk_size = std::min (difference_type (MAX_CHUNK_SIZE), chunk_size); 

    // std::cout << "Choosing a chunk size of " << chunk_size  << std::endl;
  }


  bool findRichSeq (const unsigned poor_id, unsigned& rich_id) {
    bool succ = false;

    unsigned numT = LL::getMaxThreads ();
    for (unsigned i = 1; i < numT; ++i) { // skip poor_id by starting at 1

      unsigned t = (poor_id + i) % numT;
      if (workers.get (t).hasWorkWeak ()) {

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

      if (workers.get (t).hasWorkWeak ()) {
        rich_id = t;
        succ = true;
        break;
      }
    }

    return succ;
  }


  void stealWork (ThreadContext& poor) {
    assert (!poor.hasWork ());

    unsigned rich_id = 0;

    unsigned chunks_to_steal = 1;

    if (!findRichInPackage (poor.id, rich_id)) {
      if (!findRichSeq (poor.id, rich_id)) {
        // failure to find a thread to steal from
        return;

      } else {
        // failed to find in package, but found work outside package, so instead of stealing one
        // chunk, we steal more i.e. num chunks == num cores per package
        chunks_to_steal = std::max (unsigned (1), LL::getMaxCores() / LL::getMaxPackages ());
      }
    }

    Iter begin;
    Iter end;

    if (workers.get (rich_id).stealWork (begin, end, (chunks_to_steal*chunk_size))) {
      poor.assignWork (begin, end);
    }
  }

  void printStats () {
    size_t total_iter = 0;
    size_t min = std::numeric_limits<size_t>::max ();
    size_t max = 0;

    for (unsigned i = 0; i < Galois::getActiveThreads (); ++i) {
      total_iter += workers.get (i).num_iter;

      min = std::min (min, workers.get (i).num_iter);
      max = std::max (max, workers.get (i).num_iter);
      // printf ("Worker %d did %zd iterations\n", i, workers.get (i).num_iter);
    }

    size_t  ave =  total_iter / Galois::getActiveThreads ();
    printf ("Total iterations %s: %zd,   chunk_size=%ld\n", loopname, total_iter, chunk_size);
    printf ("Work distribution: Workers=%d,          min=%zd, max=%zd, average=%zd\n\n"
        , Galois::getActiveThreads (), min, max, ave);

  }


private:
  FuncTp func;
  const char* loopname;
  difference_type chunk_size;
  GaloisRuntime::PerCPU<ThreadContext> workers;
  TerminationDetection term;


public:

  DoAllCoupledExec (
      const PerCPU<Range<Iter> >& ranges, 
      FuncTp& _func, 
      const char* _loopname)
    : 
      func (_func), 
      loopname (_loopname),
      chunk_size (1),
      // default contruction, will construct again
      workers (ThreadContext (0, ranges.get (0)))
  {

    assert (ranges.size () == workers.size ());

    size_t totalSize = 0;

    for (unsigned i = 0; i < ranges.size (); ++i) {
      workers.get (i) = ThreadContext (i, ranges.get (i));
      totalSize += ranges.get (i).size;
    }

    computeChunkSize (totalSize);

  }


  ~DoAllCoupledExec () {
    // executed serially
    for (unsigned i = 0; i < workers.size (); ++i) {
      assert (!workers.get (i).hasWork () &&  "Unprocessed work left");
    }

    // printStats ();
  }

  void operator () () {

    ThreadContext& ctx = workers.get ();
    TerminationDetection::TokenHolder* localterm = term.getLocalTokenHolder ();

    do {

      if (!ctx.hasWork ()) {
        stealWork (ctx);
      }


      while (ctx.getWork (chunk_size)) {

        localterm->workHappened ();
        ctx.doWork (func);
      }

      term.localTermination ();


    } while (!term.globalTermination ());


  }

};


template <typename Iter, typename FuncTp>
void do_all_coupled_impl (PerCPU<Range<Iter> >& ranges, FuncTp& func, const char* loopname) {

  assert (!inGaloisForEach);
  inGaloisForEach = true;


  DoAllCoupledExec<Iter, FuncTp> exec (ranges, func, loopname);

  RunCommand w[2] = { Config::ref (exec), Config::ref (getSystemBarrier ()) };

  getSystemThreadPool ().run (&w[0], &w[2]);
  
  inGaloisForEach = false;
}



template <typename WL, typename FuncTp>
void do_all_coupled (WL& workList, FuncTp func, const char* loopname=0) {
  typedef typename WL::iterator Iter;

  // default construction
  PerCPU<Range<Iter> > ranges (Range<Iter> (workList.begin (0), workList.begin (0)));


  for (unsigned i = 0; i < workList.numRows (); ++i) {
    ranges.get (i) = Range<Iter> (workList.begin (i), workList.end (i), workList.size (i));
  }


  do_all_coupled_impl (ranges, func, loopname);

}

template <typename WL, typename FuncTp>
void do_all_serial (WL& workList, FuncTp func, const char* loopname=0) {

  for (unsigned i = 0; i < workList.numRows (); ++i) {
    std::for_each (workList.begin (i), workList.end (i), func);
  }
}


// template <typename WL, typename FuncTp>
// void do_all_coupled (WL& workList, 
    // typename std::iterator_traits<typename WL::iterator>::difference_type blockSize, 
    // FuncTp func, const char* loopname=0) {
// 
  // typedef typename WL::iterator Iter;
  // typedef typename std::iterator_traits<Iter>::difference_type Diff_ty;
// 
// 
  // assert (blockSize > 0);
// 
  // // default construction
  // PerCPU<Range<Iter> > ranges (Range<Iter> (workList.begin (0), workList.begin (0)));
// 
  // for (unsigned i = 0; i < workList.numRows (); ++i) {
    // Iter blockEnd = workList.begin (i);
// 
    // Diff_ty sz = 0;
// 
    // if (Diff_ty (workList.size (i)) > blockSize) {
      // 
      // std::advance (blockEnd, blockSize);
      // sz = blockSize;
// 
    // } else {
      // // size is <= blockSize
      // blockEnd = workList.end (i);
      // sz = workList.size (i);
    // }
// 
// 
    // ranges.get (i) = Range<Iter> (workList.begin (i), blockEnd, sz);
  // }
// 
  // do_all_coupled_impl (ranges, func, loopname);
// 
// }


template <typename Iter, typename FuncTp>
void do_all_coupled (const Iter begin, const Iter end, FuncTp func, const char* loopname=0) {
  typedef typename std::iterator_traits<Iter>::difference_type difference_type;

  // corner case
  if (begin == end) { 
    return;
  }

  PerCPU<Range<Iter> > ranges (Range<Iter> (begin, begin)); // default construction

  difference_type total = std::distance (begin, end);

  unsigned numT = Galois::getActiveThreads ();

  assert (numT >= 1);
  difference_type perThread = (total + (numT -1)) / numT; // rounding the integer division up
  assert (perThread >= 1);

  

  // We want to support forward iterators as efficiently as possible
  // therefore, we traverse from begin to end once in blocks of numThread
  // except, when we get to last block, we need to make sure iterators
  // don't go past end
  Iter b = begin; // start at beginning
  difference_type inc_amount = perThread;

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

    ranges.get (i) = Range<Iter> (b, e, inc_amount);

    b = e;
  }

  for (unsigned i = last + 1; i < numT; ++i) {
    ranges.get (i) = Range<Iter> (end, end);
  }


  do_all_coupled_impl (ranges, func, loopname);
}




}
#endif //  GALOIS_RUNTIME_DO_ALL_COUPLED_H_
