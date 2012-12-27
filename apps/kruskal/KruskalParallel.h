/** Kruskal MST -*- C++ -*-
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
 * Kruskal MST.
 *
 * @author <ahassaan@ices.utexas.edu>
 */

#ifndef KRUSKAL_PARALLEL_H
#define KRUSKAL_PARALLEL_H

#include "Galois/Atomic.h"
#include "Galois/Accumulator.h"
#include "Galois/Runtime/PerThreadWorkList.h"
#include "Galois/Runtime/DoAllCoupled.h"
#include "Galois/Runtime/ll/CompilerSpecific.h"

#ifdef GALOIS_USE_EXP
#include "Galois/DoAllWrap.h"
#endif

#include "Kruskal.h"

namespace kruskal {

template <typename T>
struct MyAtomic {
  GALOIS_ATTRIBUTE_ALIGN_CACHE_LINE volatile const T* val;

  MyAtomic (const T* t): val (t) {}

  MyAtomic (): val () {}

  bool cas (const T* expected, const T* toupdate) {
    if (val != expected) { return false; }

    return __sync_bool_compare_and_swap (&val
        , reinterpret_cast<uintptr_t> (expected)
        , reinterpret_cast<uintptr_t> (toupdate));
  }

  operator T* () const { return const_cast<T*> (val); }

};


struct EdgeCtx;
template <typename T> struct Padded;

typedef VecRep VecRep_ty;
// typedef std::vector<Padded<int> > VecRep_ty;

typedef GaloisRuntime::PerThreadVector<Edge> EdgeWL;
typedef GaloisRuntime::PerThreadVector<EdgeCtx> EdgeCtxWL;
typedef GaloisRuntime::MM::FSBGaloisAllocator<EdgeCtx> EdgeCtxAlloc;
typedef Edge::Comparator Cmp;
typedef Galois::GAccumulator<size_t> Accumulator;

// typedef Galois::GAtomicPadded<EdgeCtx*> AtomicCtxPtr;
// typedef Galois::GAtomic<EdgeCtx*> AtomicCtxPtr;
typedef MyAtomic<EdgeCtx> AtomicCtxPtr;
typedef std::vector<AtomicCtxPtr> VecAtomicCtxPtr;

static const int NULL_EDGE_ID = -1;

template <typename T>
struct Padded {
  GALOIS_ATTRIBUTE_ALIGN_CACHE_LINE T data;

  Padded (const T& x): data (x) {}

  operator T& () { return data; }

  operator const T& () const { return data; }

};


struct EdgeCtx: public Edge {


  bool srcFail;
  bool dstFail;

  EdgeCtx (const Edge& e)
    : Edge (e)
      , srcFail (false), dstFail(false)
  {}

  void setFail (const int rep) {
    assert (rep != NULL_EDGE_ID);

    if (rep == src) {
      srcFail = true;

    } else if (rep == dst) {
      dstFail = true;

    } else {
      abort ();
    }
  }

  void resetStatus () {
    srcFail = false;
    dstFail = false;
  }

  bool isSrcFail () const { 
    return srcFail;
  }

  bool isDstFail () const {
    return dstFail;
  }

  bool isSelf () const { 
    return src == dst;
  }

  bool statusIsReset () const {
    return (!srcFail && !dstFail);
  }


};


template <typename Iter>
Edge pick_kth_internal (Iter b, Iter e, const typename std::iterator_traits<Iter>::difference_type k) {
  typedef typename std::iterator_traits<Iter>::difference_type Dist_ty;
  assert (std::distance (b, e) > k);

  std::sort (b, e, Cmp ());

  return *(b + k);
}

template <typename Iter>
Edge pick_kth (Iter b, Iter e, const typename std::iterator_traits<Iter>::difference_type k) {
  typedef typename std::iterator_traits<Iter>::difference_type Dist_ty;

  static const size_t MAX_SAMPLE_SIZE = 1024;

  Dist_ty total = std::distance (b, e);
  assert (total > 0);

  if (size_t (total) < MAX_SAMPLE_SIZE) {

    std::vector<Edge> sampleSet (b, e);
    return pick_kth_internal (sampleSet.begin (), sampleSet.end (), k);

  } else {

    size_t step = (total + MAX_SAMPLE_SIZE - 1)  / MAX_SAMPLE_SIZE; // round up;

    std::vector<Edge> sampleSet;

    size_t numSamples = total / step;

    for (Iter i = b; i != e; std::advance (i, step)) {

      if (sampleSet.size () == numSamples) {
        break;
      }

      sampleSet.push_back (*i);
    }

    assert (numSamples == sampleSet.size ());
    assert (numSamples <= MAX_SAMPLE_SIZE);

    Dist_ty sample_k = k / step;

    return pick_kth_internal (sampleSet.begin (), sampleSet.end (), sample_k);
  }

}


struct Partition {
  Edge pivot;
  EdgeCtxWL& lighter;
  EdgeWL& heavier;

  Partition (
      const Edge& pivot,
      EdgeCtxWL& lighter,
      EdgeWL& heavier)
    :
      pivot (pivot),
      lighter (lighter),
      heavier (heavier)
  {}


  GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (const Edge& edge) const {

    if (Cmp::compare (edge, pivot) <= 0) { // edge <= pivot
      // EdgeCtx* ctx = alloc.allocate (1);
      // assert (ctx != NULL);
      // alloc.construct (ctx, EdgeCtx(edge));
      EdgeCtx ctx (edge);

      lighter.get ().push_back (ctx);

    } else {
      heavier.get ().push_back (edge);
    }
  }
};


template <typename Iter> 
void partition_edges (Iter b, Iter e, 
    const typename std::iterator_traits<Iter>::difference_type k,
    EdgeCtxWL& lighter, EdgeWL& heavier) {

  assert (b != e);


  size_t old_sz = lighter.size_all () + heavier.size_all ();

  Edge pivot = pick_kth (b, e, k);

  Galois::do_all_choice (b, e, Partition (pivot, lighter, heavier), "partition_loop");

  assert ((lighter.size_all () + heavier.size_all ()) == old_sz + size_t (std::distance (b, e)));

  // TODO: print here
  std::printf ("total size = %ld, input index = %ld\n", std::distance (b, e), k);
  std::printf ("lighter size = %zd, heavier size = %zd\n", 
      lighter.size_all (), heavier.size_all ());

}



struct FindLoop {

  // typedef char tt_does_not_need_parallel_push;

  VecRep_ty& repVec;
  VecAtomicCtxPtr& repOwnerCtxVec;
  Accumulator& findIter;

  FindLoop (
      VecRep_ty& repVec,
      VecAtomicCtxPtr& repOwnerCtxVec,
      Accumulator& findIter)
    :
      repVec (repVec),
      repOwnerCtxVec (repOwnerCtxVec),
      findIter (findIter)
  {}

  GALOIS_ATTRIBUTE_PROF_NOINLINE void claimAsMin (EdgeCtx& ctx, const int rep) {


    bool succ = repOwnerCtxVec[rep].cas (NULL, &ctx);

    assert (repOwnerCtxVec[rep] != NULL);

    if (!succ) {
      for (EdgeCtx* curr = repOwnerCtxVec[rep];
          Cmp::compare (*curr, ctx) > 0; curr = repOwnerCtxVec[rep]) {

        assert (curr != NULL);
        succ = repOwnerCtxVec[rep].cas (curr, &ctx);

        if (succ) {
          curr->setFail (rep);
          assert (Cmp::compare (*repOwnerCtxVec[rep], ctx) <= 0);
          break;
        }
      }
    }

    if (!succ) {
      ctx.setFail (rep);
    }
    
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (EdgeCtx& ctx) {
    findIter += 1;

    assert (ctx.statusIsReset ());

    ctx.src = kruskal::findPCiter_int (ctx.src, repVec);
    ctx.dst = kruskal::findPCiter_int (ctx.dst, repVec);
    
    if (ctx.src != ctx.dst) {
      claimAsMin (ctx, ctx.src);
      claimAsMin (ctx, ctx.dst);

    }     

  }

  template <typename C>
  GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (EdgeCtx& ctx, C&) {
    (*this) (ctx);
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (EdgeCtx& ctx) const {
    const_cast<FindLoop*>(this)->operator () (ctx);
  }
};

template <bool usingOrderedRuntime=false>
struct LinkUpLoop {
  // typedef char tt_does_not_need_parallel_push;

  VecRep_ty& repVec;
  VecAtomicCtxPtr& repOwnerCtxVec;
  EdgeCtxWL& nextWL;
  Accumulator& mstSum;
  Accumulator& linkUpIter;

  LinkUpLoop (
      VecRep_ty& repVec,
      VecAtomicCtxPtr& repOwnerCtxVec,
      EdgeCtxWL& nextWL,
      Accumulator& mstSum,
      Accumulator& linkUpIter)
    :
      repVec (repVec),
      repOwnerCtxVec (repOwnerCtxVec),
      nextWL (nextWL),
      mstSum (mstSum),
      linkUpIter (linkUpIter)
  {}


  GALOIS_ATTRIBUTE_PROF_NOINLINE bool updateODG_test (EdgeCtx& ctx, const int rep) {
    assert (rep >= 0 && size_t (rep) < repOwnerCtxVec.size ());

    return (repOwnerCtxVec[rep] == &ctx);
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void updateODG_reset (EdgeCtx& ctx, const int rep) {

    assert (rep >= 0 && size_t (rep) < repOwnerCtxVec.size ());
    assert (updateODG_test (ctx, rep));

    repOwnerCtxVec[rep] = NULL;
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (EdgeCtx& ctx) {


    if (!ctx.isSelf () ) {


      if (ctx.isSrcFail () && ctx.isDstFail ()) {

        ctx.resetStatus ();

        if (usingOrderedRuntime) {
          GaloisRuntime::signalConflict ();

        } else {
          nextWL.get ().push_back (ctx);
        }

      } else {


        if (!ctx.isSrcFail ()) {
          assert (updateODG_test (ctx, ctx.src));
          linkUp_int (ctx.src, ctx.dst, repVec);

        } else if (!ctx.isDstFail ()) {
          assert (updateODG_test (ctx, ctx.dst));
          linkUp_int (ctx.dst, ctx.src, repVec);
        }

        linkUpIter += 1;
        mstSum += ctx.weight;

        if (!ctx.isSrcFail ()) {
          updateODG_reset (ctx, ctx.src);
        }

        if (!ctx.isDstFail ()) {
          updateODG_reset (ctx, ctx.dst);
        }

      } // end else
    }


    
  }

  template <typename C>
  GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (EdgeCtx& ctx, C&) {
    (*this) (ctx);
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (EdgeCtx& ctx) const {
    const_cast<LinkUpLoop*>(this)->operator () (ctx);
  }
};

template <typename WL>
struct FilterSelfEdges {
  VecRep_ty& repVec;
  WL& filterWL;

  FilterSelfEdges (
      VecRep_ty& repVec, 
      WL& filterWL)
    : 
      repVec (repVec), 
      filterWL (filterWL) 
  {}

  GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (const Edge& edge) {

    int rep1 = findPCiter_int (edge.src, repVec);
    int rep2 = findPCiter_int (edge.dst, repVec);

    if (rep1 != rep2) {
      // EdgeCtx* ctx = alloc.allocate (1);
      // assert (ctx != NULL);
      // alloc.construct (ctx, EdgeCtx (edge));

      filterWL.get ().push_back (edge);
    }
  }
};

struct FillUp {
  EdgeCtxWL& wl;

  explicit FillUp (EdgeCtxWL& wl): wl (wl) {}

  GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (const Edge& edge) {

    // EdgeCtx* ctx = alloc.allocate (1);
    // assert (ctx != NULL);
    // alloc.construct (ctx, EdgeCtx (edge));
    EdgeCtx ctx (edge);

    wl.get ().push_back (ctx);
  }
};


// struct PtrCmp {
// 
  // inline bool operator () (const EdgeCtx* left, const EdgeCtx* right) {
    // assert (left != NULL && right != NULL);
    // return (Cmp::compare (*left, *right) < 0);
  // }
// };

struct PreSort {
  EdgeCtxWL& wl;

  PreSort (EdgeCtxWL& wl): wl (wl) {}

  GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (unsigned tid, unsigned numT) {
    assert (tid < wl.numRows ());
    for (unsigned i = numT; i < wl.numRows (); ++i) {
      assert (wl[i].empty ());
    }
    std::sort (wl[tid].begin (), wl[tid].end (), Cmp ());
  }
};

template <typename WL>
void presort (WL& wl, Galois::TimeAccumulator& sortTimer) {
  sortTimer.start ();
  Galois::on_each (PreSort (wl), "pre_sort");
  sortTimer.stop ();
}

template <typename Iter>
struct Range {
  typedef typename std::iterator_traits<Iter>::difference_type difference_type;
  typedef typename std::iterator_traits<Iter>::value_type value_type;

  typedef GaloisRuntime::PerThreadStorage<Range> PTS;

  Iter m_beg;
  Iter m_end;

  Range (): m_beg (), m_end () {}

  Range (Iter b, Iter e): m_beg (b), m_end (e) {} 

  // TODO: improve for non-random iterators
  const value_type* atOffset (difference_type d) {
    if (m_beg == m_end) {
      return NULL;

    } else {
      if (d >= std::distance (m_beg, m_end)) {
        d = std::distance (m_beg, m_end) - 1;
      }
      Iter i (m_beg);
      std::advance (i, d);
      return &(*i);
    }
  }
};


#if 0
template <typename Iter, typename D>
computeRanges (Iter beg, Iter end, ...) {
  Diff_ty numT = Galois::getActiveThreads ();
  Diff_ty perThread = (totalDist + (numT - 1)) / numT ; // rounding up 
  assert (perThread >= 1);
  
  // We want to support forward iterators as efficiently as possible
  // therefore, we traverse from begin to end once in blocks of numThread
  // except, when we get to last block, we need to make sure iterators
  // don't go past end
  Iter b = begin; // start at beginning
  Diff_ty inc_amount = perThread;

  // iteration when we are in the last interval [b,e)
  // inc_amount must be adjusted at that point
  assert (totalDist >= 0);
  assert (perThread >= 0);
    
  unsigned last = std::min ((numT - 1), unsigned(totalDist/perThread));

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

}


T* findWindowLimit (VecRange& ranges, Diff_ty windowSize, Diff_ty numT) {
  Diff_ty avgIndex = windowSize / numT;

  std::vector<T> values;

  for (unsigned i = 0; i < ranges.size (); ++i) {

    T* val = ranges[i].getIndex (avgIndex);
    if (val != NULL) {
      values.push_back (*val);
    }
  }

  if (!values.empty ()) { 
     std::vector<T>::iterator mid = values.begin () + values.size () / 2;

     std::nth_element (values.begin (), mid, values.end (), cmp);

     return &(*mid);

  } else {
    return NULL;
  }

}
#endif


template <typename T, typename I, typename WL>
struct RefillWorkList {

  const T* windowLimit;
  typename Range<I>::PTS& ranges;
  WL& wl;

  RefillWorkList (
      const T* windowLimit,
      typename Range<I>::PTS& ranges,
      WL& wl)
    :
      windowLimit (windowLimit),
      ranges (ranges),
      wl (wl)
  {}

  void operator () (unsigned i ) {

    assert (i < ranges.size ());

    Range<I>& r = *ranges.getRemote (i);

    for (;r.m_beg != r.m_end; ++r.m_beg) {
      if (Cmp::compare (*r.m_beg, *windowLimit) <= 0) {
        wl.get ().push_back (*r.m_beg);
      } else {
        break;
      }
    }
  } // end method
};


template <typename WL, typename I>
void refillWorkList (WL& wl, typename Range<I>::PTS& ranges, const size_t windowSize, const size_t P) {


  typedef typename Range<I>::value_type T;

  size_t perThrdSize = windowSize / P;

  const T* windowLimit = NULL;

  for (unsigned i = 0; i < ranges.size (); ++i) {
    const T* lim = ranges.getRemote (i)->atOffset (perThrdSize);

    if (lim != NULL) {
      if (windowLimit == NULL || (Cmp::compare (*lim, *windowLimit) > 0)) {
        windowLimit = lim;
      }
    }
  }

  // std::cout << "size before refill: " << wl.size_all () << std::endl;

  if (windowLimit != NULL) {
    // std::cout << "new limit: " << *windowLimit << std::endl;

    Galois::do_all_choice (
        boost::counting_iterator<unsigned> (0),
        boost::counting_iterator<unsigned> (ranges.size ()),
        RefillWorkList<T, I, WL> (windowLimit, ranges, wl));

    for (unsigned i = 0; i < ranges.size (); ++i) {
      Range<I>& r = *ranges.getRemote (i);

      if (r.m_beg != r.m_end) {
        // assuming that ranges are sorted
        // after refill, the first element in each range should be bigger
        // than windowLimit
        assert (Cmp::compare (*r.m_beg, *windowLimit) > 0); 
      }
    }
  } else {

    for (unsigned i = 0; i < ranges.size (); ++i) {
      Range<I>& r = *ranges.getRemote (i);
      assert (r.m_beg == r.m_end);
    }


  }

  // std::cout << "size after refill: " << wl.size_all () << std::endl;
}

struct UnionFindWindow {

  size_t maxRounds;
  size_t lowThresh;

  UnionFindWindow (): maxRounds (64), lowThresh (2) {}

  UnionFindWindow (size_t maxRounds, size_t lowThresh)
    : maxRounds (maxRounds), lowThresh (lowThresh) 
  {}

  void operator () (
      EdgeCtxWL& perThrdWL, 
      VecRep_ty& repVec, 
      VecAtomicCtxPtr& repOwnerCtxVec, 
      size_t& mstWeight, 
      size_t& totalIter,
      Galois::TimeAccumulator& sortTimer,
      Galois::TimeAccumulator& findTimer,
      Galois::TimeAccumulator& linkUpTimer,
      Accumulator& findIter,
      Accumulator& linkUpIter) const {


    typedef EdgeCtxWL::local_iterator Iter;
    typedef Range<Iter> Range_ty;
    typedef Range_ty::PTS PerThrdRange;
    typedef typename Range_ty::difference_type Diff_ty;

    PerThrdRange ranges;

    presort (perThrdWL, sortTimer);

    for (unsigned i = 0; i < ranges.size (); ++i) {
      *ranges.getRemote (i) = Range_ty (perThrdWL[i].begin (), perThrdWL[i].end ());
    }


    const size_t P = Galois::getActiveThreads ();

    const size_t totalDist = perThrdWL.size_all ();
    const size_t windowSize = totalDist / maxRounds;

    const size_t lowThreshSize = windowSize / lowThresh;

    unsigned round = 0;
    size_t numUnions = 0;
    Accumulator mstSum;

    EdgeCtxWL* currWL = new EdgeCtxWL ();
    EdgeCtxWL* nextWL = new EdgeCtxWL ();

    while (true) {
      ++round;
      std::swap (nextWL, currWL);
      nextWL->clear_all ();

      if (currWL->size_all () < lowThreshSize) {
        // size_t s = lowThreshSize - currWL->size_all () + 1;
        sortTimer.start ();
        refillWorkList<EdgeCtxWL, Iter> (*currWL, ranges, windowSize, P);
        sortTimer.stop ();
      }

      if (currWL->empty_all ()) {
        break;
      }

      // GaloisRuntime::beginSampling ();
      findTimer.start ();
      Galois::do_all_choice (*currWL,
          FindLoop (repVec, repOwnerCtxVec, findIter),
          "find_loop");
      findTimer.stop ();
      // GaloisRuntime::endSampling ();


      // GaloisRuntime::beginSampling ();
      linkUpTimer.start ();
      Galois::do_all_choice (*currWL,
          LinkUpLoop<false> (repVec, repOwnerCtxVec, *nextWL, mstSum, linkUpIter),
          "link_up_loop");
      linkUpTimer.stop ();
      // GaloisRuntime::endSampling ();

      int u = linkUpIter.reduce () - numUnions;
      numUnions = linkUpIter.reduce ();

      if (!nextWL->empty_all ()) {
        assert (u > 0 && "no unions, no progress?");
      }

    }

    totalIter += findIter.reduce ();
    mstWeight += mstSum.reduce ();

    std::cout << "Number of rounds: " << round << std::endl;

    delete currWL;
    delete nextWL;

  }

};


template <bool use_presort_tp=false> 
struct UnionFindNaive {

  void operator () (
      EdgeCtxWL& perThrdWL, 
      VecRep_ty& repVec, 
      VecAtomicCtxPtr& repOwnerCtxVec, 
      size_t& mstWeight, 
      size_t& totalIter,
      Galois::TimeAccumulator& sortTimer,
      Galois::TimeAccumulator& findTimer,
      Galois::TimeAccumulator& linkUpTimer,
      Accumulator& findIter,
      Accumulator& linkUpIter) const {



    unsigned round = 0;
    size_t numUnions = 0;

    Accumulator mstSum;

    EdgeCtxWL* const tmp = new EdgeCtxWL ();

    EdgeCtxWL* nextWL = &perThrdWL;
    EdgeCtxWL* currWL = tmp;

    if (use_presort_tp) {
      presort (perThrdWL, sortTimer);
    }

    while (!nextWL->empty_all ()) {
      ++round;
      std::swap (nextWL, currWL);
      nextWL->clear_all ();


      // GaloisRuntime::beginSampling ();
      findTimer.start ();
      Galois::do_all_choice (*currWL,
          FindLoop (repVec, repOwnerCtxVec, findIter),
          "find_loop");
      // std::for_each (currWL->begin_all (), currWL->end_all (),
      // FindLoop (repVec, repOwnerCtxVec, findIter));
      findTimer.stop ();
      // GaloisRuntime::endSampling ();


      // GaloisRuntime::beginSampling ();
      linkUpTimer.start ();
      Galois::do_all_choice (*currWL,
          LinkUpLoop<false> (repVec, repOwnerCtxVec, *nextWL, mstSum, linkUpIter),
          "link_up_loop");
      //    std::for_each (currWL->begin_all (), currWL->end_all (),
      //        LinkUpLoop<false> (repVec, repOwnerCtxVec, *nextWL, mstSum, linkUpIter));
      linkUpTimer.stop ();
      // GaloisRuntime::endSampling ();

      int u = linkUpIter.reduce () - numUnions;
      numUnions = linkUpIter.reduce ();

      if (!nextWL->empty_all ()) {
        assert (u > 0 && "no unions, no progress?");
      }

    }

    totalIter += findIter.reduce ();
    mstWeight += mstSum.reduce ();

    std::cout << "Number of rounds: " << round << std::endl;


    delete tmp;
  }
};


template <typename UF>
void runMSTfilter (const size_t numNodes, const VecEdge& edges,
    size_t& mstWeight, size_t& totalIter, UF ufLoop) {

  totalIter = 0;
  mstWeight = 0;

  Galois::TimeAccumulator runningTime;
  Galois::TimeAccumulator partitionTimer;
  Galois::TimeAccumulator sortTimer;
  Galois::TimeAccumulator findTimer;
  Galois::TimeAccumulator linkUpTimer;
  Galois::TimeAccumulator filterTimer;

  Accumulator findIter;
  Accumulator linkUpIter;



  VecRep_ty repVec (numNodes, -1);
  VecAtomicCtxPtr repOwnerCtxVec (numNodes, AtomicCtxPtr (NULL));


  Galois::preAlloc (16*Galois::getActiveThreads ());

  runningTime.start ();

  partitionTimer.start ();
  EdgeCtxWL lighter;
  EdgeWL heavier;

  typedef std::iterator_traits<VecEdge::iterator>::difference_type Dist_ty;
  Dist_ty splitPoint = numNodes;
  partition_edges (edges.begin (), edges.end (), splitPoint, lighter, heavier);

  partitionTimer.stop ();

  ufLoop (lighter, repVec, repOwnerCtxVec, 
      mstWeight, totalIter, sortTimer, findTimer, linkUpTimer, findIter, linkUpIter);


  // reuse lighter for filterWL
  lighter.clear_all ();

  filterTimer.start ();
  Galois::do_all_choice (heavier,
  // GaloisRuntime::do_all_coupled (heavier, 
      FilterSelfEdges<EdgeCtxWL> (repVec, lighter),
      "filter_loop");
  filterTimer.stop ();


  ufLoop (lighter, repVec, repOwnerCtxVec, 
      mstWeight, totalIter, sortTimer, findTimer, linkUpTimer, findIter, linkUpIter);
  runningTime.stop ();

  std::cout << "Number of FindLoop iterations = " << findIter.reduce () << std::endl;
  std::cout << "Number of LinkUpLoop iterations = " << linkUpIter.reduce () << std::endl;

  std::cout << "MST running time without initialization/destruction: " << runningTime.get () << std::endl;
  std::cout << "Time taken by presort: " << sortTimer.get () << std::endl;
  std::cout << "Time taken by FindLoop: " << findTimer.get () << std::endl;
  std::cout << "Time taken by LinkUpLoop: " << linkUpTimer.get () << std::endl;
  std::cout << "Time taken by partitioning Loop: " << partitionTimer.get () << std::endl;
  std::cout << "Time taken by filter Loop: " << filterTimer.get () << std::endl;

}


template <typename UF>
void runMSTnaive (const size_t numNodes, const VecEdge& edges, 
    size_t& mstWeight, size_t& totalIter, UF ufLoop) {

  totalIter = 0;
  mstWeight = 0;

  Galois::TimeAccumulator runningTime;
  Galois::TimeAccumulator sortTimer;
  Galois::TimeAccumulator findTimer;
  Galois::TimeAccumulator linkUpTimer;

  Accumulator findIter;
  Accumulator linkUpIter;


  VecRep_ty repVec (numNodes, -1);
  VecAtomicCtxPtr repOwnerCtxVec (numNodes, AtomicCtxPtr (NULL));



  Galois::preAlloc (16*Galois::getActiveThreads ());

  EdgeCtxWL initWL;

  Galois::do_all_choice (edges.begin (), edges.end (), FillUp (initWL), "fill_init");

  runningTime.start ();

  ufLoop (initWL, repVec, repOwnerCtxVec, 
      mstWeight, totalIter, sortTimer, findTimer, linkUpTimer, findIter, linkUpIter);

  runningTime.stop ();

  std::cout << "Number of FindLoop iterations = " << findIter.reduce () << std::endl;
  std::cout << "Number of LinkUpLoop iterations = " << linkUpIter.reduce () << std::endl;

  std::cout << "MST running time without initialization/destruction: " << runningTime.get () << std::endl;
  std::cout << "Time taken by sortTimer: " << sortTimer.get () << std::endl;
  std::cout << "Time taken by FindLoop: " << findTimer.get () << std::endl;
  std::cout << "Time taken by LinkUpLoop: " << linkUpTimer.get () << std::endl;

}


}// end namespace kruskal



#endif //  KRUSKAL_PARALLEL_H

