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
#include "Galois/Runtime/ll/CompilerSpecific.h"

#include "Kruskal.h"

namespace kruskal {

struct EdgeCtx;

typedef VecRep VecRep_ty;

typedef GaloisRuntime::PerThreadVector<Edge> EdgeWL;
typedef GaloisRuntime::PerThreadVector<EdgeCtx> EdgeCtxWL;
typedef GaloisRuntime::MM::FSBGaloisAllocator<EdgeCtx> EdgeCtxAlloc;
typedef Edge::Comparator Cmp;
typedef Galois::GAccumulator<size_t> Accumulator;

// typedef Galois::GAtomicPadded<EdgeCtx*> AtomicCtxPtr;
typedef Galois::GAtomic<EdgeCtx*> AtomicCtxPtr;
typedef std::vector<AtomicCtxPtr> VecAtomicCtxPtr;

static const int NULL_EDGE_ID = -1;

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

    if (usingOrderedRuntime) {
      return ((EdgeCtx*) repOwnerCtxVec[rep])->id == ctx.id;

    } else {
      return (repOwnerCtxVec[rep] == &ctx);
    }
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
struct PreSort {
  WL& wl;

  PreSort (WL& wl): wl (wl) {}

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
  Galois::on_each (PreSort<WL> (wl), "pre_sort");
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

  void operator () (unsigned tid, unsigned numT) {

    assert (tid < ranges.size ());

    for (unsigned i = numT; i < ranges.size (); ++i) {
      Range<I>& r = *ranges.getRemote (i);
      assert (r.m_beg == r.m_end);
    }

    Range<I>& r = *ranges.getRemote (tid);

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
void refillWorkList (WL& wl, typename Range<I>::PTS& ranges, const size_t windowSize, const size_t numT) {


  typedef typename Range<I>::value_type T;

  size_t perThrdSize = windowSize / numT;

  const T* windowLimit = NULL;

  for (unsigned i = 0; i < numT; ++i) {
    assert (i < ranges.size ());
    const T* lim = ranges.getRemote (i)->atOffset (perThrdSize);

    if (lim != NULL) {
      if (windowLimit == NULL || (Cmp::compare (*lim, *windowLimit) > 0)) {
        windowLimit = lim;
      }
    }
  }

  GALOIS_DEBUG ("size before refill: %zd", wl.size_all ());

  if (windowLimit != NULL) {
    GALOIS_DEBUG ("new window limit: %s", windowLimit->str ().c_str ());

    Galois::on_each (RefillWorkList<T, I, WL> (windowLimit, ranges, wl), "refill");

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

  GALOIS_DEBUG ("size after refill: %zd", wl.size_all ())
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
    typedef Range_ty::difference_type Diff_ty;

    PerThrdRange ranges;

    presort (perThrdWL, sortTimer);

    for (unsigned i = 0; i < ranges.size (); ++i) {
      *ranges.getRemote (i) = Range_ty (perThrdWL[i].begin (), perThrdWL[i].end ());
    }


    const size_t numT = Galois::getActiveThreads ();

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

      if (currWL->size_all () <= lowThreshSize) {
        // size_t s = lowThreshSize - currWL->size_all () + 1;
        sortTimer.start ();
        refillWorkList<EdgeCtxWL, Iter> (*currWL, ranges, windowSize, numT);
        sortTimer.stop ();
      }

      if (currWL->empty_all ()) {
        break;
      }

      // GaloisRuntime::beginSampling ();
      findTimer.start ();
      Galois::do_all (currWL->begin_all (), currWL->end_all (),
          FindLoop (repVec, repOwnerCtxVec, findIter),
          "find_loop");
      findTimer.stop ();
      // GaloisRuntime::endSampling ();


      // GaloisRuntime::beginSampling ();
      linkUpTimer.start ();
      Galois::do_all (currWL->begin_all (), currWL->end_all (),
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


struct FillUp {
  EdgeCtxWL& wl;

  explicit FillUp (EdgeCtxWL& wl): wl (wl) {}

  GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (const Edge& edge) {
    wl.get ().push_back (edge);
  }
};



template <typename UF>
void runMSTsimple (const size_t numNodes, const VecEdge& edges, 
    size_t& mstWeight, size_t& totalIter, UF ufLoop) {

  totalIter = 0;
  mstWeight = 0;

  Galois::TimeAccumulator runningTime;
  Galois::TimeAccumulator sortTimer;
  Galois::TimeAccumulator findTimer;
  Galois::TimeAccumulator linkUpTimer;
  Galois::TimeAccumulator fillUpTimer;

  Accumulator findIter;
  Accumulator linkUpIter;


  VecRep_ty repVec (numNodes, -1);
  VecAtomicCtxPtr repOwnerCtxVec (numNodes, AtomicCtxPtr (NULL));


  fillUpTimer.start ();
  EdgeCtxWL initWL;
  unsigned numT = Galois::getActiveThreads ();
  for (unsigned i = 0; i < numT; ++i) {
    initWL[i].reserve ((edges.size () + numT - 1) / numT);
  }

  Galois::do_all (edges.begin (), edges.end (), FillUp (initWL), "fill_init");

  fillUpTimer.stop ();

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
  std::cout << "Time taken by FillUp: " << fillUpTimer.get () << std::endl;

}





}// end namespace kruskal



#endif //  KRUSKAL_PARALLEL_H

