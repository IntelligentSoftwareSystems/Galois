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
#include "Galois/PerThreadContainer.h"
#include "Galois/DynamicArray.h"

#include "Galois/Runtime/KDGtwoPhase.h"
#include "Galois/Substrate/CompilerSpecific.h"

#include "Kruskal.h"

namespace kruskal {

struct EdgeCtx;

// typedef galois::LazyDynamicArray<int>  VecRep_ty;
typedef galois::LazyDynamicArray<int, galois::runtime::SerialNumaAllocator<int> >  VecRep_ty;

typedef galois::PerThreadVector<Edge> EdgeWL;
typedef galois::PerThreadVector<EdgeCtx> EdgeCtxWL;
typedef galois::FixedSizeAllocator<EdgeCtx> EdgeCtxAlloc;
typedef Edge::Comparator Cmp;

// typedef galois::GAtomicPadded<EdgeCtx*> AtomicCtxPtr;
typedef galois::GAtomic<EdgeCtx*> AtomicCtxPtr;
// typedef galois::LazyDynamicArray<AtomicCtxPtr> VecAtomicCtxPtr;
typedef galois::LazyDynamicArray<AtomicCtxPtr, galois::runtime::SerialNumaAllocator<AtomicCtxPtr> > VecAtomicCtxPtr;

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

  typedef char tt_does_not_need_push;

  static const unsigned CHUNK_SIZE = 16;

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

    // int s = kruskal::getRep_int (ctx.src, repVec);
    // int d = kruskal::getRep_int (ctx.dst, repVec);
    // 
    // if (s != d) {
      // ctx.src = kruskal::findPCiter_int (ctx.src, repVec);
      // ctx.dst = kruskal::findPCiter_int (ctx.dst, repVec);
      // 
      // claimAsMin (ctx, ctx.src);
      // claimAsMin (ctx, ctx.dst);
// 
    // } else {
      // ctx.src = s;
      // ctx.dst = d;
    // }

    // ctx.src = kruskal::getRep_int (ctx.src, repVec);
    // ctx.dst = kruskal::getRep_int (ctx.dst, repVec);
    // 
    // if (ctx.src != ctx.dst) {
      // claimAsMin (ctx, ctx.src);
      // claimAsMin (ctx, ctx.dst);
    // }
    
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
  typedef char tt_does_not_need_push;
  static const size_t CHUNK_SIZE = 16;

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
          galois::runtime::signalConflict ();

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
void presort (WL& wl, galois::TimeAccumulator& sortTimer) {
  sortTimer.start ();
  galois::on_each (PreSort<WL> (wl), galois::loopname("pre_sort"));
  sortTimer.stop ();
}

template <typename Iter>
struct Range {
  typedef typename std::iterator_traits<Iter>::difference_type difference_type;
  typedef typename std::iterator_traits<Iter>::value_type value_type;

  typedef galois::substrate::PerThreadStorage<Range> PTS;

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
void refillWorkList (WL& wl, typename Range<I>::PTS& ranges, const size_t prevWindowSize, const size_t numCommits) {

  typedef typename Range<I>::value_type T;

  const size_t chunkSize = LinkUpLoop<false>::CHUNK_SIZE;
  const size_t numT = galois::getActiveThreads ();

  const size_t minWinSize = numT * chunkSize;

  double TARGET_COMMIT_RATIO = galois::runtime::commitRatioArg;

  double commitRatio = double (numCommits) / double (prevWindowSize);

  size_t windowSize = 0;

  if (prevWindowSize == 0) {
    windowSize = minWinSize;

  } else {
    if (commitRatio >= TARGET_COMMIT_RATIO) {
      windowSize = 2 * prevWindowSize;

    } else {
      windowSize = size_t (commitRatio * prevWindowSize);
    }
  }

  if (windowSize < minWinSize) {
    windowSize = minWinSize;
  }

  if (wl.size_all () >= windowSize) {
    return; 
  }

  size_t fillSize = windowSize - wl.size_all ();

  size_t perThrdSize = fillSize / numT;

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

  // galois::runtime::LL::gDebug("size before refill: ", wl.size_all ());

  if (windowLimit != NULL) {
    // galois::runtime::LL::gDebug("new window limit: ", windowLimit->str ().c_str ());

    galois::on_each (RefillWorkList<T, I, WL> (windowLimit, ranges, wl), galois::loopname("refill"));

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

  // galois::runtime::LL::gDebug("size after refill: ", wl.size_all ());
}


struct UnionFindWindow {

  void operator () (
      EdgeCtxWL& perThrdWL, 
      VecRep_ty& repVec, 
      VecAtomicCtxPtr& repOwnerCtxVec, 
      size_t& mstWeight, 
      size_t& totalIter,
      galois::TimeAccumulator& sortTimer,
      galois::TimeAccumulator& findTimer,
      galois::TimeAccumulator& linkUpTimer,
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


    unsigned round = 0;
    size_t numUnions = 0;
    Accumulator mstSum;
    size_t prevWindowSize = 0;
    size_t numCommits = 0;

    EdgeCtxWL* currWL = new EdgeCtxWL ();
    EdgeCtxWL* nextWL = new EdgeCtxWL ();

    while (true) {
      ++round;
      prevWindowSize = currWL->size_all ();
      numCommits = prevWindowSize - nextWL->size_all ();
      std::swap (nextWL, currWL);
      nextWL->clear_all_parallel ();


      // size_t s = lowThreshSize - currWL->size_all () + 1;
      sortTimer.start ();
      refillWorkList<EdgeCtxWL, Iter> (*currWL, ranges, prevWindowSize, numCommits);
      sortTimer.stop ();

      if (currWL->empty_all ()) {
        break;
      }

      // galois::runtime::beginSampling ();
      findTimer.start ();
      galois::do_all_local (*currWL,
          FindLoop (repVec, repOwnerCtxVec, findIter),
          galois::do_all_steal<true>(),
          galois::loopname("find_loop"));
      findTimer.stop ();
      // galois::runtime::endSampling ();


      // galois::runtime::beginSampling ();
      linkUpTimer.start ();
      galois::do_all_local (*currWL,
          LinkUpLoop<false> (repVec, repOwnerCtxVec, *nextWL, mstSum, linkUpIter),
          galois::do_all_steal<true>(),
          galois::loopname("link_up_loop"));
      linkUpTimer.stop ();
      // galois::runtime::endSampling ();

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

  GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (const Edge& edge) const {
    wl.get ().push_back (edge);
  }
};



template <typename UF>
void runMSTsimple (const size_t numNodes, const VecEdge& edges, 
    size_t& mstWeight, size_t& totalIter, UF ufLoop) {

  totalIter = 0;
  mstWeight = 0;

  galois::TimeAccumulator runningTime;
  galois::TimeAccumulator sortTimer;
  galois::TimeAccumulator findTimer;
  galois::TimeAccumulator linkUpTimer;
  galois::TimeAccumulator fillUpTimer;

  Accumulator findIter;
  Accumulator linkUpIter;


  // VecRep_ty repVec (numNodes, -1);
  // VecAtomicCtxPtr repOwnerCtxVec (numNodes, AtomicCtxPtr (NULL));
  VecRep_ty repVec (numNodes);
  VecAtomicCtxPtr repOwnerCtxVec (numNodes);

 
  galois::substrate::getThreadPool().burnPower(galois::getActiveThreads());

  fillUpTimer.start ();
  galois::do_all (
      boost::counting_iterator<size_t>(0),
      boost::counting_iterator<size_t>(numNodes),
      [&repVec, &repOwnerCtxVec] (size_t i) {
        repVec.initialize (i, -1);
        repOwnerCtxVec.initialize (i, AtomicCtxPtr(nullptr));
      },
      galois::loopname ("init-vectors"));



  EdgeCtxWL initWL;
  unsigned numT = galois::getActiveThreads ();
  for (unsigned i = 0; i < numT; ++i) {
    initWL[i].reserve ((edges.size () + numT - 1) / numT);
  }

  galois::do_all (edges.begin (), edges.end (), 
      FillUp (initWL), 
      galois::loopname("fill_init"));

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

  galois::substrate::getThreadPool().beKind();
}

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

  galois::do_all (b, e, 
      Partition (pivot, lighter, heavier), 
      galois::loopname ("partition_loop"));

  assert ((lighter.size_all () + heavier.size_all ()) == old_sz + size_t (std::distance (b, e)));

  // TODO: print here
  std::printf ("total size = %ld, input index = %ld\n", std::distance (b, e), k);
  std::printf ("lighter size = %zd, heavier size = %zd\n", 
      lighter.size_all (), heavier.size_all ());

}

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

  GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (const Edge& edge) const {

    int rep1 = findPCiter_int (edge.src, repVec);
    int rep2 = findPCiter_int (edge.dst, repVec);

    if (rep1 != rep2) {
      filterWL.get ().push_back (edge);
    }
  }
};

template <typename UF>
void runMSTfilter (const size_t numNodes, const VecEdge& edges,
    size_t& mstWeight, size_t& totalIter, UF ufLoop) {

  totalIter = 0;
  mstWeight = 0;

  galois::TimeAccumulator runningTime;
  galois::TimeAccumulator partitionTimer;
  galois::TimeAccumulator sortTimer;
  galois::TimeAccumulator findTimer;
  galois::TimeAccumulator linkUpTimer;
  galois::TimeAccumulator filterTimer;

  Accumulator findIter;
  Accumulator linkUpIter;

  galois::substrate::getThreadPool().burnPower(galois::getActiveThreads());


  VecRep_ty repVec (numNodes);
  VecAtomicCtxPtr repOwnerCtxVec (numNodes);
  galois::do_all (
      boost::counting_iterator<size_t>(0),
      boost::counting_iterator<size_t>(numNodes),
      [&repVec, &repOwnerCtxVec] (size_t i) {
        repVec.initialize (i, -1);
        repOwnerCtxVec.initialize (i, AtomicCtxPtr(nullptr));
      },
      galois::loopname ("init-vectors"));




  runningTime.start ();

  partitionTimer.start ();

  EdgeCtxWL lighter;
  EdgeWL heavier;

  typedef std::iterator_traits<VecEdge::iterator>::difference_type Dist_ty;
  Dist_ty splitPoint = (4 * numNodes) / 3;
  lighter.reserve_all (numNodes / galois::getActiveThreads ());
  assert (edges.size () >= numNodes);
  heavier.reserve_all (numNodes / galois::getActiveThreads ());
  partition_edges (edges.begin (), edges.end (), splitPoint, lighter, heavier);

  partitionTimer.stop ();

  ufLoop (lighter, repVec, repOwnerCtxVec, 
      mstWeight, totalIter, sortTimer, findTimer, linkUpTimer, findIter, linkUpIter);


  // reuse lighter for filterWL
  lighter.clear_all_parallel ();

  filterTimer.start ();
  galois::do_all_local (heavier,
      FilterSelfEdges<EdgeCtxWL> (repVec, lighter),
      galois::do_all_steal<true>(),
      galois::loopname ("filter_loop"));
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

  galois::substrate::getThreadPool().beKind();
}
}// end namespace kruskal


#endif //  KRUSKAL_PARALLEL_H

