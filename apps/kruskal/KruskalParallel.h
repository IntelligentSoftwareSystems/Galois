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

#include "Galois/DoAllWrap.h"

#include "Kruskal.h"

namespace kruskal {



struct EdgeCtx;
template <typename T> struct Padded;

typedef VecRep VecRep_ty;
// typedef std::vector<Padded<int> > VecRep_ty;

typedef GaloisRuntime::PerThreadVector<Edge> EdgeWL;
typedef GaloisRuntime::PerThreadVector<EdgeCtx> EdgeCtxWL;
typedef GaloisRuntime::MM::FSBGaloisAllocator<EdgeCtx> EdgeCtxAlloc;
typedef Edge::Comparator Cmp;
typedef Galois::GAccumulator<size_t> Accumulator;
typedef Galois::GAtomicPadded<EdgeCtx*> AtomicCtxPtr;
// typedef Galois::GAtomic<int> AtomicInt;
typedef std::vector<AtomicCtxPtr> VecAtomicCtxPtr;

static const int NULL_EDGE_ID = -1;

template <typename T>
struct Padded {
  GALOIS_ATTRIBUTE_ALIGN_CACHE_LINE T data;

  Padded (const T& x): data (x) {}

  operator T& () { return data; }

  operator T () const { return data; }

};


struct EdgeCtx: public Edge {
  bool srcPass;
  bool dstPass;

  explicit EdgeCtx (const Edge& e): Edge (e), srcPass (true), dstPass (true)
  {}

  void updatePass (const int rep, VecRep_ty& repVec) {

    if (getRep_int (this->src, repVec) == rep) {
      srcPass = false;

    } else if (getRep_int (this->dst, repVec) == rep) {
      dstPass = false;

    } else {
      abort ();
    }
  }

  void resetPass () {
    srcPass = true;
    dstPass = true;
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
  EdgeCtxAlloc& alloc;
  EdgeCtxWL& lighter;
  EdgeWL& heavier;

  Partition (
      const Edge& pivot,
      EdgeCtxAlloc& alloc,
      EdgeCtxWL& lighter,
      EdgeWL& heavier)
    :
      pivot (pivot),
      alloc (alloc),
      lighter (lighter),
      heavier (heavier)
  {}


  GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (const Edge& edge) const {

    if (Cmp::compare (edge, pivot) < 0) { // edge < pivot
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
    EdgeCtxAlloc& alloc, EdgeCtxWL& lighter, EdgeWL& heavier) {

  assert (b != e);

  Edge pivot = pick_kth (b, e, k);

  Galois::do_all_choice (b, e, Partition (pivot, alloc, lighter, heavier), "partition_loop");

  assert ((lighter.size_all () + heavier.size_all ()) == size_t (std::distance (b, e)));

  // TODO: print here
  std::printf ("total size = %ld, input index = %ld\n", std::distance (b, e), k);
  std::printf ("lighter size = %zd, heavier size = %zd\n", 
      lighter.size_all (), heavier.size_all ());

}



struct FindLoop {

  // typedef char tt_does_not_need_parallel_push;

  VecRep_ty& repVec;
  VecAtomicCtxPtr& minEdgeCtxVec;
  Accumulator& findIter;

  FindLoop (
      VecRep_ty& repVec,
      VecAtomicCtxPtr& minEdgeCtxVec,
      Accumulator& findIter)
    :
      repVec (repVec),
      minEdgeCtxVec (minEdgeCtxVec),
      findIter (findIter)
  {}

  GALOIS_ATTRIBUTE_PROF_NOINLINE void claimAsMin (EdgeCtx& ctx, const int rep) {

    assert (rep >= 0 && size_t (rep) < minEdgeCtxVec.size ());

    bool succ = minEdgeCtxVec[rep].cas (NULL, &ctx);

    assert (minEdgeCtxVec[rep] != NULL);

    for (EdgeCtx* curr = minEdgeCtxVec[rep];
        !succ && Cmp::compare (*curr, ctx) > 0; curr = minEdgeCtxVec[rep]) {

      succ = minEdgeCtxVec[rep].cas (curr, &ctx);

      if (succ) {
        curr->updatePass (rep, repVec);
      }
    }

    if (!succ) {
      ctx.updatePass (rep, repVec);
    }


  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (EdgeCtx& ctx) {
    findIter += 1;

    int rep1 = kruskal::findPCiter_int (ctx.src, repVec);
    int rep2 = kruskal::findPCiter_int (ctx.dst, repVec);
    
    if (rep1 != rep2) {
      claimAsMin (ctx, rep1);
      claimAsMin (ctx, rep2);

    } else {
      // self-edge, disable for early cleanup
      ctx.srcPass = false;
      ctx.dstPass = false;
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
  VecAtomicCtxPtr& minEdgeCtxVec;
  EdgeCtxWL& nextWL;
  Accumulator& mstSum;
  Accumulator& linkUpIter;

  LinkUpLoop (
      VecRep_ty& repVec,
      VecAtomicCtxPtr& minEdgeCtxVec,
      EdgeCtxWL& nextWL,
      Accumulator& mstSum,
      Accumulator& linkUpIter)
    :
      repVec (repVec),
      minEdgeCtxVec (minEdgeCtxVec),
      nextWL (nextWL),
      mstSum (mstSum),
      linkUpIter (linkUpIter)
  {}


  GALOIS_ATTRIBUTE_PROF_NOINLINE bool updateODG_test (EdgeCtx& ctx, const int rep) {
    assert (rep >= 0 && size_t (rep) < minEdgeCtxVec.size ());

    return (minEdgeCtxVec[rep] == &ctx);
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void updateODG_reset (EdgeCtx& ctx, const int rep) {

    assert (rep >= 0 && size_t (rep) < minEdgeCtxVec.size ());
    assert (updateODG_test (ctx, rep));

    minEdgeCtxVec[rep] = NULL;
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (EdgeCtx& ctx) {

    // relies on find with path-compression
    int rep1 = getRep_int (ctx.src, repVec);
    int rep2 = getRep_int (ctx.dst, repVec);

    if (ctx.srcPass || ctx.dstPass) {

      if (ctx.srcPass) {
        assert (updateODG_test (ctx, rep1));
        linkUp_int (rep1, rep2, repVec);

      } else if (ctx.dstPass) {
        assert (updateODG_test (ctx, rep2));
        linkUp_int (rep2, rep1, repVec);
      }

      linkUpIter += 1;
      mstSum += ctx.weight;

      if (ctx.srcPass) {
        updateODG_reset (ctx, rep1);
      }

      if (ctx.dstPass) {
        updateODG_reset (ctx, rep2);
      }

      // TODO: deallocate

    } else if (rep1 != rep2) {

      ctx.resetPass ();

      if (usingOrderedRuntime) {
        GaloisRuntime::signalConflict ();

      } else {
        nextWL.get ().push_back (ctx);
      }

    }  else {
      // TODO: deallocate
    }


    // if (rep1 != rep2) {
// 
      // bool succ1 = updateODG_test (edge, rep1);
      // bool succ2 = updateODG_test (edge, rep2);
// 
      // if (succ1) {
        // linkUp_int (rep1, rep2, repVec);
// 
      // } else if (succ2) {
        // linkUp_int (rep2, rep1, repVec);
// 
      // } else {
        // if (usingOrderedRuntime) {
          // GaloisRuntime::signalConflict ();
        // } else {
          // nextWL.get ().push_back (edge);
        // }
      // }
// 
// 
      // if (succ1 || succ2) {
        // linkUpIter += 1;
        // mstSum += edge.weight;
// 
      // }
// 
      // if (succ1) {
        // updateODG_reset (edge, rep1);
      // }
// 
      // if (succ2) {
        // updateODG_reset (edge, rep2);
      // }
    // }
    
  }

  template <typename C>
  GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (EdgeCtx& ctx, C&) {
    (*this) (ctx);
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (EdgeCtx& ctx) const {
    const_cast<LinkUpLoop*>(this)->operator () (ctx);
  }
};

struct FilterSelfEdges {
  VecRep_ty& repVec;
  EdgeCtxAlloc& alloc;
  EdgeCtxWL& filterWL;

  FilterSelfEdges (
      VecRep_ty& repVec, 
      EdgeCtxAlloc& alloc,
      EdgeCtxWL& filterWL)
    : 
      repVec (repVec), 
      alloc (alloc),
      filterWL (filterWL) 
  {}

  GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (const Edge& edge) {

    int rep1 = findPCiter_int (edge.src, repVec);
    int rep2 = findPCiter_int (edge.dst, repVec);

    if (rep1 != rep2) {
      // EdgeCtx* ctx = alloc.allocate (1);
      // assert (ctx != NULL);
      // alloc.construct (ctx, EdgeCtx (edge));
      EdgeCtx ctx (edge);

      filterWL.get ().push_back (ctx);
    }
  }
};


template <typename UF>
void runMSTparallel (const size_t numNodes, const VecEdge& edges,
    size_t& mstWeight, size_t& totalIter, UF ufLoop) {

  totalIter = 0;
  mstWeight = 0;

  Galois::TimeAccumulator runningTime;
  Galois::TimeAccumulator partitionTimer;
  Galois::TimeAccumulator findTimer;
  Galois::TimeAccumulator linkUpTimer;
  Galois::TimeAccumulator filterTimer;

  Accumulator findIter;
  Accumulator linkUpIter;



  VecRep_ty repVec (numNodes, -1);
  VecAtomicCtxPtr minEdgeCtxVec (numNodes, AtomicCtxPtr (NULL));
  EdgeCtxAlloc alloc;



  Galois::preAlloc (16*Galois::getActiveThreads ());

  runningTime.start ();

  partitionTimer.start ();
  EdgeCtxWL lighter;
  EdgeWL heavier;

  typedef std::iterator_traits<VecEdge::iterator>::difference_type Dist_ty;
  Dist_ty splitPoint = numNodes;
  partition_edges (edges.begin (), edges.end (), splitPoint, alloc, lighter, heavier);

  partitionTimer.stop ();

  ufLoop (lighter, repVec, minEdgeCtxVec, 
      mstWeight, totalIter, findTimer, linkUpTimer, findIter, linkUpIter);


  // reuse lighter for filterWL
  lighter.clear_all ();

  filterTimer.start ();
  Galois::do_all_choice (heavier,
  // GaloisRuntime::do_all_coupled (heavier, 
      FilterSelfEdges (repVec, alloc, lighter),
      "filter_loop");
  filterTimer.stop ();


  ufLoop (lighter, repVec, minEdgeCtxVec, 
      mstWeight, totalIter, findTimer, linkUpTimer, findIter, linkUpIter);
  runningTime.stop ();

  std::cout << "Number of FindLoop iterations = " << findIter.reduce () << std::endl;
  std::cout << "Number of LinkUpLoop iterations = " << linkUpIter.reduce () << std::endl;

  std::cout << "MST running time without initialization/destruction: " << runningTime.get () << std::endl;
  std::cout << "Time taken by FindLoop: " << findTimer.get () << std::endl;
  std::cout << "Time taken by LinkUpLoop: " << linkUpTimer.get () << std::endl;
  std::cout << "Time taken by partitioning Loop: " << partitionTimer.get () << std::endl;
  std::cout << "Time taken by filter Loop: " << filterTimer.get () << std::endl;

}


}// end namespace kruskal



#endif //  KRUSKAL_PARALLEL_H

